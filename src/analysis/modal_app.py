"""Deploy ESMFold2 binder design on Modal, with one design per remote call.

Clients supply target sequences. Containers retain loaded models between calls.
``ESMFOLD2_SS_APP_NAME`` and ``ESMFOLD2_SS_MAX_CONTAINERS`` configure the app.
Local deployment needs Modal and the analysis package's CPU dependencies;
ESM and the Biohub transformers fork are loaded inside the GPU container.
"""

import gzip
import json
import os
from pathlib import Path
from typing import Any

import modal

from analysis import loss_trace
from analysis.results import CampaignOutput, DesignResult, RunSpec
from analysis.targets import DEFAULT_TARGET, SGD_STEPS
from analysis.targets import arm as get_arm

APP_NAME = os.environ.get("ESMFOLD2_SS_APP_NAME", "esmfold2-ss-steering")
MAX_CONTAINERS = int(os.environ.get("ESMFOLD2_SS_MAX_CONTAINERS", "30"))
ESM_REVISION = "ba4d7124864eed323a93bf3cfefcd958f573b75a"
TRANSFORMERS_REVISION = "ef32577f55da19a4989cd7b22e004dc43a4998cb"
ESMC_REPO = "biohub/ESMC-6B"

ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
"""Pinned ESMC checkpoint revision used by the design container."""


def pin_esmc_revision(revision: str) -> None:
    """Default ESMC checkpoint loads to ``revision`` unless explicitly overridden.

    Patch the inherited classmethod so model subclasses retain their own type.
    """
    from transformers.models.esmc.modeling_esmc import ESMCPreTrainedModel

    unpinned = ESMCPreTrainedModel.from_pretrained.__func__

    def pinned(cls: Any, repo_id: str, *args: Any, **kwargs: Any) -> Any:
        if repo_id == ESMC_REPO and "revision" not in kwargs:
            kwargs["revision"] = revision
        return unpinned(cls, repo_id, *args, **kwargs)

    ESMCPreTrainedModel.from_pretrained = classmethod(pinned)


DEFAULT_ARM = "vanilla"
BINDER_NAME = "minibinder"


def get_base_image():
    return (
        modal.Image.micromamba(python_version="3.12")
        .run_commands("apt update && apt install -y git build-essential")
        .micromamba_install(
            "anarci>=2020.04.03",
            "hmmer=3.4",
            "cuda-version=12.8",
            "cuda-libraries-dev=12.8",
            "cuda-nvcc=12.8",
            "cmake",
            "ninja",
            channels=["conda-forge", "bioconda"],
        )
        .pip_install(
            "torch==2.8.0",
            "triton==3.4.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "flash-attn==2.8.3",
            "transformer-engine[core-cu12,pytorch]==2.13.0",
            "xformers==0.0.32.post1",
            extra_options="--no-build-isolation",
            env={
                "CPATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/include"
                ),
                "LIBRARY_PATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/lib"
                ),
                "MAX_JOBS": "8",
                "NVTE_FRAMEWORK": "pytorch",
            },
        )
        .pip_install("abnumber", "modal")
        .add_local_file(
            Path(__file__).with_name("gpu_dependencies.py"),
            "/tmp/install_legacy_esm.py",
            copy=True,
        )
        .run_commands(f"python /tmp/install_legacy_esm.py {ESM_REVISION} {TRANSFORMERS_REVISION}")
        .env(
            {
                "HF_HOME": "/models",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            }
        )
    )


image = get_base_image().pip_install("pandas", "pydssp").add_local_python_source("analysis")

app = modal.App(
    name=APP_NAME,
    image=image,
    volumes={
        "/models": modal.Volume.from_name("esmfold2-models", create_if_missing=True),
    },
)


def _pack_structures(structures: tuple[str, ...]) -> bytes:
    """Encode per-step PDB strings as gzipped JSON bytes, or empty bytes if absent."""
    if not structures:
        return b""
    return gzip.compress(json.dumps(list(structures)).encode(), mtime=0)


def unpack_structures(blob: bytes) -> list[str]:
    """Decode per-step PDB strings in order; empty bytes yield an empty list."""
    if not blob:
        return []
    return json.loads(gzip.decompress(blob).decode())


@app.cls(
    gpu="H100",
    timeout=60 * 60,
    cpu=16,
    memory=80 * 1024,
    max_containers=MAX_CONTAINERS,
)
class ESMFold2DesignModal:
    """Keep loaded design models in a GPU container.

    Optionally load scaling critics in addition to the four hero critics.
    """

    use_scaling_critics: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self) -> None:
        from analysis.design.loop import ESMFold2Design

        pin_esmc_revision(ESMC_REVISION)

        self.driver = ESMFold2Design()
        self.driver.load(self.use_scaling_critics)

    @modal.method()
    def design(
        self,
        seed: int = 0,
        steps: int = SGD_STEPS,
        target: str = DEFAULT_TARGET,
        target_sequence: str = "",
        arm: str = DEFAULT_ARM,
        sampling_steps: int = 0,
        log_step_structures: bool = False,
        stop_after_steps: int | None = None,
        snapshot_sampling_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run a design against the supplied target sequence and registered loss arm.

        Return scores, the final complex, and per-step trace rows. ``sampling_steps=0``
        uses the loop's separate ordinary and confidence budgets; a nonzero value
        sets both. ``log_step_structures`` adds gzipped JSON bytes of per-step PDBs.

        ``steps`` sets the annealing horizon. A shorter ``stop_after_steps`` returns
        diagnostics without final critic scores or a selected final complex.
        ``snapshot_sampling_steps`` resamples logged structures after optimization
        and requires ``log_step_structures``.
        """
        from analysis.design import loop as design_loop

        if not target_sequence:
            raise ValueError(
                "target_sequence is required. Resolve it on the client with "
                "analysis.targets.target(key).sequence()."
            )

        arm_spec = get_arm(arm)
        solver_steps = sampling_steps or design_loop.SAMPLING_STEPS
        _, trajectory, critic_results, trace, step_structures = self.driver.design(
            target_name=target,
            binder_name=BINDER_NAME,
            target_sequence=target_sequence,
            seed=int(seed),
            batch_size=1,
            schedule=arm_spec.schedule,
            steps=int(steps),
            sampling_steps=solver_steps,
            confidence_sampling_steps=sampling_steps or design_loop.CONFIDENCE_SAMPLING_STEPS,
            log_step_structures=log_step_structures,
            stop_after_steps=stop_after_steps,
            snapshot_sampling_steps=snapshot_sampling_steps,
        )

        if stop_after_steps is not None and stop_after_steps < steps:
            return {
                "seed": int(seed),
                "steps": int(steps),
                "completed_steps": len(trace),
                "stopped_early": True,
                "target": target,
                "arm": arm_spec.key,
                "schedule": arm_spec.schedule.to_dict(),
                "sampling_steps": solver_steps,
                "snapshot_sampling_steps": snapshot_sampling_steps,
                "step_trace": loss_trace.rows(trace),
                "step_complexes_gz": _pack_structures(step_structures),
            }

        result = DesignResult.from_critics(
            RunSpec(target=target, arm=arm_spec.key, seed=int(seed), steps=int(steps)),
            critic_results,
            binder_name=BINDER_NAME,
        )
        primary = next(c for c in critic_results if c["critic_name"] == result.primary.name)
        output = CampaignOutput.from_design(
            result,
            loss_trace.rows(trace),
            primary["complex"].to_pdb_string(),
            _pack_structures(step_structures),
        )
        return output.to_payload(schedule=arm_spec.schedule.to_dict())


@app.local_entrypoint()
def main(
    seed: int = 0,
    steps: int = 2,
    target: str = DEFAULT_TARGET,
    arm: str = DEFAULT_ARM,
):
    """Smoke-test one short design against a registered target and arm."""
    from analysis.targets import target as get_target

    spec = get_target(target)
    result = ESMFold2DesignModal().design.remote(
        seed=seed,
        steps=steps,
        target=spec.key,
        target_sequence=spec.sequence(),
        arm=arm,
    )
    print(f"seed={seed} steps={steps} target={result['target']} arm={result['arm']}")
    print(f"schedule={result['schedule']['summary']}")
    print(f"binder_length_residues={result['binder_length_residues']}")
    print(f"critic_iptm={result['critic_iptm']}")
    print(f"critic_plddt_binder={result['critic_plddt_binder']}")
    print(f"step_trace_rows={len(result['step_trace'])}")
    print(f"pdb_chars={len(result['complex_pdb'])}")
