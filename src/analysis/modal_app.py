"""Modal app for ESMFold2 binder design, one design per remote call.

The design code is :mod:`analysis.design`, a verbatim port of the ESM cookbook
binder-design tutorial. This module is only the deployment shell: it builds the
container image, holds the loaded models for the lifetime of a container, and
turns one design into a JSON-friendly result the client can store.

Deploy:
    modal deploy src/analysis/modal_app.py

The app name defaults to ``esmfold2-ss-steering`` and can be overridden with
``ESMFOLD2_SS_APP_NAME`` so several people can deploy side by side.

The image carries no target data. ``design`` takes the target sequence as a
string, which the client reads from ``data/targets`` via
:meth:`analysis.targets.TargetSpec.sequence`. Registering a new target
therefore does not change the image and does not force a redeploy. ``target``
is a label, recorded in the returned metrics and never resolved to a path
inside the container.

``MAX_CONTAINERS`` caps how many designs run at once and is read from
``ESMFOLD2_SS_MAX_CONTAINERS``. The published grids come to well over a
thousand designs, so the cap is what decides whether that takes hours or most
of a day. Lower it if your Modal workspace has a smaller GPU quota than the
concurrency you ask for here.

``get_base_image`` is verbatim from ``cookbook/tutorials/binder_design.py`` at
upstream commit ``ba4d712``, lines 1182-1231. Two additions sit on top of it:
``pandas`` and ``pydssp``, which the ``analysis`` package imports at package
import time, and the local source of that package itself. Both are runtime
mounts or thin wheels; neither carries data.

Unlike upstream's ``ESMFold2DesignModal``, the Modal class here holds an
``ESMFold2Design`` rather than subclassing it, and imports it inside
:meth:`load`. That keeps ``modal deploy`` runnable from an environment that has
only ``modal`` installed, without ``esm`` or the Biohub ``transformers`` fork.

The same design worker runs prior-initialized designs and paired trajectories.
Paired runs disable critic loading. ``FixedInputModal`` shares the app and image
but loads one uncompiled model for the zero-update screen. Model outputs and
provenance from these experiments are preserved in the app's run volume.
"""

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import modal

from analysis import loss_trace
from analysis.results import CampaignOutput, DesignResult, RunSpec
from analysis.targets import DEFAULT_TARGET, INITIALIZATION_SPECS, SGD_STEPS
from analysis.targets import arm as get_arm

APP_NAME = os.environ.get("ESMFOLD2_SS_APP_NAME", "esmfold2-ss-steering")
MAX_CONTAINERS = int(os.environ.get("ESMFOLD2_SS_MAX_CONTAINERS", "30"))
ESMC_REPO = "biohub/ESMC-6B"

ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
"""The ESMC trunk revision to load, pinned because ``main`` cannot be read.

This is the revision every campaign in this publication was designed against.
``main`` pointed at it from 2026-06-04 until 2026-09-15, which spans every
campaign run.

``main`` ships checkpoint keys named for the ``esm`` SDK's own module tree,
without the ``esmc.`` prefix that ``transformers``' ESMC module expects. Nothing
raises: ``from_pretrained`` reports the mismatch as a warning and randomly
initialises every parameter it cannot match, which is the entire trunk.
``REUSE_ESMC`` then shares that trunk across all six ESMFold2 models, so the
first thing to actually fail is a downstream SVD on an all-NaN matrix.

Biohub/esm#333 asked for exactly this republishing, to fix loading through the
SDK. It broke loading through ``transformers``, which is the path this app uses.
"""


def pin_esmc_revision(revision: str) -> None:
    """Load a named revision of the ESMC trunk.

    Both entry points inherit ``from_pretrained`` from ``ESMCPreTrainedModel``:
    the design loop builds an ``ESMCForMaskedLM`` for its sequence loss, and
    each ESMFold2 model builds an ``ESMCModel`` inside ``load_esmc``. The
    revision is injected here because both call sites are ported code this
    repository reproduces verbatim.

    The replacement has to be a ``classmethod`` over the underlying function.
    Binding the base class instead drops the caller's ``cls``, so every
    subclass call returns a bare ``ESMCPreTrainedModel``.
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
        .pip_install("abnumber", "esm@git+https://github.com/Biohub/esm.git@ba4d712", "modal")
        .env(
            {
                "HF_HOME": "/models",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            }
        )
    )


image = get_base_image().pip_install("pandas", "pydssp").add_local_python_source("analysis")

study = modal.Volume.from_name(f"{APP_NAME}-runs", create_if_missing=True)

app = modal.App(
    name=APP_NAME,
    image=image,
    volumes={
        "/models": modal.Volume.from_name("esmfold2-models", create_if_missing=True),
        "/study": study,
    },
)


def _pack_structures(structures: tuple[str, ...]) -> bytes:
    """Per-step complexes as one gzipped JSON array, empty when none were logged.

    A 150-step run holds tens of megabytes of PDB text, which is coordinates
    written out as decimal digits and compresses by roughly an order of
    magnitude. The payload is a function return value crossing a network, so it
    is worth not sending the uncompressed form.

    JSON rather than the records concatenated with a delimiter, because PDB text
    already contains newlines and ends with a record that also appears inside
    multi-model files, so every delimiter worth choosing is one the payload can
    contain. An array needs no delimiter at all.
    """
    if not structures:
        return b""
    return gzip.compress(json.dumps(list(structures)).encode(), mtime=0)


def unpack_structures(blob: bytes) -> list[str]:
    """The per-step complexes a design returned, in step order.

    Lives beside the packing so the two cannot drift apart, and is called by the
    client rather than in the container.
    """
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
    """One container per design, holding the loaded inversion and critic models.

    Hero critics are the four HuggingFace experimental exports with confidence
    heads. ``use_scaling_critics`` also loads the 15-checkpoint scaling
    ensemble, which contributes distogram scores only. It defaults to off here
    because this publication's campaigns never read those scores and the
    checkpoints cost load time and memory. Upstream defaults it on.
    """

    use_scaling_critics: bool = modal.parameter(default=False)
    load_critics: bool = modal.parameter(default=True)

    @modal.enter()
    def load(self) -> None:
        from analysis.design.loop import ESMFold2Design

        pin_esmc_revision(ESMC_REVISION)

        self.driver = ESMFold2Design()
        if not self.load_critics:
            self.driver.hero_critic_hf_paths = []
        self.driver.load(self.use_scaling_critics)
        self.sources, self.audit = runtime_provenance()
        self.audit.update(
            models={n: m.config.to_dict() for n, m in self.driver.inversion_models.items()},
            torch_compile=True,
        )

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
        """Run one binder design; return metrics, the complex, and the loss log.

        ``target`` is a label only. ``target_sequence`` is the construct the
        binder is designed against, resolved by the client.

        ``arm`` names a loss configuration in :mod:`analysis.targets`, which
        resolves to the steering weight schedule the design loop applies. The
        default arm carries no steering term, so it reproduces the cookbook
        objective.

        ``steps`` sets the annealing horizon. ``stop_after_steps`` optionally
        truncates optimization while preserving that horizon and skips final
        sequence selection and critic evaluation for a truncated run.
        ``snapshot_sampling_steps`` samples logged structures from saved soft
        states after optimization, using a separate solver budget.

        ``sampling_steps`` sets how many solver steps the structure module takes
        on every step of the run, confidence steps included, so a logged
        trajectory is sampled the same way throughout. Zero means leave the
        cookbook's split alone, which is one step ordinarily and fifty on a
        confidence step. ``log_step_structures`` returns every step's predicted
        complex under ``step_complexes_gz``, compressed because 150 all-atom
        complexes are tens of megabytes as text and the payload crosses a
        network.

        The returned ``step_trace`` is one row per optimization step: every
        loss, every steering weight in force, the per-step binding readout and
        the sequence entropy. It is the log the steering analysis reads, and it
        is a few tens of kilobytes for a 150-step run.
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

    @modal.method()
    def design_prior(self, bundle: dict, protocol: dict) -> bytes:
        from analysis.design.initialization import design_prior

        if not self.load_critics:
            raise ValueError("Full designs require the critic models")
        return design_prior(self.driver, self.audit, self.sources, bundle, protocol, study)

    @modal.method()
    def design_pair(self, bundle: dict, protocol: dict) -> bytes:
        from analysis.design.initialization import design_pair

        return design_pair(self.driver, self.audit, self.sources, bundle, protocol, study)


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


def runtime_provenance() -> tuple[dict, dict]:
    """Record model implementation and environment once for each worker container."""
    import importlib.metadata
    import inspect

    import torch
    from transformers.models.esmfold2 import modeling_esmfold2_common as common
    from transformers.models.esmfold2 import modeling_esmfold2_experimental as experimental

    sources = {
        Path(inspect.getfile(m)).name: Path(inspect.getfile(m)).read_bytes()
        for m in (common, experimental)
    }
    audit = {
        "esmc_revision": ESMC_REVISION,
        "packages": {
            n: importlib.metadata.version(n) for n in ("torch", "transformers", "esm", "pydssp")
        },
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in sources.items()},
    }
    for name in ("torch", "transformers", "esm"):
        direct = importlib.metadata.distribution(name).read_text("direct_url.json")
        if direct:
            audit[name + "_direct_url"] = json.loads(direct)
    return sources, audit


@app.function(timeout=600, cpu=2)
def prepare_screen_inputs() -> dict:
    from analysis.design.initialization import prepare_screen_inputs as prepare

    return prepare()


@app.cls(gpu="H100", timeout=3600, cpu=8, memory=64 * 1024, max_containers=5)
class FixedInputModal:
    """Single uncompiled model for the zero-update screen and native controls."""

    @modal.enter()
    def load(self):
        import inspect

        from analysis.design.models import _load_hf_model

        checkpoint = INITIALIZATION_SPECS["softness"]["checkpoint"]
        pin_esmc_revision(ESMC_REVISION)
        self.model = _load_hf_model(checkpoint, 0.5, True, "cuda")
        self.sources, self.audit = runtime_provenance()
        self.audit.update(
            checkpoint=checkpoint,
            model_config=self.model.config.to_dict(),
            esmc_config=self.model._esmc.config.to_dict(),
            parameter_dtype=str(next(self.model.parameters()).dtype),
            torch_compile=False,
            forward_signature=str(inspect.signature(self.model.forward)),
        )

    @modal.method()
    def screen(self, bundle: bytes, target_sequence: str, protocol_id: str) -> bytes:
        from analysis.design.initialization import screen

        return screen(
            self.model, self.audit, self.sources, bundle, target_sequence, protocol_id, study
        )

    @modal.method()
    def controls(self, controls: list[dict], protocol_id: str) -> bytes:
        from analysis.design.initialization import controls as run_controls

        return run_controls(self.model, self.audit, controls, protocol_id, study)
