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
``ESMFOLD2_SS_MAX_CONTAINERS``. The full grid is six targets by three arms by
thirty seeds, so the cap is what decides whether that takes an hour or most of
a day. Lower it if your Modal workspace has a smaller GPU quota than the
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
"""

import os
from typing import Any

import modal

from analysis import loss_trace
from analysis.targets import DEFAULT_TARGET, SGD_STEPS
from analysis.targets import arm as get_arm

APP_NAME = os.environ.get("ESMFOLD2_SS_APP_NAME", "esmfold2-ss-steering")
MAX_CONTAINERS = int(os.environ.get("ESMFOLD2_SS_MAX_CONTAINERS", "30"))
DEFAULT_ARM = "vanilla"
BINDER_NAME = "minibinder"
PRIMARY_CRITIC = "ESMFold2-Experimental-Fast-Cutoff2025"


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
        .pip_install(
            "abnumber", "esm@git+https://github.com/Biohub/esm.git@main", "modal"
        )
        .env(
            {
                "HF_HOME": "/models",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            }
        )
    )


image = (
    get_base_image()
    .pip_install("pandas", "pydssp")
    .add_local_python_source("analysis")
)

app = modal.App(
    name=APP_NAME,
    image=image,
    volumes={
        "/models": modal.Volume.from_name("esmfold2-models", create_if_missing=True)
    },
)


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

    @modal.enter()
    def load(self) -> None:
        from analysis.design.loop import ESMFold2Design

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
    ) -> dict[str, Any]:
        """Run one binder design; return metrics, the complex, and the loss log.

        ``target`` is a label only. ``target_sequence`` is the construct the
        binder is designed against, resolved by the client.

        ``arm`` names a loss configuration in :mod:`analysis.targets`, which
        resolves to the steering weight schedule the design loop applies. The
        default arm carries no steering term, so it reproduces the cookbook
        objective.

        ``steps`` is applied by rebinding ``analysis.design.loop.STEPS``, which
        is the global ``design_binder`` reads for its step count.

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
        design_loop.STEPS = int(steps)
        _, trajectory, critic_results, trace = self.driver.design(
            target_name=target,
            binder_name=BINDER_NAME,
            target_sequence=target_sequence,
            seed=int(seed),
            batch_size=1,
            schedule=arm_spec.schedule,
        )

        by_critic = {record["critic_name"]: record for record in critic_results}
        primary = by_critic[PRIMARY_CRITIC]
        designed_sequence = primary["designed_sequence"]

        return {
            "seed": int(seed),
            "steps": int(steps),
            "target": target,
            "arm": arm_spec.key,
            "arm_label": arm_spec.label,
            "schedule": arm_spec.schedule.to_dict(),
            "target_length_residues": len(target_sequence),
            "binder_name": BINDER_NAME,
            "designed_sequence": designed_sequence,
            "binder_sequence": designed_sequence.split("|")[-1],
            "binder_length_residues": len(designed_sequence.split("|")[-1]),
            "final_loss": primary["final_loss"],
            "critic_name": PRIMARY_CRITIC,
            "critic_iptm": primary["iptm"],
            "critic_plddt_binder": primary["plddt_binder"],
            "critic_plddt_target": primary["plddt_target"],
            "critic_plddt_complex": primary["plddt_complex"],
            "critic_distogram_iptm_proxy": primary["distogram_iptm_proxy"],
            "critic_iptm_by_model": {
                name: record["iptm"] for name, record in by_critic.items()
            },
            "critic_plddt_binder_by_model": {
                name: record["plddt_binder"] for name, record in by_critic.items()
            },
            "critic_distogram_iptm_proxy_by_model": {
                name: record["distogram_iptm_proxy"]
                for name, record in by_critic.items()
            },
            "design_steps_recorded": len(trajectory),
            "step_trace": loss_trace.rows(trace),
            "complex_pdb": primary["complex"].to_pdb_string(),
        }


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
