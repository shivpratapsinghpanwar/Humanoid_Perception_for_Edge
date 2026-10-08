"""SmolVLA-base on the CPU: how long does one observation -> one action chunk take?

This row is a LATENCY PROBE. The base checkpoint was trained on other robots' arms; its
actions mean nothing on a humanoid upper body and are never applied here. What is measured
is the cost the robot would pay per chunk once a fine-tuned checkpoint exists.

The call path is LeRobot's own inference path (lerobot_eval.py), so the number includes
everything the robot pays: the pre-processor pipeline (tokenising the instruction,
normalising the state, moving tensors), the policy's image resize-with-padding to 512x512,
the SmolVLM2-500M backbone, the flow-matching action expert (``num_steps`` denoising
steps) for a ``chunk_size``-step chunk, and the post-processor (un-normalising actions).
A second, model-only number (``predict_action_chunk`` on an already pre-processed batch)
is recorded alongside.

Input is what our robot would give it: the Gemini E frame rendered by the standing
Tiangong's head camera (640x480), a proprioceptive state vector of the width the
checkpoint declares, and one instruction from the task suite. Image keys and state width
are read from the loaded policy's ``input_features``, not assumed.
"""

from __future__ import annotations

import time

import numpy as np

from ..checkpoints import fetch, revision_of
from ..machine import capture
from ..results import CensusResult
from ..timing import measure, peak_rss_mb

REPO = "lerobot/smolvla_base"
INSTRUCTION = "look at the red cup"


def _load():
    """Policy + its pre/post-processors, exactly as lerobot_eval.py builds them, on the CPU."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = str(fetch(REPO))
    policy = SmolVLAPolicy.from_pretrained(path)
    policy.config.device = "cpu"
    policy.to("cpu").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post


def _raw_observation(config, rgb: np.ndarray) -> tuple[dict, list[str], int]:
    """The dict LeRobot's evaluator would hand the pre-processor: batched float images in
    [0, 1] under the checkpoint's own image keys, a state of the declared width, the task."""
    import torch

    img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    obs: dict = {}
    image_keys: list[str] = []
    state_dim = 0
    for key, feat in config.input_features.items():
        kind = str(getattr(getattr(feat, "type", None), "name", feat)).upper()
        if "VISUAL" in kind:
            obs[key] = img.clone()
            image_keys.append(key)
        elif "STATE" in kind:
            state_dim = int(np.prod(feat.shape))
            obs[key] = torch.zeros(1, state_dim)
    obs["task"] = INSTRUCTION
    return obs, image_keys, state_dim


def run(n: int = 20, warmup: int = 3, threads: int = 4, precision: str = "fp32") -> CensusResult:
    import torch

    if precision != "fp32":
        raise NotImplementedError("SmolVLA on torch-cpu is measured in fp32 here; int8/GGUF rows come from other runtimes")
    torch.set_num_threads(threads)

    from humanoid_perception.sim.stand import StandingSim
    sim = StandingSim()
    rgb = sim.render().rgb
    sim.close()

    t0 = time.perf_counter()
    policy, pre, post = _load()
    load_s = time.perf_counter() - t0
    config = policy.config
    obs, image_keys, state_dim = _raw_observation(config, rgb)

    def full():
        with torch.inference_mode():
            batch = pre(dict(obs))
            chunk = policy.predict_action_chunk(batch)
            return post(chunk)

    with torch.inference_mode():
        fixed_batch = pre(dict(obs))

    def model_only():
        with torch.inference_mode():
            return policy.predict_action_chunk(fixed_batch)

    actions = full()
    chunk_shape = tuple(actions.shape) if hasattr(actions, "shape") else None
    timing = measure(full, n=n, warmup=warmup)
    timing_model = measure(model_only, n=n, warmup=1)

    revision = revision_of(REPO)

    return CensusResult(
        model="smolvla_base", role="vla", source=REPO, revision=revision, licence="Apache-2.0",
        runtime="torch-cpu", precision=precision, threads=threads,
        what_is_timed="observation -> 50-action chunk, LeRobot pre/post-processors included",
        input_shape={"image_hw": [int(rgb.shape[0]), int(rgb.shape[1])], "image_keys": image_keys,
                     "state_dim": state_dim, "instruction": INSTRUCTION, "chunk_shape": chunk_shape,
                     "chunk_size": getattr(config, "chunk_size", None),
                     "n_action_steps": getattr(config, "n_action_steps", None),
                     "flow_steps": getattr(config, "num_steps", None),
                     "resize_imgs_with_padding": list(getattr(config, "resize_imgs_with_padding", ()) or [])},
        timing=timing, peak_rss_mb=peak_rss_mb(), machine=capture(),
        extra={"model_only_ms": timing_model.to_dict(), "load_seconds": round(load_s, 1),
               "vlm_backbone": getattr(config, "vlm_model_name", None), "torch": torch.__version__,
               "parallel_info": torch.__config__.parallel_info().splitlines()[:6]},
    )
