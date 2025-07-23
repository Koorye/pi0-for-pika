import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PikaInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = data['state']
        state = transforms.pad_to_dim(state, self.action_dim)

        left_wrist_image = _parse_image(data["observation.images.cam_left_wrist_fisheye"])
        right_wrist_image = _parse_image(data["observation.images.cam_right_wrist_fisheye"])

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (np.zeros_like(left_wrist_image), left_wrist_image, right_wrist_image)
                image_masks = (np.False_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (np.zeros_like(left_wrist_image), left_wrist_image, right_wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = data["actions"]
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PikaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        return {"actions": np.asarray(data["actions"][:, :14])}
