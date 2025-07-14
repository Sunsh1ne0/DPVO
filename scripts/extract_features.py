import os
from typing import List, Literal

from PIL import Image
import torch
from torchvision import transforms as tvf
from torch import nn
from torch.nn import functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

_DINO_V2_MODELS = Literal[
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
]
_DINO_FACETS = Literal["query", "key", "value", "token"]


class DinoV2ExtractFeatures:
    """
    Extract features from an intermediate layer in Dino-v2
    """

    def __init__(
        self,
        dino_model: _DINO_V2_MODELS,
        layer: int,
        facet: _DINO_FACETS = "token",
        use_cls=False,
        norm_descs=True,
        device: str = "cuda",
        gem_p: int = 3,
    ) -> None:
        """
        Parameters:
        - dino_model:   The DINO-v2 model to use
        - layer:        The layer to extract features from
        - facet:    "query", "key", or "value" for the attention
                    facets. "token" for the output of the layer.
        - use_cls:  If True, the CLS token (first item) is also
                    included in the returned list of descriptors.
                    Otherwise, only patch descriptors are used.
        - norm_descs:   If True, the descriptors are normalized
        - device:   PyTorch device to use
        """
        self.vit_type: str = dino_model
        try:
            self.dino_model: nn.Module = torch.load(
                f"weights/{dino_model}.pt", weights_only=False
            )
            print(f"Loaded local {dino_model} model")
        except FileNotFoundError:
            if not os.path.exists("./weights"):
                os.mkdir("./weights")
            self.dino_model: nn.Module = torch.hub.load(
                "facebookresearch/dinov2", dino_model
            )
            # torch.save(self.dino_model, f'weights/{dino_model}.pt')
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].register_forward_hook(
                self._generate_forward_hook()
            )
        else:
            self.fh_handle = self.dino_model.blocks[
                self.layer
            ].attn.qkv.register_forward_hook(self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs

        self._hook_out = None

        self._gem_p = gem_p

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        return _forward_hook

    def __call__(self, imgs: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters:
        - img:   The input image
        """
        with torch.no_grad():
            img_tensors = []
            for img_pt in imgs:
                c, h, w = img_pt.shape
                h_new, w_new = (h // 14) * 14, (w // 14) * 14
                img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
                img_tensors.append(img_pt)

            img_batch = torch.cat(img_tensors, dim=0)

            res = self.dino_model(img_batch)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len : 2 * d_len]
                else:
                    res = res[:, :, 2 * d_len :]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None
        return self.get_gem_descriptors(res)

    def __del__(self):
        self.fh_handle.remove()

    def get_gem_descriptors(self, patch_descs: torch.Tensor) -> torch.Tensor:
        g_res = torch.mean(torch.abs(patch_descs) ** self._gem_p, dim=-2) ** (
            1 / self._gem_p
        )

        g_res = g_res.squeeze(0)

        return g_res


parser = argparse.ArgumentParser()
parser.add_argument("images_dir")
parser.add_argument("save_dir")
args = parser.parse_args()
print(args.images_dir)
print(args.save_dir)

IMG_EXT: str = "jpg"
batch_size: int = 16

extractor = DinoV2ExtractFeatures('dinov2_vitb14', 11)
base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])

input_path = Path(args.images_dir)
images = sorted((input_path).glob(f"*.{IMG_EXT}"), 
                key=lambda file_name: int(file_name.name.split(".")[0]))

features_list = []

for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
    batch_imgs = torch.stack([base_tf(Image.open(path).convert('RGB')) for path in images[i:i + batch_size]]).cuda()
    batch_features = extractor(batch_imgs)
    features_list.append(batch_features.cpu().numpy())

features_array = np.concatenate(features_list, axis=0)
np.save(f"{args.save_dir}/features.npy", features_array)
