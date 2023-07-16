import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
from utils import to_normalized_coords, to_pixel_coords
from silk.backbones.silk.silk import from_feature_coords_to_image_coords


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy

import numpy as np
import skimage.io as io
import torch

#import torchvision
from silk.backbones.silk.silk import SiLKVGG as SiLK
from silk.backbones.superpoint.vgg import ParametricVGG

from silk.config.model import load_model_from_checkpoint
from silk.models.silk import matcher


CHECKPOINT_PATH = "coco-rgb-aug.ckpt"
DEVICE = "cpu" #"cuda:0"

SILK_NMS = 0  # NMS radius, 0 = disabled
SILK_BORDER = 0  # remove detection on border, 0 = disabled
SILK_THRESHOLD = 1  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
SILK_TOP_K = 10000  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
    "dense_positions",
    "normalized_descriptions",
    "probability",
)
SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change
SILK_BACKBONE = ParametricVGG(
    use_max_pooling=False,
    padding=0,
    normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
)


def load_images(*paths, as_gray=True):
    images = np.stack([np.array(Image.open(path).convert('L').resize((512,512)))/255. for path in paths])
    images = torch.tensor(images, device=DEVICE, dtype=torch.float32)
    images = images.unsqueeze(1)  # add channel dimension
    return images


def get_model(
    checkpoint=CHECKPOINT_PATH,
    nms=SILK_NMS,
    device=DEVICE,
    default_outputs=SILK_DEFAULT_OUTPUT,
):
    # load model
    model = SiLK(
        in_channels=1,
        backbone=deepcopy(SILK_BACKBONE),
        detection_threshold=SILK_THRESHOLD,
        detection_top_k=SILK_TOP_K,
        nms_dist=nms,
        border_dist=SILK_BORDER,
        default_outputs=default_outputs,
        descriptor_scale_factor=SILK_SCALE_FACTOR,
        padding=0,
    )
    model = load_model_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
        device=device,
        freeze=True,
        eval=True,
    )
    return model


class SiLKDetector(nn.Module):
    
    def __init__(self, device = "cuda"):
        super().__init__()
        self.detector = get_model(default_outputs=("sparse_positions")).to(device)
    
    @torch.inference_mode()
    def detect(self, batch, device = "cuda"):
        self.train(False)
        images = batch["image"]
        B,C,H,W = images.shape

        # run model
        sparse_positions_0 = self.detector(images.to(device))
        
        sparse_positions_0 = from_feature_coords_to_image_coords(self.detector, sparse_positions_0)

        keypoints = torch.stack([to_normalized_coords(
            torch.stack((silk_detections[:,1],silk_detections[:,0]),dim=-1), H, W).to(device) 
                     for silk_detections in sparse_positions_0])
        return {"keypoints": keypoints}
    
    def read_image(self, im_path):
        return load_images(im_path)

    def detect_from_path(self, im_path):
        batch = {"image": self.read_image(im_path)}
        return self.detect(batch)

    def to_pixel_coords(self, x, H, W):
        return to_pixel_coords(x, H, W)
    
    def to_normalized_coords(self, x, H, W):
        return to_normalized_coords(x, H, W)