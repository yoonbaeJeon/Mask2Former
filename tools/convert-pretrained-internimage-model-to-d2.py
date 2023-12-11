#!/usr/bin/env python

import pickle as pkl
import sys

import torch

"""
Usage:
  # download pretrained internimage model:
  wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth
  # run the conversion
  ./convert-pretrained-model-to-d2.py internimage_xl_22k_192to384.pth internimage_xl_22k_192to384.pkl
  # Then, use internimage_xl_22k_192to384.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/internimage_xl_22k_192to384.pkl"
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")["model"]
    if 'model' in obj:
        obj = obj['model']
    elif 'module' in obj:
        obj = obj['module']
    res = {"model": obj, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
