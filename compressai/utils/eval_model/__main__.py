# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import sys
import time
import os

from collections import defaultdict
from pathlib import Path
# from posixpath import split
from typing import Any, Dict, List

import pandas as pd
import numpy as np
# import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.ops import compute_padding
from compressai.zoo import load_state_dict, models
# from compressai.zoo.image import model_architectures as architectures

from compressai.utils.eval_model.huffman import HuffmanCoding
# from compressai.utils.eval_model.tex_sem_sample_visualize1 import sample_patch

# import lpips

# import cv2

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(os.path.join(rootpath,'images/test')).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(org: torch.Tensor, rec: torch.Tensor, max_val: int = 255) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def reconstruct(reconstruction, filename, recon_path):
    reconstruction = reconstruction.squeeze()
    reconstruction.clamp_(0, 1)
    reconstruction = transforms.ToPILImage()(reconstruction.cpu())
    if not os.path.join(recon_path):
        os.mkdir(recon_path)
    reconstruction.save(os.path.join(recon_path, filename))


@torch.no_grad()
def inference(model, x, filename, recon_path, exp_name, dataset):
    if not os.path.exists(recon_path):
        os.makedirs(recon_path)

    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(x, pad, mode="constant", value=0)

    data_info = pd.read_csv(os.path.join(dataset, 'scores/test/texture.csv'), sep=',', encoding="utf-8")

    # Note that the suffix of the image should be consistent, e.g. '.png', '.jpg'
    # If an error is reported, changing to the code below may be valid:
    # t_scores = data_info.loc[(data_info['image'].replace('jpg','png') == filename)].iloc[:, 1:].values.tolist()
    t_scores = data_info.loc[(data_info['image'] == filename)].iloc[:, 1:].values.tolist()
    t_scores = torch.Tensor(t_scores)

    data_info2 = pd.read_csv(os.path.join(dataset, 'scores/test/structure.csv'), sep=',', encoding="utf-8")

    # Note that the suffix of the image should be consistent, e.g. '.png', '.jpg'
    s_scores = data_info2.loc[(data_info2['image'] == filename)].iloc[:, 1:].values.tolist()
    s_scores = torch.Tensor(s_scores)

    start = time.time()
    out_enc = model.compress(x_padded, t_scores, s_scores)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], out_enc["ids_keep"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
    reconstruct(out_dec["x_hat"], filename, recon_path)  # add

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    ids_keep = np.array(out_enc["ids_keep"].cpu())
    ids_keep_path = os.path.join('./compressai/utils/eval_model/bin', exp_name)
    if not os.path.exists('./compressai/utils/eval_model/bin'):
        os.mkdir('./compressai/utils/eval_model/bin')
    if not os.path.exists(ids_keep_path):
        os.system(r"touch {}".format(ids_keep_path))
    np.savetxt(ids_keep_path, ids_keep, fmt="%d")
    h = HuffmanCoding(ids_keep_path)
    output_path = h.compress()
    bpp += (os.path.getsize(output_path) * 8 / num_pixels)
    bpp_path = os.path.join(recon_path, "bpp.txt")
    with open(bpp_path,'a') as f:
        f.write(filename+'\t'+'\t'+str(bpp)+" bpp"+'\n')

    return {
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_net["likelihoods"].values())

    return {
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](quality=quality, metric=metric, pretrained=True).eval()


def load_checkpoint(arch: str, vis_num: int, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['model'])
    return models[arch].from_state_dict(vis_num, state_dict).eval()


def eval_model(model, filepaths, entropy_estimation=False, half=False, recon_path='reconstruction', exp_name=None, dataset='./dataset/coco/'):
    device = next(model.parameters()).device
    metrics = defaultdict(float)

    for f in filepaths:
        _filename = str(f).split("/")[-1]

        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, _filename, recon_path, exp_name, dataset)
        else:
            rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser()

    # Common options.
    parent_parser.add_argument("-d", "--dataset", type=str, help="dataset path")
    parent_parser.add_argument("-r", "--recon_path", type=str, default="reconstruction", help="where to save recon img")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    parent_parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
    )
    parent_parser.add_argument(
        "--vis_num",
        type=int,
        required=True,
    )
    return parent_parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    runs = args.paths
    opts = (args.architecture, )
    load_func = load_checkpoint
    log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, args.vis_num, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")

        model.update(force=True)

        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.recon_path, args.exp_name, args.dataset)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = ("entropy estimation" if args.entropy_estimation else args.entropy_coder)
    output = {
        "name": args.architecture,
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
