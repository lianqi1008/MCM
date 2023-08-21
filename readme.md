# MCM

This is the repository of the paper "You Can Mask More For Extremely Low-Bitrate Image Compression". 

[[Paper](http://www.google.com/)]

# Dependencies and Installation
```
git clone https://github.com/lianqi1008/MCM.git
cd MCM
```
# Get Started
## Preparation
Generate the patch scores:
```
python util/score_cal.py -d0 path/to/image -d1 path/to/structure -d2 path/to/save/score
```
The final file structure is as follows:
```
checkpoint
    |- pretrained
        |- pretrained.pth
    |- finetuned
        |- coco
            |- xxx.pth
            |- ...
        |- face
            |- xxx.pth
            |- ...
dataset
    |- coco
        |- images
            |- train
            |- test
        |- scores
        |- structure|- coco
    |- celeba
```
## Train
```
CUDA_VISIBLE_DEVICES=0 python main_compress.py \
-m MCM -d ./dataset/coco -e 100 --batch_size 32 \
--checkpoint ./checkpoint/pretrained/pretrain_vit_base.pth \
--output_dir dirpath/to/save/checkpoint \
--log_dir dirpath/to/save/logs --cuda
```
## Inference
Note that '--exp_name' is the location where the bit stream of the token index is saved, you can name it arbitrarily, and you can delete the folder after inferencing, which is not important.

"Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU)."(mentioned in CompressAI), so please run on CPU for inference.
```
CUDA_VISIBLE_DEVICES=0 python -m compressai.utils.eval_model \
-a MCM -d './dataset/coco/' -r dirpath/to/output \
-p './checkpoint/finetuned/coco/checkpoint_xxx.pth' \
--exp_name coco --vis_num 144 --cuda
```
# Citation
```
@misc{li2023mask,
      title={You Can Mask More For Extremely Low-Bitrate Image Compression},
      author={Anqi Li and Feng Li and Jiaxin Han and Huihui Bai and Runmin Cong and Chunjie Zhang and Meng Wang and Weisi Lin and Yao Zhao},
      year={2023},
      eprint={2306.15561},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
 }  
```
# Related Links
MAE: https://github.com/facebookresearch/mae.git  
CompressAI: https://github.com/InterDigitalInc/CompressAI  
Tensorflow compression library: https://github.com/tensorflow/compression  
MS COCO Dataset: https://cocodataset.org  
CelebAMask-HQ Dataset: https://github.com/switchablenorms/CelebAMask-HQ  