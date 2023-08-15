# MCM

This is the repository of the paper "You Can Mask More For Extremely Low-Bitrate Image Compression". 

## Todo
- Release codes
- Website page

# Dependencies and Installation

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
## Training
```
CUDA_VISIBLE_DEVICES=0 python main_compress.py -d ./dataset/coco -e 100 --batch_size 32 --checkpoint ./checkpoint/pretrained/pretrain_vit_base.pth --output_dir dirpath/to/save/checkpoint --log_dir dirpath/to/save/logs -m MCM --cuda
```
## Inference

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