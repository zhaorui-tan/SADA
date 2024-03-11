# Semantic-aware Data Augmentation for Text-to-Image Synthesis with DF-GAN Framework for ICML Review

## What is included:
Due to file size limitation, we offer one checkpoint with our Semantic-aware Data Augmentation (Task 3) in `final_models/birds_1300e`.
- df_itac_Lr_930.pth for DF-GAN + ITA_C + L_r

ITA_C and ITA_T code are in 
- models/semantic_aug.py 


### For Review, you can
  ```
  cd DF-GAN/code/
  ```
  - For visual: `bash scripts/sample_visual.sh ./cfg/bird_review.yml` 
  - For CS: `bash scripts/calc_cs.sh ./cfg/bird_review.yml` -- Download CLIP for checking CS : [CLIP](https://github.com/openai/CLIP)
  - For FID: `bash scripts/calc_fid.sh ./cfg/bird_review.yml` 

For reviewing our code, you need download some files from DF_GAN.
Download the preprocessed metadata: [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing)
[coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`

---
## Requirements
- python 3.8
- Pytorch 1.9
- At least 1x12GB NVIDIA GPU

## Preparation
### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`


### DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis (CVPR 2022 Oral)
The code is modified form Official Pytorch implementation [DF-GAN](https://github.com/tobran/DF-GAN) for DF-GAN paper [DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis](https://arxiv.org/abs/2008.05865) by [Ming Tao](https://scholar.google.com/citations?user=5GlOlNUAAAAJ=en), [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en), [Fei Wu](https://scholar.google.com/citations?user=tgeCjhEAAAAJ&hl=en), [Xiao-Yuan Jing](https://scholar.google.com/citations?hl=en&user=2IInQAgAAAAJ), [Bing-Kun Bao](https://scholar.google.com/citations?user=lDppvmoAAAAJ&hl=en), [Changsheng Xu](https://scholar.google.com/citations?user=hI9NRDkAAAAJ). 
You can obtain the original released checkpoint for DF-GAN and compare them with our trained version (With ITA and L_r).
#### Download Pretrained Model
- [DF-GAN for bird](https://drive.google.com/file/d/1rzfcCvGwU8vLCrn5reWxmrAMms6WQGA6/view?usp=sharing). Download and save it to `./code/saved_models/bird/`
- [DF-GAN for coco](https://drive.google.com/file/d/1e_AwWxbClxipEnasfz_QrhmLlv2-Vpyq/view?usp=sharing). Download and save it to `./code/saved_models/coco/`
#### Citing for DF-GAN
```
@inproceedings{tao2022df,
  title={DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis},
  author={Tao, Ming and Tang, Hao and Wu, Fei and Jing, Xiao-Yuan and Bao, Bing-Kun and Xu, Changsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16515--16525},
  year={2022}
}
```

