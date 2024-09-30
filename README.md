# CION_ReIDZoo
<div align="center"><img src="assets/ReIDZoo.png" width="900"></div>

<div align="center">

[Jialong Zuo](https://scholar.google.jp/citations?user=R5OWszMAAAAJ&hl=zh-CN&oi=ao) <sup>1</sup>,
[Ying Nie](https://scholar.google.jp/citations?user=1eOYln4AAAAJ&hl=zh-CN&oi=ao) <sup>2</sup>,
[Hanyu Zhou](https://scholar.google.jp/citations?hl=zh-CN&user=FgHTmS4AAAAJ) <sup>1</sup>,
[Huaxin Zhang](https://scholar.google.cz/citations?user=oyfu0pgAAAAJ&hl=zh-CN&oi=ao)  <sup>1</sup>,
[Haoyu Wang](haoyuwang199705@gmail.com) <sup>2</sup>,
[Tianyu Guo](https://scholar.google.jp/citations?hl=zh-CN&user=RPK3oQgAAAAJ) <sup>2</sup>,
[Nong Sang](https://scholar.google.jp/citations?hl=zh-CN&user=ky_ZowEAAAAJ) <sup>1</sup>,
[Changxin Gao](https://scholar.google.jp/citations?hl=zh-CN&user=ky_ZowEAAAAJ) <sup>1,*</sup>
<br>
<sup>1</sup> Huazhong University of Science and Technology,
<sup>2</sup> Huawei Noah’s Ark Lab.
<br>
\* Corresponding Author.
<br>

</div>


* **ReIDZoo** is a new fully open-sourced **pre-trained model zoo** to meet diverse research and application needs in the field of person re-identification. It contains a series of CION pre-trained models with spanning structures and parameters, totaling 32 models with 10 different structures, including GhostNet, ConvNext, RepViT, FastViT and so on. 

* **CION** is our proposed person re-identification **pre-training framework** that deeply utilizes cross-video identity correlations. It simply consists of a progressive identity correlation seeking strategy and an identity-guided self-distillation pre-training technology.

* **CION-AL** is a new large-scale person re-identification **pre-training dataset** with almost accurate identity labels. The images are obtained from LUPerson-NL based on our proposed progressive identity correlation seeking strategy. It contains 3,898,086 images of 246,904 identities totally.

Our pre-trained models enable existing person ReID algorithms to achieve significantly better performance without bells and whistles. In this project, we will open-source the code, models and dataset. More details can be found at our paper [Cross-video Identity Correlating for Person Re-identification Pre-training](https://arxiv.org/abs/2409.18569v1).

Note:  To address the privacy concerns associated with crawling videos from the internet and the application of person ReID technology, we will implement a controlled release of our models and dataset, thereby preventing privacy violations and ensuring information security.

## News
* 🙂[2024.9.30] Our paper is released on [Arxiv](https://arxiv.org/abs/2409.18569v1).
* 🙂[2024.9.26] Good News! Our paper is accepted by **NeurIPS2024**.

## TODO
- [x] Release the paper.
- [x] Release the 32 models of ReIDZoo.
- [x] Release the CION-AL dataset.
- [x] Release the pre-training code of CION.
- [ ] Release the downstream fine-tuning code.

## ReIDZoo
**ReIDZoo contains 32 CION pre-trained models with 10 different structures**. Among them, the **GhostNet, EdgeNext, RepViT and FastViT** are representative models with lightweight designs, which have smaller computational overhead and are convenient for practical deployment. Meanwhile, the **ResNet, ResNet-IBN, ConvNext, VOLO, Vision Transformer and Swin Transformer** are conventional models, which usually have more parameters and enjoy better performance. 

The supervised fine-tuning performance of each model is shown in the table below. The up-arrow value represents the performance improvement compared to each corresponding ImageNet pre-trained model. As a common practice, we utilized MGN and TransReID as the fine-tuning algorithms. Please refer to [our paper](https://arxiv.org/abs/2409.18569v1) for more experimental details and results. 

To obtain the ReIDZoo, please download and sign the license agreement (ReIDZoo_License.pdf) and send it to jlongzuo@hust.edu.cn or cgao@hust.edu.cn . Once the procedure is approved, the download link will be sent to your email.

<div align="center"><img src="assets/performance.jpg" width="520"></div>

## CION-AL(Dataset)

CION-AL is a large-scale person re-identification pre-training dataset with almost accurate identity labels. In constructing the CION-AL dataset, we utilize our proposed progressive identity correlation seeking strategy to tag the images in LUPerson-NL with more accurate identity labels. It contains **3,898,086** images of **246,904** identities totally. 

To obtain the CION-AL dataset, please download and sign the license agreement (CION_AL_License.pdf) and send it to jlongzuo@hust.edu.cn or cgao@hust.edu.cn . Once the procedure is approved, the download link will be sent to your email.



## CION(Pre-train)

CION is our proposed **C**ross-video **I**dentity-c**O**rrelating pre-trai**N**ing framework for person re-identification. By utilizing our CION to pre-train the models, it is easier to learn identity-invariant representations for person re-identification. Now, we introduce the guidelines of how to pre-train the models from scratch.

### Dataset Prepare

Download the CION-AL dataset and organize it in `dataset` folder as follows:

```
|-- CION_Pretrain/
|   |-- dataset/
|   |   |-- CION_AL/
|   |       |-- sec0
|   |       |-- sec1
|   |       |-- ...
|   |       |-- sec129
|   |       |-- cional_annos.json
|   |-- data/
|   |-- models/
|   |-- loss.py
|   |-- run.py
|   |-- utils.py
```

### Pre-train from Scratch

As a default setting, we use 8×V100 (32GB) GPUs for pre-training the models. We set different batch sizes and numbers of local cropped views for different model to achieve better computational resouce utilization. Specific settings for each model can be found in Table 5 of [our paper](https://arxiv.org/abs/2409.18569v1). 

Taking ResNet50-IBN as an example, run the following command to implement the pre-training process. On 8×V100s, the entire pre-training process will take approximately 5 days.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 run.py --arch resnet50_ibn --batch_size_per_gpu 120 --local_crops_number 8 --output_dir logs/resnet50_ibn
```

## Downstream Fine-tuning


## Acknowledgements
We thank these great works and open-source repositories: [DINO](https://github.com/facebookresearch/dino), [LUPerson-NL](https://github.com/DengpanFu/LUPerson-NL), [TransReID](https://github.com/damo-cv/TransReID), [FastReID](https://github.com/JDAI-CV/fast-reid), [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid), [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL), [GhostNet](https://github.com/huawei-noah/Efficient-AI-Backbones), [ResNet](https://github.com/pytorch/vision), [ResNet-IBN](https://github.com/XingangPan/IBN-Net), [EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt), [RepViT](https://github.com/THU-MIG/RepViT), [FastViT](https://github.com/apple/ml-fastvit), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [Vision Transformer](https://github.com/google-research/vision_transformer), [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [VOLO](https://github.com/sail-sg/volo).
