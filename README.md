## Introduction
CELA-MFP is a deep learning framework for identifying multi-functional therapeutic peptides. CELA-MFP incorporates feature Contrastive Enhancement and Label Adaptation for predicting Multi-Functional therapeutic Peptides. CELA-MFP utilizes a pretrained protein language model to extract features from peptide sequences, which are then fed into a Transformer decoder for function prediction, effectively modeling correlations between different functions. To enhance the representation of each peptide sequence, contrastive learning is employed during training. The CELA-MFP web server is freely available in [here](http://dreamai.cmii.online/CELA-MFP).

## System requirement
python==3.7.11 \
numpy==1.21.2 \
torch==1.8.0 \
transformers==4.15.0 \
sentencepiece==0.1.96

## Download pretrained weights
Before running this code, you should download three pretrained weights:
1. pretrained_model_dir: this dir includes some files for the protein language model. You can download it from [pLM](https://pan.baidu.com/s/1ShILwO13popFFwUXlEwjHg), and the extracting code is `q34b`.
2. weights_mfbp: this dir includes ten pretrained CELA-MFP model weight files for MFBP. You can download it from [MFBP_weights](https://pan.baidu.com/s/1HvjeZDSHx3ZOXNDMzZzgZw), and the extracting code is `66qe`.
3. weights_mftp: this dir includes ten pretrained CELA-MFP model weight files for MFTP. You can download it from [MFTP_weights](https://pan.baidu.com/s/12plv6HBShaqhi7iCZQHmNw), and the extracting code is `navp`.

## Running CELA-MFP for prediction (GPU or CPU)
You can run this code with GPU or CPU. \
When use GPU:
```
CUDA_VISIBLE_DEVICES=0 python predict_MFBP.py --seq FGLPMLSILPKALCILLKRKC

CUDA_VISIBLE_DEVICES=0 python predict_MFTP.py --seq ACDTATCVTHRLAGLLSRSGGVVKNNFVPTNVGSKAF
```
When use CPU:
```
CUDA_VISIBLE_DEVICES='' python predict_MFBP.py --seq FGLPMLSILPKALCILLKRKC

CUDA_VISIBLE_DEVICES='' python predict_MFTP.py --seq ACDTATCVTHRLAGLLSRSGGVVKNNFVPTNVGSKAF
```

## Citation and contact
```
@article{Fang2024CELA-MFP,
    author = {Fang, Yitian and Luo, Mingshuang and Ren, Zhixiang and Wei, Leyi and Wei, Dong-Qing},
    title = "{CELA-MFP: a contrast-enhanced and label-adaptive framework for multi-functional therapeutic peptides prediction}",
    journal = {Briefings in Bioinformatics},
    volume = {25},
    number = {4},
    pages = {bbae348},
    year = {2024},
    month = {07},
    issn = {1477-4054},
    doi = {10.1093/bib/bbae348},
    url = {https://doi.org/10.1093/bib/bbae348},
}
```
