## CAN: Co-embedding Attributed Networks
This repository contains the Python implementation for CAN. Further details about CAN can be found in our paper:
> Zaiqiao Meng, Shangsong Liang, Hongyan Bao, Xiangliang Zhang. Co-embedding Attributed Networks. (WSDM2019)

- A pytorch implementation can be found [here](https://github.com/GuanZhengChen/CAN-Pytorch).

A semi-supervised version of the CAN model can be found in [SCAN](https://github.com/mengzaiqiao/SCAN)
## Introduction

Co-embedding for Attributed Networks (CAN) is a model that learns low-dimensional representations of both attributes and nodes in the same semantic space such that the affinities between them can be effectively captured and measured. The node and attribute embeddings obtained in the unified manner in CAN can benefit not only node-oriented network problems (e.g., node classification and link prediction), but also attribute inference problems (e.g., predicting the value of attributes of nodes). More importantly, the common semantic embedding space provides a simple but effective solution to user profiling problem, as the relevance of users (nodes) and keywords (attributes) can be directly measured, e.g., by cosine similarity, or dot product. 

## Requirements

=================
* TensorFlow (1.0 or later)
* python 2.7/3.6
* scikit-learn
* scipy

## Run the demo
=================

```bash
python train.py
```

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{meng2019co,
  title={Co-embedding Attributed Networks},
  author={Meng, Zaiqiao and Liang, Shangsong and Bao, Hongyan and Zhang, Xiangliang},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  pages={393--401},
  year={2019},
  organization={ACM}
}
```
