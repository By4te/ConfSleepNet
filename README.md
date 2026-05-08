<h2 align="center"> A Conflict-aware Evidential Framework for Reliable Sleep Stage Classification </a></h2>

<div align="center">

**_Yunzhi Tian_<sup>1</sup>, _Dekui Wang_<sup>1</sup>, [_Jun Feng_<sup>1</sup>](https://scholar.google.com/citations?user=3PU_g78AAAAJ&hl=zh-CN&oi=sra), _Qirong Bo_<sup>1</sup>, _Wei Zhou_<sup>1</sup>, _Xingxing Hao_<sup>1</sup>**

<sup>1</sup> College of Computer Science, Northwest University, Xi'an, China


</div>


## Abstract
Multi-view learning has been widely applied for sleep stage classification using multi-modal data. However, existing methods typically assume that different modalities are well-aligned, which is often unattainable in real-world scenarios, thereby compromising the reliability of the staging results. In this paper, we propose ConfSleepNet, a conflict-aware evidential framework that dynamically resolves inter-view conflicts. The framework consists of multi-view evidence extraction and conflict-aware aggregation. In the first phase, it learns category-related evidence from different modalities, which represents the degree of support for individual sleep stages. Considering the inherent characteristics of varying modalities, we propose adaptive category structures for different modalities to promote more reasonable evidence learning. In the second phase, view-specific opinions, including prediction results and uncertainty, are constructed from the learned evidence. Notably, we propose a novel conflict-aware aggregation method that integrates these view-specific opinions into a reliable joint decision. This mechanism can effectively resolve conflicts among opinions and synthesize them into a reliable joint decision. Both theoretical analysis and experimental results demonstrate the effectiveness of ConfSleepNet in sleep staging tasks.

## 🏗️Model
<div align="center">
  <img src="model.png" />
</div>

## 🛏️Experiment 1: Sleep Stage Classification
In this experiment, we investigate the effectiveness of TMCEK on sleep stage classification using EEG signals.
### Directory Structure
The experiment directory is organized as follows:
```bash
Sleep stage classification/
├── data/
│   └── Sleep-EDF 20/
│       └── SC4001E0.npz
├── E1_model_test.py
├── E1_model_training.py
├── E2_lstm_data_prep.py
├── E2_lstm_test.py
├── E2_lstm_training.py
└── loss_function.py
```

### Data

We used three public datasets in this experiment:
- [Sleep-EDF20](https://www.physionet.org/content/sleep-edfx/1.0.0/)
- [Sleep-EDF78](https://www.physionet.org/content/sleep-edfx/1.0.0/)
- [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs)
  
### Experiment Workflow
Below we split the workflow into two phases: **Single-Epoch Network (E1)** which processes each epoch independently, and **Multi-Epoch Network (E2)** which leverages sequential epoch information for enhanced modeling.

#### Training
1. **Train E1 (Single-Epoch Network)**
```bash
python E1_model_training.py 
```
2. **Preprocess data for E2 (Multi-Epoch Network)**  
```bash
 python E2_lstm_data_prep.py
```
3. **Train E2 (Multi-Epoch Network)**
```bash
python E2_lstm_training.py 
```

#### Inference
Once training is complete, evaluate each network on datasets.
1. **Evaluate E1 (Single-Epoch Network)**
```bash
python E1_model_test.py 
```
2. **Evaluate E2 (Multi-Epoch Network)**
```bash
python E2_lstm_test.py 
```

## 🎞️Experiment 2: Multi-view Classification
In this experiment, we evaluate TMCEK across standard multi-view benchmarks.
### Directory Structure
The experiment directory is organized as follows:
```bash
Multi-view Classification/
├── data/
│   └── PIE_face_10.mat
├── dataset.py
├── loss_function.py
├── main.py
└── model.py  
```

### Data
We used four public datasets in this experiment:
- [HandWritten (HW)](https://archive.ics.uci.edu/dataset/72/multiple+features)
- [Scene15](https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177/1)
- [CUB](https://www.vision.caltech.edu/visipedia/CUB-200.html)
- [PIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Home.html)

### Experiment Workflow
The training process can be completed using a single script:
```bash
python main.py
```
## 📑Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{
liang2025trusted,
title={Trusted Multi-View Classification with Expert Knowledge Constraints},
author={Xinyan Liang, Shijie Wang, Yuhua Qian, Qian Guo, Liang Du, Bingbing Jiang, Tingjin Luo, Feijiang Li},
booktitle={Proceedings of the 42th International Conference on Machine Learning},
pages={37409--37426},
year={2025},
volume={267},
}
```

## 🔬 Related Work
We list below the works most relevant to this paper, including but not limited to the following (roughly ordered from most recent to earliest):
- Navigating Conflicting Views: Harnessing Trust for Learning [[paper]](https://arxiv.org/abs/2406.00958)
- Trusted Multi-View Classification via Evolutionary Multi-View Fusion [[paper]](https://openreview.net/pdf?id=M3kBtqpys5)
- Enhancing Multi-View Classification Reliability with Adaptive Rejection [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26066)
- Enhancing Testing-Time Robustness for Trusted Multi-View Classification in the Wild [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Enhancing_Testing-Time_Robustness_for_Trusted_Multi-View_Classification_in_the_Wild_CVPR_2025_paper.pdf)
- Trusted Multi-view Learning with Label Noise [[paper]](https://www.ijcai.org/proceedings/2024/0582.pdf) 
- Trusted Multi-view Learning under Noisy Supervision [[paper]](https://arxiv.org/abs/2404.11944)
- Reliable Conflictive Multi-View Learning [[paper]](https://arxiv.org/abs/2402.16897)
- Safe multi-view deep classification [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26066)
- Trusted Multi-View Deep Learning with Opinion Aggregation [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20724)
- Trusted Multi-View Classification with Dynamic Evidential Fusion [[paper]](https://arxiv.org/abs/2204.11423)
- Trusted Multi-View Classification [[paper]](https://arxiv.org/abs/2102.02051)

<!-- ## 🙏 Acknowledgement -->

## 📬Contact
If you have any detailed questions or suggestions, you can email us: [wshijie0@163.com](mailto:wshijie0@163.com)
