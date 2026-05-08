<h2 align="center"> A Conflict-aware Evidential Framework for Reliable Sleep Stage Classification </a></h2>

<div align="center">

**_Yunzhi Tian_<sup>1</sup>, _Dekui Wang_<sup>1,*</sup>, [_Jun Feng_<sup>1,*</sup>](https://scholar.google.com/citations?user=3PU_g78AAAAJ&hl=zh-CN&oi=sra), _Qirong Bo_<sup>1</sup>, _Wei Zhou_<sup>1</sup>, _Xingxing Hao_<sup>1</sup>**

<sup>1</sup> College of Computer Science, Northwest University, Xi'an, China

<sup>*</sup> Corresponding Authors

</div>

## Abstract
Multi-view learning has been widely applied for sleep stage classification using multi-modal data. However, existing methods typically assume that different modalities are well-aligned, which is often unattainable in real-world scenarios, thereby compromising the reliability of the staging results. In this paper, we propose ConfSleepNet, a conflict-aware evidential framework that dynamically resolves inter-view conflicts. The framework consists of multi-view evidence extraction and conflict-aware aggregation. In the first phase, it learns category-related evidence from different modalities, which represents the degree of support for individual sleep stages. Considering the inherent characteristics of varying modalities, we propose adaptive category structures for different modalities to promote more reasonable evidence learning. In the second phase, view-specific opinions, including prediction results and uncertainty, are constructed from the learned evidence. Notably, we propose a novel conflict-aware aggregation method that integrates these view-specific opinions into a reliable joint decision. This mechanism can effectively resolve conflicts among opinions and synthesize them into a reliable joint decision. Both theoretical analysis and experimental results demonstrate the effectiveness of ConfSleepNet in sleep staging tasks.

## Overall Framework
<div align="center">
  <img width="7175" height="3563" alt="图" src="https://github.com/user-attachments/assets/c6815e09-a109-4529-aae4-026cfb286bd6" />

</div>

## Sleep Stage Classification
Taking the **SleepEDF-20** dataset as an example, we provide the complete experimental procedure of ConfSleepNet to validate its effectiveness in sleep stage classification.


### Data

Multiple public datasets, including SleepEDF-20, were used for model validation, and these datasets are available at the following links (apply or download directly):
- [SleepEDF-20](https://www.physionet.org/content/sleep-edfx/1.0.0/)
- [SleepEDF-78](https://www.physionet.org/content/sleep-edfx/1.0.0/)
- [MASS-SS3](https://ceams-carsm.ca/mass/)
- [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs)
  

## Additional Experiments
The proposed conflict-aware multi-view aggregation method can be evaluated on multi-view tasks, and these datasets are available at the following links:
- [PIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Home.html)
- [HandWritten](https://archive.ics.uci.edu/dataset/72/multiple+features)
- [Scene15](https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177/1)
- [CUB](https://www.vision.caltech.edu/visipedia/CUB-200.html)

## Citation
If you find that ConfSleepNet helps your research, please cite our paper:
```
@inproceedings{
Tian2026Conflict,
title={A Conflict-aware Evidential Framework for Reliable Sleep Stage Classification},
author={Yunzhi Tian, Dekui Wang, Jun Feng, Qirong Bo, Wei Zhou, Xingxing Hao},
booktitle={Proceedings of the 43rd International Conference on Machine Learning},
pages={},
year={2026},
volume={},
}
```

## Contact
If you have any detailed questions or suggestions, you can email us: [nwu_tyz@stumail.nwu.edu.cn](mailto:nwu_tyz@stumail.nwu.edu.cn)
