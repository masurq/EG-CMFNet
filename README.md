\# EF-CMFNet

This repo holds code for \[EG-CMFNet: An Edge-Guided Cross-Modal Fusion

Network for Remote Sensing Semantic Segmentation](https://ieeexplore.ieee.org/document/11481201)

\# The overall architecture

We propose a novel Edge guided Cross-modal Fusion Network (EG-CMFNet). Specifically, a multi-stream CNN and Transformer architecture extracts global semantic features from optical and SAR data. Meanwhile, Multimodal Adaptive Feature Interaction (MAFI) modules and Lightweight Feature Fusion (LFF) modules conduct adaptive cross-modal interactions and efficient multiscale fusion across hierarchical feature levels. Additionally, a CNN-based edge branch incorporates an Edge Auxiliary (EA) module to refine boundary by leveraging edge information.

!\[overall architecture](image/figure.png)

\# Citation

If you find this work useful, please consider citing:



```bibtex

@ARTICLE{11481201,

&#x20; author={Zhao, Jinqi and Zhou, Zhonghuai and Zhang, Liansong and Wang, Linxin and Lu, Zhong},

&#x20; journal={IEEE Transactions on Geoscience and Remote Sensing}, 

&#x20; title={EG-CMFNet: An Edge-Guided Cross-Modal Fusion Network for Remote Sensing Semantic Segmentation}, 

&#x20; year={2026},

&#x20; volume={64},

&#x20; number={},

&#x20; pages={1-20},

&#x20; keywords={Satellite images;Earth Observing System;Feeds;Apertures;Antennas;Filtering;Filters;Speckle;Circuits;Feedback;CNN;edge information;multimodal remote sensing data;semantic segmentation;Transformer},

&#x20; doi={10.1109/TGRS.2026.3680719}}



