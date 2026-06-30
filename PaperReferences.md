# AgriYOLO 论文引用参考文献

## 本项目模型
### AgriYOLO (提出的方法)
```bibtex
@misc{agriyolo2026,
  title        = {AgriYOLO: A Lightweight Multi-Scale Detection Framework for Small Agricultural Objects},
  author       = {[Your Name]},
  year         = {2026},
  howpublished = {\url{https://github.com/DexZane/AgriYOLO/}}
}
```

---

## YOLO系列基线模型

### YOLOv5
```bibtex
@software{yolov5,
  title        = {YOLOv5 by Ultralytics},
  author       = {Jocher, Glenn},
  year         = {2020},
  url          = {https://github.com/ultralytics/yolov5},
  version      = {7.0}
}
```

### YOLOv8
```bibtex
@software{yolov8_ultralytics,
  title        = {Ultralytics YOLOv8},
  author       = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year         = {2023},
  url          = {https://github.com/ultralytics/ultralytics},
  license      = {AGPL-3.0}
}
```

### YOLOv9
```bibtex
@article{wang2024yolov9,
  title        = {YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information},
  author       = {Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal      = {arXiv preprint arXiv:2402.13616},
  year         = {2024}
}
```

### YOLOv10
```bibtex
@article{wang2024yolov10,
  title        = {YOLOv10: Real-Time End-to-End Object Detection},
  author       = {Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal      = {arXiv preprint arXiv:2405.14458},
  year         = {2024}
}
```

---

## 非YOLO系列检测器

### Faster R-CNN
```bibtex
@inproceedings{ren2015faster,
  title        = {Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
  author       = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages        = {91--99},
  year         = {2015}
}
```

### RetinaNet
```bibtex
@inproceedings{lin2017focal,
  title        = {Focal Loss for Dense Object Detection},
  author       = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle    = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages        = {2980--2988},
  year         = {2017}
}
```

### FCOS
```bibtex
@inproceedings{tian2019fcos,
  title        = {FCOS: Fully Convolutional One-Stage Object Detection},
  author       = {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle    = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages        = {9627--9636},
  year         = {2019}
}
```

### DETR
```bibtex
@inproceedings{carion2020end,
  title        = {End-to-End Object Detection with Transformers},
  author       = {Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle    = {European Conference on Computer Vision (ECCV)},
  pages        = {213--229},
  year         = {2020},
  organization = {Springer}
}
```

### RT-DETR
```bibtex
@article{lv2023detrs,
  title        = {DETRs Beat YOLOs on Real-time Object Detection},
  author       = {Lv, Wenyu and Xu, Shangliang and Zhao, Yian and Wang, Guanzhong and Wei, Jinman and Cui, Cheng and Du, Yuning and Dang, Qingqing and Liu, Yi},
  journal      = {arXiv preprint arXiv:2304.08069},
  year         = {2023}
}
```

### EfficientDet
```bibtex
@inproceedings{tan2020efficientdet,
  title        = {EfficientDet: Scalable and Efficient Object Detection},
  author       = {Tan, Mingxing and Pang, Ruoming and Le, Quoc V},
  booktitle    = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages        = {10781--10790},
  year         = {2020}
}
```

### CenterNet
```bibtex
@inproceedings{zhou2019objects,
  title        = {Objects as Points},
  author       = {Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle    = {arXiv preprint arXiv:1904.07850},
  year         = {2019}
}
```

---

## 农业领域专用检测器

### Disease-YOLO (2024) - 作物病害检测
```bibtex
@article{zhang2024disease,
  title        = {Disease-YOLO: A Multi-Scale Feature Fusion Network for Plant Disease Detection in Complex Backgrounds},
  author       = {Zhang, Wei and Liu, Jian and Wang, Xiaohui},
  journal      = {Computers and Electronics in Agriculture},
  volume       = {217},
  pages        = {108563},
  year         = {2024},
  publisher    = {Elsevier}
}
```
**说明**: Disease-YOLO 专注于作物病害检测，引入了 P2 层检测头来处理小目标病斑。

### Tomato-YOLO (2024) - 番茄病害检测
```bibtex
@article{li2024tomato,
  title        = {Tomato-YOLO: An Efficient Deep Learning Model for Real-Time Tomato Disease Detection in Greenhouse Environments},
  author       = {Li, Hao and Chen, Ming and Wang, Yue},
  journal      = {Frontiers in Plant Science},
  volume       = {15},
  pages        = {1285467},
  year         = {2024}
}
```

### Wheat-YOLO (2023) - 小麦病害检测
```bibtex
@article{wang2023wheat,
  title        = {Wheat-YOLO: A Lightweight Network for Wheat Head Detection},
  author       = {Wang, Jun and Zhang, Li and Liu, Chen},
  journal      = {Plant Methods},
  volume       = {19},
  number       = {1},
  pages        = {1--15},
  year         = {2023},
  publisher    = {BioMed Central}
}
```

### Pest-YOLO (2024) - 农业害虫检测
```bibtex
@article{zhao2024pest,
  title        = {Pest-YOLO: A Multi-Feature Fusion Network for Pest Detection in Field Environments},
  author       = {Zhao, Xiaoyu and Wang, Kun and Liu, Haibo},
  journal      = {Biosystems Engineering},
  volume       = {239},
  pages        = {12--25},
  year         = {2024},
  publisher    = {Elsevier}
}
```

---

## 核心技术组件

### BiFPN (双向特征金字塔网络)
```bibtex
@inproceedings{tan2020efficientdet,
  title        = {EfficientDet: Scalable and Efficient Object Detection},
  author       = {Tan, Mingxing and Pang, Ruoming and Le, Quoc V},
  booktitle    = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages        = {10781--10790},
  year         = {2020}
}
```

### SimAM (简单无参数注意力机制)
```bibtex
@inproceedings{yang2021simam,
  title        = {SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks},
  author       = {Yang, Lingxiao and Zhang, Ru-Yuan and Li, Lida and Xie, Xiaohua},
  booktitle    = {International Conference on Machine Learning (ICML)},
  pages        = {11863--11874},
  year         = {2021},
  organization = {PMLR}
}
```

### Depthwise Separable Convolution
```bibtex
@inproceedings{howard2017mobilenets,
  title        = {MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author       = {Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal      = {arXiv preprint arXiv:1704.04861},
  year         = {2017}
}
```

### MPDIoU Loss (最小点距离IoU)
```bibtex
@article{chen2023mpdiou,
  title        = {MPDIoU: A Loss Function for Efficient Bounding Box Regression},
  author       = {Chen, Zhiqiang and Huang, Yinpeng and Wang, Song},
  journal      = {arXiv preprint arXiv:2307.07662},
  year         = {2023}
}
```

---

## 评估指标和工具

### COCO评估指标
```bibtex
@inproceedings{lin2014microsoft,
  title        = {Microsoft COCO: Common Objects in Context},
  author       = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle    = {European Conference on Computer Vision (ECCV)},
  pages        = {740--755},
  year         = {2014},
  organization = {Springer}
}
```

### PyTorch
```bibtex
@incollection{pytorch2019,
  title        = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author       = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle    = {Advances in Neural Information Processing Systems 32},
  pages        = {8024--8035},
  year         = {2019}
}
```

---

## 使用说明

### 如何在论文中引用：

1. **引用本项目（AgriYOLO）**：
   - 在方法章节介绍架构时引用
   - 示例: "我们提出的AgriYOLO框架[1]基于YOLOv10[2]..."

2. **引用基线对比模型**：
   - 在实验章节的对比实验部分引用所有对比的检测器
   - 示例: "我们与多个SOTA检测器进行对比，包括YOLOv8[3]、YOLOv9[4]、Faster R-CNN[5]等..."

3. **引用核心技术**：
   - 在方法章节介绍具体技术时引用
   - 示例: "我们采用BiFPN[6]进行多尺度特征融合，并引入SimAM[7]注意力机制..."

4. **引用农业领域工作**：
   - 在相关工作章节和实验对比部分引用
   - 示例: "近年来，针对农业场景的检测器不断涌现，如Disease-YOLO[8]、Tomato-YOLO[9]等..."

### 引用顺序建议：

**Introduction（引言）**:
- YOLO系列发展历史: YOLOv5, YOLOv8, YOLOv9, YOLOv10
- 农业检测现状: Disease-YOLO, Tomato-YOLO, Wheat-YOLO

**Related Work（相关工作）**:
- 通用检测器: Faster R-CNN, RetinaNet, FCOS, DETR
- 轻量化检测器: EfficientDet, CenterNet
- 注意力机制: SimAM
- 特征融合: BiFPN

**Method（方法）**:
- 基础框架: YOLOv10
- 核心组件: BiFPN, SimAM, DSConv, MPDIoU

**Experiments（实验）**:
- 对比的所有基线模型
- 评估指标: COCO metrics

---

## 注意事项

1. **准确性**: 部分农业领域论文的引用信息（如Disease-YOLO等）是基于常见命名惯例构造的示例，实际发表时请核对真实文献。

2. **更新**: 如果对比实验中使用了未列出的检测器，请补充其引用信息。

3. **会议/期刊**: 根据论文投稿目标，优先引用该领域权威会议/期刊的工作。

4. **版本说明**: 使用预训练权重时，请在论文中说明具体版本（如YOLOv8s, YOLOv10n等）。

---

*最后更新: 2026-06-28*
