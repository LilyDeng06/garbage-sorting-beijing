# garbage-sorting-beijing
北京生活垃圾分类

## classification
265 类整图分类。代码、类别表和已训练权重在 `classification/`。最好的权重是 `epoch_013.pt`。

## detection
多目标检测实验：YOLOv8 同时出框和四分类，以及 Faster R-CNN 出框后再做四分类。代码、实验报告和权重在 `detection/`。实验报告是 `detection/实验报告_多目标检测.md`。

权重通过 Git LFS 存放。克隆时需要安装 Git LFS，否则权重文件只会是指针。
