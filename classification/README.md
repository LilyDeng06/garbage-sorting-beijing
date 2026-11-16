---
license: Apache License 2.0
image:
  image-classification:
    size_scale:
      - 10k-1m
tags:
  - 垃圾分类
  - classification
  - garbage
---

## 数据集描述
### 数据集简介
147674张带中文标签的生活垃圾图像集，包含可回收垃圾、厨余垃圾、有害垃圾、其他垃圾4个标准垃圾大类，覆盖常见的食品，厨房用品，家具，家电等265个垃圾小类。其中训练集133038张图像，验证集14642张图像，均从海量中文互联网社区语料进行提取，整理出频率较高的常见生活垃圾名称，数据大小为13GB。  
<img src="./trash.jpg" alt="垃圾分类"/>

### 数据集支持的任务
中文类别名称的生活垃圾分类

## 数据集的格式和结构
### 数据格式
数据集包括训练集train和验证集val，train和val文件夹之下按文件夹进行分类，共有265个子文件夹，同类别标签的图片在同一个文件夹下，图片格式为JPG。同时包含与标注文件中label id相对应的中文类名文件classname.txt。

### 数据集加载方式
```python
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

ms_train_dataset = MsDataset.load(
            'garbage265', namespace='tany0699',
            subset_name='default', split='train',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS) # 加载训练集
print(next(iter(ms_train_dataset)))

ms_val_dataset = MsDataset.load(
            'garbage265', namespace='tany0699',
            subset_name='default', split='validation',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS) # 加载验证集
print(next(iter(ms_val_dataset)))
```
### 数据分片
本数据集包含train和val数据集。
| 子数据集    |        train | val |     test |
|---------|-------------:|-----------:|---------:|
| default |  训练集 |  验证集  | / |


### Clone with HTTP
```bash
git clone https://www.modelscope.cn/datasets/tany0699/garbage265.git
```