# steel-defect-detection
描述: 这是一个深度学习项目，利用 HRNet 和 FPN 架构进行钢材表面缺陷检测。本项目针对噪声较大的比赛数据进行了优化，包括椒盐噪声模拟和拼接数据处理，采用自定义增强技术、高级损失函数及多尺度特征融合。仓库包含训练脚本、模型定义和评估指标。
# 钢材表面缺陷检测项目

本项目实现了基于 **HRNet** 和 **FPN** 的钢材表面缺陷检测模型，主要用于语义分割任务。该模型针对工业环境中钢材缺陷的检测优化，具有处理噪声数据的能力，同时集成了先进的多尺度特征融合和损失函数设计。

---

## 项目特点
- **HRNet + FPN 架构**：结合 HRNet 的高分辨率特性与 FPN 的多尺度特征融合能力。
- **鲁棒的噪声处理能力**：优化模型应对噪声数据（如椒盐噪声和拼接数据）的性能。
- **高级损失函数**：支持 Focal Loss 和 Dice Loss 的组合，有效处理类别不平衡问题。
- **性能评估工具**：内置 IoU、mIoU 和 FPS 计算方法，量化分割性能和推理速度。

---

## 数据集
本项目适配比赛提供的钢材表面缺陷数据，数据已处理完成，无需额外预处理。数据集目录结构如下：
dataset/ ├── images/ │ ├── train/ # 训练集图片 │ ├── val/ # 验证集图片 ├── masks/ │ ├── train/ # 训练集掩码 │ ├── val/ # 验证集掩码

请将您的数据放置到上述路径中。

---

## 安装步骤

1. 克隆仓库到本地：
   ```bash
   git clone https://github.com/xiaohuanlaile/steel-defect-detection.git
   cd steel-defect-detection
2.安装依赖项：
pip install -r requirements.txt
3.文件结构：
├── dataset/              # 数据集文件夹
├── train.py              # 训练脚本，用于训练模型
├── test.py               # 验证脚本，用交并比的方式生成npy文件并计算iou
├── run.py                # 生成用训练出来的权重用于分割test中的图片并生成npy文件的运行标本
├── README.md             # 项目文档
├── requirements.txt      # 依赖项文件
