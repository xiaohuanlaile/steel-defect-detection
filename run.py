import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torchinfo import summary

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        # 使用 1x1 卷积将输入特征的通道数统一到 out_channels
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1)
                                            for in_channels in in_channels_list])
        # 3x3 卷积用于减少上采样融合后的伪影
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                                           for _ in in_channels_list])

    def forward(self, feature_maps):
        # 先将最高分辨率的特征图加入到输出特征列表中
        fpn_outs = [self.lateral_convs[-1](feature_maps[-1])]

        # 从低分辨率到高分辨率进行逐步上采样和融合
        for i in range(len(feature_maps) - 2, -1, -1):
            upsampled = F.interpolate(fpn_outs[-1], size=feature_maps[i].shape[2:], mode="nearest")
            lateral = self.lateral_convs[i](feature_maps[i])
            fpn_out = upsampled + lateral
            fpn_outs.append(self.output_convs[i](fpn_out))

        # 返回各分辨率的 FPN 输出，按从高分辨率到低分辨率的顺序
        return fpn_outs[::-1]

class self_net(nn.Module):
    def __init__(self, num_classes=3, base_channels=32):
        super(self_net, self).__init__()

        # 定义 HRNet 的各个分支
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 2
        self.stage2_high = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage2_low = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # Stage 3
        self.stage3_high = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage3_low1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.stage3_low2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Stage 4 (新增加的 stage)
        self.stage4_high = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage4_low1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.stage4_low2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.stage4_low3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # 使用 FPN 进行多尺度特征融合
        self.fpn = FPN(in_channels_list=[base_channels, base_channels * 2, base_channels * 4, base_channels * 8],
                       out_channels=base_channels)

        # Enhanced ASPP 模块
        self.aspp = EnhancedASPP(in_channels=base_channels, out_channels=base_channels * 4)

        # 输出头部
        self.final_layer = nn.Conv2d(base_channels * 4, num_classes, kernel_size=1)

    def forward(self, x):
        # 各个阶段
        x_stage1 = self.stage1(x)

        # Stage 2
        x_high = self.stage2_high(x_stage1)
        x_low = self.stage2_low(x_stage1)

        # Stage 3
        x_high = self.stage3_high(x_high)
        x_low1 = self.stage3_low1(x_low)
        x_low2 = self.stage3_low2(x_low1)

        # Stage 4
        x_high = self.stage4_high(x_high)
        x_low1 = self.stage4_low1(x_low1)
        x_low2 = self.stage4_low2(x_low2)
        x_low3 = self.stage4_low3(x_low2)

        # 使用 FPN 进行多尺度特征融合
        fpn_outs = self.fpn([x_high, x_low1, x_low2, x_low3])

        # 选择最高分辨率的 FPN 输出进入 ASPP
        x_fused = fpn_outs[0]

        # 通过 Enhanced ASPP 层
        x_aspp = self.aspp(x_fused)

        # 最后一层
        output = self.final_layer(x_aspp)

        # 上采样输出到目标尺寸
        output = F.interpolate(output, size=(200, 200), mode='bilinear', align_corners=True)
        return output

class EnhancedASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedASPP, self).__init__()

        # 1x1 卷积，保持尺度不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 空洞卷积，使用不同的膨胀率
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.atrous_block4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=24, dilation=24)  # 新增更大的膨胀率

        # 全局平均池化层，可以更好地捕获全局信息
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 输出层，融合所有分支的特征
        self.conv_out = nn.Conv2d(out_channels * 6, out_channels, kernel_size=1)  # 注意这里是6个分支

    def forward(self, x):
        # 各分支进行卷积操作
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.atrous_block1(x))
        x3 = F.relu(self.atrous_block2(x))
        x4 = F.relu(self.atrous_block3(x))
        x5 = F.relu(self.atrous_block4(x))  # 新增的空洞卷积分支
        x6 = self.global_avg_pool(x)  # 全局平均池化
        x6 = F.interpolate(x6, size=x4.size()[2:], mode='bilinear', align_corners=True)  # 上采样到原输入尺寸

        # 融合所有特征
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        return self.conv_out(x)

# 定义数据集类
class TestDataset(Dataset):
    def __init__(self, img_dir: str, transforms=None):
        super(TestDataset, self).__init__()
        assert os.path.exists(img_dir), f"Image path '{img_dir}' does not exist."

        self.img_dir = img_dir
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image=np.array(image))['image']

        return image, img_name

# 保存分割结果为 .npy 文件
def save_prediction_npy_files(loader, model, device, output_pred_dir):
    model.eval()  # 切换模型到评估模式
    model.to(device)  # 将模型移到指定设备上

    # 确保输出文件夹存在
    os.makedirs(output_pred_dir, exist_ok=True)

    with torch.no_grad():  # 关闭梯度计算
        for images, img_names in loader:
            images = images.to(device)  # 将输入数据也移动到同一设备

            # 模型预测
            outputs = model(images)
            if isinstance(outputs, dict):  # 检查输出是否为字典，获取 `out` 键
                outputs = outputs['out']
            preds = torch.argmax(outputs, dim=1)

            for pred, img_name in zip(preds, img_names):
                # 生成分割结果的文件名
                base_name = os.path.splitext(img_name)[0]
                pred_npy_path = os.path.join(output_pred_dir, f"c_prediction_{base_name}.npy")

                # 保存预测结果为 .npy 文件
                np.save(pred_npy_path, pred.cpu().numpy())
                print(f"Saved prediction to {pred_npy_path}")

def create_model(num_classes, input_size=(3, 200, 200)):
    model = self_net(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用 torchinfo.summary 获取模型信息
    summary(model, input_size=(1, *input_size), device=device.type)

    return model

# 获取数据集的 transforms
def get_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ])

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4  # 包括背景类
    height, width = 200, 200

    # 设置测试图像目录和输出分割结果目录
    test_img_dir = "./dataset/images/test"
    output_pred_dir = "./c_test_predictions"

    # 获取数据集和数据加载器
    test_transforms = get_transforms(height=height, width=width)
    test_dataset = TestDataset(img_dir=test_img_dir, transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)

    # 创建模型并加载权重
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load("model.pth"))

    # 保存预测的分割结果为 .npy 文件
    save_prediction_npy_files(test_loader, model, device, output_pred_dir)


if __name__ == '__main__':
    main()
