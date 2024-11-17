import os
import time
from torchinfo import summary
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
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


class HRNetCustom(nn.Module):
    def __init__(self, num_classes=3, base_channels=32):
        super(HRNetCustom, self).__init__()

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


# 训练一个 epoch 的函数
def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, lr_scheduler, print_freq, scaler,
                    criterion):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(data_loader):
        images, masks = images.to(device), masks.to(device)
        masks = masks.long()  # 将 masks 转换为 Long 类型

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss.item()

        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item()}")

    return running_loss / len(data_loader), optimizer.param_groups[0]["lr"]

# 学习率调度器的函数
def create_lr_scheduler(optimizer, num_steps, epochs, warmup=True, warmup_epochs=5, warmup_factor=1e-3):
    def lr_lambda(step):
        if warmup and step < warmup_epochs * num_steps:
            alpha = float(step) / (warmup_epochs * num_steps)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 将 targets 转换为 Long 类型
        targets = targets.long()

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-BCE_loss)  # 防止 log(0)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 对输入使用 sigmoid 来处理二分类情况，如果是多分类需要 softmax
        if inputs.size(1) > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)

        # 将 ignore_index 的位置替换为 0，避免 one-hot 出错
        targets = torch.where(targets == self.ignore_index, torch.tensor(0).to(targets.device), targets)

        # 将 targets 转换为 one-hot 编码
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 确保 targets_one_hot 与 inputs 形状一致
        targets_one_hot = targets_one_hot.type_as(inputs)  # 确保数据类型一致
        inputs = inputs.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)

        # 计算 Dice 系数
        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets_one_hot.sum() + self.smooth)
        return 1 - dice

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ignore_index=255):
        super(DiceCrossEntropyLoss, self).__init__()
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # 计算 Cross-Entropy Loss
        ce_loss = self.ce_loss(inputs, targets)
        # 计算 Dice Loss
        dice_loss = self.dice_loss(inputs, targets)
        # 组合损失
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, dice_weight=0.5, ignore_index=-100):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # 计算 Focal Loss
        focal_loss = self.focal_loss(inputs, targets)
        # 计算 Dice Loss
        dice_loss = self.dice_loss(inputs, targets)
        # 综合损失
        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        return total_loss


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.transforms = transforms
        self.flag = "train" if train else "val"
        img_dir = os.path.join(root, "images", self.flag)
        mask_dir = os.path.join(root, "masks", self.flag)

        img_names = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
        self.valid_samples = []

        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))

            # 检查掩码文件是否存在
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file '{mask_path}' does not exist. Skipping this file.")
                continue

            # 加载并检查掩码类别
            mask = np.array(Image.open(mask_path).convert('L'))
            unique_classes = np.unique(mask)
            if np.all(np.isin(unique_classes, [0, 1, 2, 3])):  # 根据需要调整类别
                self.valid_samples.append((img_path, mask_path))
            else:
                print(f"Warning: Skipping sample with invalid mask classes: {unique_classes}")

        # 打印有效样本数量
        print(f"Total valid samples: {len(self.valid_samples)}")

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_samples[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask

    def __len__(self):
        return len(self.valid_samples)

    @staticmethod
    def collate_fn(batch):
        images, masks = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_masks = cat_list(masks, fill_value=255)
        return batched_imgs, batched_masks

def cat_list(tensors, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    batch_shape = (len(tensors),) + max_size
    batched_tensors = tensors[0].new(*batch_shape).fill_(fill_value)
    for tensor, pad_tensor in zip(tensors, batched_tensors):
        pad_tensor[..., :tensor.shape[-2], :tensor.shape[-1]].copy_(tensor)
    return batched_tensors

# 评估函数
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_images = 0
    total_time = 0

    with torch.no_grad():
        for images, masks in data_loader:
            start_time = time.time()
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # 判断 outputs 是否是字典，如果是字典，则提取 'out' 键的值
            if isinstance(outputs, dict):
                outputs = outputs['out']  # 使用 'out' 键提取预测结果
            else:
                outputs = outputs  # 如果不是字典，直接使用 outputs 作为结果

            preds = outputs.argmax(dim=1)
            total_time += time.time() - start_time

            # 累积混淆矩阵
            confusion_matrix += compute_confusion_matrix(preds, masks, num_classes)
            total_images += 1

    # 计算 IoU 时排除背景
    iou_per_class = compute_iou_from_confusion_matrix(confusion_matrix, num_classes)

    # 只计算 Class 1、2、3 的 mIoU
    mIoU = np.nanmean(iou_per_class)

    # 计算 FPS
    fps = total_images / total_time

    # 输出 Class 1、2、3 的 IoU
    print(f"Class 1 IoU: {iou_per_class[0]:.3f}, Class 2 IoU: {iou_per_class[1]:.3f}, Class 3 IoU: {iou_per_class[2]:.3f}")
    print(f"mIoU (Class 1-3): {mIoU:.3f}, FPS: {fps:.3f}")

    return iou_per_class, mIoU, fps

def create_model(num_classes, input_size=(3, 200, 200)):
    model = HRNetCustom(num_classes=num_classes)  # 使用自定义 HRNet 模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用 torchinfo.summary 获取模型信息
    summary(model, input_size=(1, *input_size), device=device.type)

    return model

# 计算混淆矩阵
def compute_confusion_matrix(preds, masks, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion_matrix[true_class, pred_class] = ((masks == true_class) & (preds == pred_class)).sum().item()
    return confusion_matrix

# 从混淆矩阵计算每类的 IoU
def compute_iou_from_confusion_matrix(confusion_matrix, num_classes):
    ious = []
    # 假设背景类的索引是 0，忽略第 0 类
    for i in range(1, num_classes):  # 从 1 开始
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # 忽略没有的类别
        else:
            ious.append(intersection / union)
    return ious


def get_transforms(height, width, train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 添加轻微高斯模糊
            A.MedianBlur(blur_limit=3, p=0.3),        # 添加中值模糊去噪
            A.ISONoise(p=0.3),                        # 添加噪声仿真
            A.ElasticTransform(p=0.3),  # 新增弹性变换
            A.RandomGamma(p=0.3),  # 新增伽马变换
            A.HueSaturationValue(p=0.3),  # 新增色调、饱和度变换
            A.Resize(height=height, width=width, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=height, width=width, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ])

# 添加测试代码部分
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1  # 包含背景类

    # 设置图像大小
    height, width =200, 200

    # 加载训练集和验证集
    train_transforms = get_transforms(height=height, width=width, train=True)
    val_transforms = get_transforms(height=height, width=width, train=False)

    train_dataset = DriveDataset("dataset", train=True, transforms=train_transforms)
    val_dataset = DriveDataset("dataset", train=False, transforms=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=4,
                                             pin_memory=True)

    # 创建模型
    model = create_model(num_classes=num_classes).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # 使用组合损失函数
    # 使用 Dice + Cross-Entropy 组合损失函数
    criterion = DiceCrossEntropyLoss(dice_weight=0.7, ignore_index=255)  # 可以调整 dice_weight

    # 学习率调度器
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    best_dice = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # 训练过程
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler, criterion=criterion)

        # 验证集评估
        iou_per_class, mIoU, fps = evaluate(model, val_loader, device=device, num_classes=num_classes)
        print(f"验证集：Epoch {epoch}, mIoU: {mIoU:.3f}, FPS: {fps:.3f}")

        if mIoU > best_dice:
            best_dice = mIoU
            torch.save(model.state_dict(), "model.pth")
    # 加载训练好的最佳模型
    model.load_state_dict(torch.load("model.pth"))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./dataset", help="DRIVE root")
    # exclude background

    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=800, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='model.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--base-channels", default=64, type=int, help="Base channels for HRNet")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)