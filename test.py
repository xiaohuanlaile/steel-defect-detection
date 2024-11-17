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

# 计算混淆矩阵
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# 计算每个类别的 IoU
def per_class_iu(hist):
    print('Defect class IoU as follows:')
    print(np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1))
    return np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1)

# 计算每个类别的准确率
def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

# 计算 mIoU 和 mPA
def compute_mIoU(gt_dir, pred_dir):
    num_classes = 4  # 根据任务类别数
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    npy_name_list = [f.split('_')[1].split('.')[0] for f in os.listdir(pred_dir) if f.endswith('.npy')]

    gt_npy_files = [os.path.join(gt_dir, f"ground_truth_{x}.npy") for x in npy_name_list]
    pred_npy_files = [os.path.join(pred_dir, f"prediction_{x}.npy") for x in npy_name_list]

    for ind in range(len(gt_npy_files)):
        if not os.path.isfile(gt_npy_files[ind]):
            print(f"Ground truth file not found: {gt_npy_files[ind]}")
            continue

        if not os.path.isfile(pred_npy_files[ind]):
            print(f"Prediction file not found: {pred_npy_files[ind]}")
            continue

        pred = np.load(pred_npy_files[ind])
        label = np.load(gt_npy_files[ind])

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                len(label.flatten()), len(pred.flatten()), gt_npy_files[ind],
                pred_npy_files[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 4)) +
          '; mPA: ' + str(round(np.nanmean(mPA) * 100, 4)))

    return mIoUs

# 计算 IoU
def get_iandu_array(pred, ann, classIdx: int):
    if isinstance(pred, torch.Tensor): pred = pred.numpy()
    if isinstance(ann, torch.Tensor): ann = ann.numpy()
    i = np.sum(np.logical_and(np.equal(pred, ann), np.equal(ann, classIdx)))
    u = np.sum(np.logical_or(np.equal(pred, classIdx), np.equal(ann, classIdx)))
    return i, u

# 计算多个类别的 IoU
def get_ious_dir(preds_dir: str, anns_dir: str):
    preds = sorted([os.path.join(preds_dir, p) for p in os.listdir(preds_dir) if p.endswith('.npy')])
    anns = sorted([os.path.join(anns_dir, a) for a in os.listdir(anns_dir) if a.endswith('.npy')])

    i1, u1, i2, u2, i3, u3 = 0, 0, 0, 0, 0, 0
    for pred_file, ann_file in zip(preds, anns):
        pred = np.load(pred_file)
        ann = np.load(ann_file)

        i, u = get_iandu_array(pred, ann, 1)
        i1, u1 = i1 + i, u1 + u

        i, u = get_iandu_array(pred, ann, 2)
        i2, u2 = i2 + i, u2 + u

        i, u = get_iandu_array(pred, ann, 3)
        i3, u3 = i3 + i, u3 + u

    return i1 / u1, i2 / u2, i3 / u3

# 定义数据集类
class DriveDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, transforms=None):
        super(DriveDataset, self).__init__()
        assert os.path.exists(img_dir), f"Image path '{img_dir}' does not exist."
        assert os.path.exists(mask_dir), f"Mask path '{mask_dir}' does not exist."

        img_names = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
        self.img_list = [os.path.join(img_dir, i) for i in img_names]
        self.mask_list = [os.path.join(mask_dir, i.replace(".jpg", ".png")) for i in img_names]

        for mask in self.mask_list:
            if not os.path.exists(mask):
                raise FileNotFoundError(f"Mask file {mask} does not exist.")

        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transforms is not None:
            augmented = self.transforms(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# 保存 ground truth 和 prediction 的 .npy 文件
def save_npy_files(loader, model, device, output_pred_dir, output_gt_dir, img_list):
    model.eval()  # 切换模型到评估模式
    model.to(device)  # 将模型移到指定设备上

    # 确保输出文件夹存在
    if not os.path.exists(output_pred_dir):
        os.makedirs(output_pred_dir)
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)

    with torch.no_grad():  # 关闭梯度计算
        for idx, (image, mask) in enumerate(loader):
            image = image.to(device)  # 将输入数据也移动到同一设备
            mask = mask.to(device)

            # 模型预测
            output = model(image)
            if isinstance(output, dict):  # 检查输出是否为字典，获取 `out` 键
                output = output['out']
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 使用 img_list 提取文件名（去掉扩展名）
            img_name = os.path.basename(img_list[idx]).split('.')[0]

            # 保存 ground truth 和预测的 .npy 文件
            gt_npy_path = os.path.join(output_gt_dir, f"ground_truth_{img_name}.npy")
            pred_npy_path = os.path.join(output_pred_dir, f"prediction_{img_name}.npy")

            np.save(gt_npy_path, mask.squeeze(0).cpu().numpy())  # 保存 ground truth
            np.save(pred_npy_path, pred)  # 保存预测结果

            print(f"Saved ground truth to {gt_npy_path} and prediction to {pred_npy_path}")

def create_model(num_classes, input_size=(3, 200, 200)):
    model = HRNetCustom(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用 torchinfo.summary 获取模型信息
    summary(model, input_size=(1, *input_size), device=device.type)

    return model





# 获取数据集的 transforms
def get_transforms(height, width, train=False):
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ])

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4  # 包括背景类
    height, width = 200,200

    img_dir = "./dataset/images/val"
    mask_dir = "./dataset/masks/val"

    test_transforms = get_transforms(height=height, width=width, train=False)
    test_dataset = DriveDataset(img_dir=img_dir, mask_dir=mask_dir, transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)

    # 创建模型并加载权重
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load("model.pth"))


    # 保存 ground truth 和 prediction 为 .npy 文件
    output_pred_dir = 'test_predictions'
    output_gt_dir = 'test_ground_truths'


    save_npy_files(test_loader, model, device, output_pred_dir, output_gt_dir, test_dataset.img_list)

    # 计算并输出 IoU
    iou1, iou2, iou3 = get_ious_dir(output_gt_dir, output_pred_dir)
    print(f"Class 1 IoU: {iou1:.3f}, Class 2 IoU: {iou2:.3f}, Class 3 IoU: {iou3:.3f}")

    # 计算并输出 mIoU
    mIoUs = compute_mIoU(output_gt_dir, output_pred_dir)
    print(f"mIoU: {np.nanmean(mIoUs):.3f}")



if __name__ == '__main__':
    main()