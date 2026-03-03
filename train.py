# coding=utf-8
import torch
import torch.nn as nn
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from data_loader_oral import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from model.MS-MuraNet import Net
import pytorch_ssim
import pytorch_iou
from torch.backends import cudnn
import utils.func as func
import os
import logging
torch.autograd.set_detect_anomaly(True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training6_1.log', filemode='w')  # 将日志写入文件

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

cudnn.benchmark = True
torch.manual_seed(2018)
torch.cuda.manual_seed_all(2018)

args = {
    'epoch': 100,
    'batch_size': 8,
    'lr': 0.001,
    'workers': 5,
    'tra_img_dir': '',              
    'tra_the_dir': '',
    'tra_his_dir': '',
    'tra_lbl_dir': '',     
    'image_ext': '.jpg',
    'label_ext': '.png',
    'checkpoint': '',
}

chkpt_dir = args['checkpoint']
func.check_mkdir(chkpt_dir)

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, pred, target):
        smooth = 1e-5
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1. - dice

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, pred, target):
        def gradient(x):
            h_x = x.size()[2]
            w_x = x.size()[3]
            
            # 计算水平梯度
            gradient_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            # 计算垂直梯度
            gradient_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
            
            # 对称填充以保持尺寸一致
            gradient_x = F.pad(gradient_x, (0, 1, 0, 0), mode='replicate')
            gradient_y = F.pad(gradient_y, (0, 0, 0, 1), mode='replicate')
            
            gradient = torch.cat([gradient_x, gradient_y], dim=1)
            return gradient
        
        pred_grad = gradient(pred)
        target_grad = gradient(target)
        
        # 确保尺寸一致
        if pred_grad.size() != target_grad.size():
            target_grad = F.interpolate(target_grad, size=pred_grad.size()[2:], mode='bilinear', align_corners=True)
        
        loss = F.l1_loss(pred_grad, target_grad)
        return loss
    
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
gradient_loss = GradientLoss()

def train_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    gradient_out = gradient_loss(pred, target)
    loss = bce_out + ssim_out + iou_out + 0.5 * gradient_out
    return loss

def muti_loss_fusion(s_out, s0, s1, s2, s3, sb, labels_v):
    loss0 = train_loss(s_out, labels_v)
    loss1 = train_loss(s0, labels_v)
    loss2 = train_loss(s1, labels_v)
    loss3 = train_loss(s2, labels_v)
    loss4 = train_loss(s3, labels_v)
    loss5 = train_loss(sb, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
    return loss0, loss

def collate_fn(batch):
    for item in batch:
        item['image'] = item['image'].float()
        item['thermal'] = item['thermal'].float()
        item['hist'] = item['hist'].float()
        item['label'] = item['label'].float()
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'thermal': torch.stack([b['thermal'] for b in batch]),
        'hist': torch.stack([b['hist'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }

def main():
    logging.info(args)  
    print(args)

    tra_img_name_list = glob.glob(os.path.join(args['tra_img_dir'], '*' + args['image_ext']))
    tra_his_name_list = glob.glob(os.path.join(args['tra_his_dir'], '*' + args['image_ext']))
    tra_the_name_list = glob.glob(os.path.join(args['tra_the_dir'], '*' + args['image_ext']))
    tra_lbl_name_list = [os.path.join(args['tra_lbl_dir'], os.path.splitext(os.path.basename(img))[0] + args['label_ext'])
                         for img in tra_img_name_list]
    logging.info('**********************************************')
    logging.info(f'train images: {len(tra_img_name_list)}')
    logging.info(f'train labels: {len(tra_lbl_name_list)}')
    logging.info('**********************************************')

    print('**********************************************')
    print('train images: ', len(tra_img_name_list))
    print('train labels: ', len(tra_lbl_name_list))
    print('**********************************************')

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        tra_the_dir=tra_the_name_list,
        tra_his_dir=tra_his_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            RandomCrop(256),
            ToTensorLab(flag=0)
        ])
    )
    # sample = next(iter(salobj_dataset))
    # print("Keys in data sample:", sample.keys())
    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['workers'],
        collate_fn=collate_fn  # 使用自定义 collate_fn
    )

    net = Net(in_channels=3).cuda()

    optimizer = optim.RMSprop(net.parameters(), lr=args['lr'], alpha=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.1)

    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    net.train()

    for epoch in range(args['epoch']):
        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            ite_num4val += 1
            images = data['image'].cuda()
            thermals = data['thermal'].cuda()
            hist = data['hist'].cuda()
            labels = data['label'].cuda()

            optimizer.zero_grad()

            s_out, s0, s1, s2, s3, s4, sb = net(images, thermals, hist)
            loss2, loss = muti_loss_fusion(s_out, s0, s1, s2, s3, s4, sb, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss2.item()

            logging.info("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %.4f, predicted loss: %.4f" % (
                epoch + 1, args['epoch'], (i + 1) * args['batch_size'], len(tra_img_name_list), ite_num,
                running_loss / ite_num4val, running_tar_loss / ite_num4val))
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %.4f, predicted loss: %.4f" % (
                epoch + 1, args['epoch'], (i + 1) * args['batch_size'], len(tra_img_name_list), ite_num,
                running_loss / ite_num4val, running_tar_loss / ite_num4val))

        scheduler.step()

        if (epoch + 1) % 1 == 0:
            torch.save(net.state_dict(), os.path.join(
                args['checkpoint'],
                f"Net_epoch_{epoch + 1}_trnloss_{running_loss / ite_num4val:.4f}_priloss_{running_tar_loss / ite_num4val:.4f}.pth"))
            running_loss = 0.0
            running_tar_loss = 0.0
            ite_num4val = 0
    logging.info('-------------Training Finish-------------')

    print('-------------Training Finish-------------')


if __name__ == "__main__":
    main()
