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
    'tra_lbl_dir': '',     
    'image_ext': '',
    'label_ext': '',
    'checkpoint': '',
}

chkpt_dir = args['checkpoint']
func.check_mkdir(chkpt_dir)

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def train_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + ssim_out + iou_out
    return loss

def muti_loss_fusion(s_out, s0, s1, s2, s3, s4, sb, labels_v):
    loss0 = train_loss(s_out, labels_v)
    loss1 = train_loss(s0, labels_v)
    loss2 = train_loss(s1, labels_v)
    loss3 = train_loss(s2, labels_v)
    loss4 = train_loss(s3, labels_v)
    loss5 = train_loss(s4, labels_v)
    loss6 = train_loss(sb, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss

def collate_fn(batch):
    for item in batch:
        item['image'] = item['image'].float()
        item['label'] = item['label'].float()
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }

def main():
    logging.info(args)
    print(args)

    tra_img_name_list = glob.glob(os.path.join(args['tra_img_dir'], '*' + args['image_ext']))
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
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            RandomCrop(224),
            ToTensorLab(flag=0)
        ])
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['workers'],
        collate_fn=collate_fn  # 使用自定义 collate_fn
    )

    net = Net(in_channels=3).cuda()

    optimizer = optim.RMSprop(net.parameters(), lr=args['lr'], alpha=0.9)

    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    net.train()

    for epoch in range(args['epoch']):
        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            ite_num4val += 1
            inputs_v, labels_v = data['image'], data['label']
            inputs_v = inputs_v.cuda()
            labels_v = labels_v.cuda()

            optimizer.zero_grad()

            s_out, s0, s1, s2, s3, s4, sb = net(inputs_v)
            loss2, loss = muti_loss_fusion(s_out, s0, s1, s2, s3, s4, sb, labels_v)
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

        if (epoch + 1) % 20 == 0:
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
