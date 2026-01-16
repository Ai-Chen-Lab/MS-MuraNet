import torch
import torch.nn as nn
from model.resattention import res_cbam
from model.resnest import resnest
import torch.nn.functional as F

class ChannelShuffle(x, groups):
  def_init__(self, groups):
    super().__init__()
    self.groups = groups
  def forward(self, x):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // self.groups
    # Reshape -> shuffle -> flatten
    x = x.view(batchsize, self.groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

class FusedDecoderBlock(nn.Module):
  def_init(self, up_in_channels, enc_in_channels, out_channels):
    super().__init__()
    self.up_conv = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.conv2d(up_in_channels, enc_in_channels, 3, padding=1),
      nn.BatchNorm2d(enc_in_channels),
      nn.PReLU()
    )

    self.channel_att = nn.Sequential(
      nn.AdaptiveAvgPoo12d(1),
      nn.conv2d(enc_in_channels*2, enc_in_channels//4, 1),
      nn.PReLU(),
      nn.Conv2d(enc_in_channels//4, enc_in_channels, 1), 
      nn.sigmoid()
  )

  self.res_path = nn.Sequential(
    self._make_res_block(enc_in_channels, enc_in_channels), 
    self._make_res_block(enc_in_channels, enc_in_channels//2)
  )

  self.identity = nn.Sequential(
    nn.conv2d(enc_in_channels, out_channels, 1),
    nn.BatchNorm2d(out_channels)
  ) if enc_in_channels != out_channels else nn.Identity()

  self.detail_enhance = nn.Sequential(
    nn.Conv2d(enc_in_channels, out_channels, 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.PReLU(),
    ChannelShuffle(groups=2)
  )

  def_make_res_block(self, in_ch, out_ch):
    return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.PReLU(),
      ChannelShuffle(groups=2),
      nn.Conv2d(out_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.PReLU()
    )

  def forward(self, x, enc_feat):
    up_feat = self.up_conv(x) 
    att_map = self.channel att(torch.cat([up_feat, enc_feat], dim=1)) 
    fused = up_feat * att_map + enc_feat 
    deep_path = self.res_path(fused)
    shallow_path = self.detail_enhance(fused) 

    return deep path + shallow path + self.identity(fused) 

class ChannelShuffle2(x, groups):
  def_init__(self, groups):
    super().__init__()
    self.groups = groups
  def forward(self, x):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // self.groups
    # Reshape -> shuffle -> flatten
    x = x.view(batchsize, self.groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

class FusedDecoderBlock2(nn.Module):
  def_init(self, up_in_channels, enc_in_channels, out_channels):
    super().__init__()
    self.up_conv = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.conv2d(up_in_channels, enc_in_channels, 3, padding=1),
      nn.BatchNorm2d(enc_in_channels),
      nn.PReLU()
    )

    self.channel_att = nn.Sequential(
      nn.AdaptiveAvgPoo12d(1),
      nn.conv2d(enc_in_channels*2, enc_in_channels//4, 1),
      nn.PReLU(),
      nn.Conv2d(enc_in_channels//4, enc_in_channels, 1), 
      nn.sigmoid()
  )

  self.res_path = nn.Sequential(
    self._make_res_block(enc_in_channels, enc_in_channels), 
    self._make_res_block(enc_in_channels, enc_in_channels//2)
  )

  self.identity = nn.Sequential(
    nn.conv2d(enc_in_channels, out_channels, 1),
    nn.BatchNorm2d(out_channels)
  ) if enc_in_channels != out_channels else nn.Identity()

  self.detail_enhance = nn.Sequential(
    nn.Conv2d(enc_in_channels, out_channels, 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.PReLU(),
    ChannelShuffle(groups=2)
  )

  def_make_res_block(self, in_ch, out_ch):
    return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.PReLU(),
      ChannelShuffle(groups=2),
      nn.Conv2d(out_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.PReLU()
    )

  def forward(self, x, enc_feat):
    up_feat = self.up_conv(x) 
    att_map = self.channel att(torch.cat([up_feat, enc_feat], dim=1)) 
    fused = up_feat * att_map + enc_feat 
    deep_path = self.res_path(fused)
    shallow_path = self.detail_enhance(fused) 

    return deep path + shallow path + self.identity(fused) 


class AttentionUNetBlock(nn.Module):
  def_init__(self, in_ch, out_ch):super(AttentionuNetBlock, self).__init__()

    self.conv1 = nn.conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)self.bn2 = nn.BatchNorm2d(out_ch)
    
    self.attention = nn.Sequential(
      nn.Conv2d(out_ch, out_ch, kernel_size=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_ch, 1, kernel_size=1),
      nn.sigmoid()
    )

    self.final_conv = nn.conv2d(out_ch, out_ch, kernel_size=1)

  def forward(self, x, skip_connection):
   
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    attention_map = self.attention(x)
    x = x * attention_map 
    x = self.final_conv(x)
    x = x + skip_connection 
    return x

class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        resnet = resnest(depth=34)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.convbg_1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, kernel_size=3, dilation=4, padding=4)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)


        self.decoder_blocks = nn.ModuleList([

          FusedDecoderBlock(up_in_channels=512, enc_in_channels=512, out_channels=256),
          FusedDecoderBlock(up_in_channels=256, enc_in_channels=256, out_channels=128),
          FusedDecoderBlock(128, 128, 64),
          FusedDecoderBlock2(64, 64, 64),
          FusedDecoderBlock2(64, 64, 64)
        ])
        self.convatt = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.salb = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.sal4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.sal3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sal2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sal1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sal0 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsampleb = nn.Upsample(scale_factor=32, mode='bilinear')

        self.attunet = AttentionUNetBlock(in_ch=1, out_ch=64)

    def forward(self, x):

        tx = self.conv1(x)
        tx = self.bn1(tx)
        f0 = self.relu(tx)
        tx = self.maxpool(f0)

        f1 = self.encoder1(tx)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = self.encoder4(f3)  

        tx = self.relubg_1(self.bnbg_1(self.convbg_1(f4)))
        tx = self.relubg_m(self.bnbg_m(self.convbg_m(tx)))
        outb = self.relubg_2(self.bnbg_2(self.convbg_2(tx)))        

        dec_feat = outb 
        dec_feat = self.decoder_blocks[0](dec_feat, f4) 
        out4 = dec_feat
        dec_feat = self.decoder_blocks[1](dec_feat, f3)
        out3 = dec_feat
        dec_feat = self.decoder_blocks[2](dec_feat, f2) 
        out2 = dec_feat
        dec_feat = self.decoder_blocks[3](dec_feat, f1) 
        out1 = dec_feat
        dec_feat = self.decoder_blocks[4](dec_feat, f0)
        out0 = dec_feat
      
        oute1 = self.convatt(out0)
      
        salb = self.salb(outb)  
      
        salb = self.upsampleb(salb)         
      

        sal4 = self.sal4(out4) 
        sal4 = self.upsample4(sal4)        

        sal3 = self.sal3(out3)
        sal3 = self.upsample3(sal3)         

        sal2 = self.sal2(out2)
        sal2 = self.upsample2(sal2)       

        sal1 = self.sal1(out1)
        sal1 = self.upsample1(sal1)        

        sal0 = self.sal0(out0)              

        sal_out = self.attunet(sal0, out01)    

        return torch.sigmoid(sal_out), torch.sigmoid(sal0), torch.sigmoid(sal1), torch.sigmoid(sal2), torch.sigmoid(sal3), torch.sigmoid(sal4), torch.sigmoid(salb)

