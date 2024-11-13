from typing import List ,Tuple
import numpy as np
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
import os
from Pos_Former.utils.utils import Hypothesis

from .decoder import Decoder , PosDecoder
from .encoder import Encoder
from Pos_Former.datamodule import vocab , label_make_muti
import torch
from torch import nn


"""
    # in_channel:输入block之前的通道数
    # channel:在block中间处理的时候的通道数（这个值是输出维度的1/4)
    # channel * block.expansion:输出的维度
"""
class BottleNeck(nn.Module):
    expansion = 2
    def __init__(self,in_channel,channel,stride=1,C=32,downsample=None):
        super().__init__()
       #in_channel = 6
        self.conv1=nn.Conv2d(in_channel,channel,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(channel)

        self.conv2=nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=False,stride=1,groups=C)
        self.bn2=nn.BatchNorm2d(channel)

        self.conv3=nn.Conv2d(channel,channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(channel*self.expansion)

        self.relu=nn.ReLU(False)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x#torch.Size([16, 64, 2, 64])

        out=self.relu(self.bn1(self.conv1(x))) #bs,c,h,w
        out=self.relu(self.bn2(self.conv2(out))) #bs,c,h,w torch.Size([16, 128, 2, 64])
        out=self.relu(self.bn3(self.conv3(out))) #bs,4c,h,wtorch.Size([16, 256, 2, 64])torch.Size([16, 128, 2, 64])torch.Size([16, 512, 1, 32])

        if(self.downsample != None):
            residual=self.downsample(residual)

        out+=residual #torch.Size([16, 256, 2, 64]) + torch.Size([16, 256, 2, 64])torch.Size([16, 1024, 1, 16])
        return self.relu(out)

    
class ResNeXt(nn.Module):
    def __init__(self,block,layers,num_classes=1000):
        super().__init__()
        #定义输入模块的维度
        self.in_channel=64
        ### stem layer
        self.conv1=nn.Conv2d(6,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(False)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True)

        ### main layer
        self.layer1=self._make_layer(block,128,layers[0])
        self.layer2=self._make_layer(block,256,layers[1],stride=2)
        self.layer3=self._make_layer(block,512,layers[2],stride=2)
        self.layer4=self._make_layer(block,1024,layers[3],stride=2)

        #classifier
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Linear(1024*block.expansion,num_classes)
        self.softmax=nn.Softmax(-1)

    def forward(self,x):
        ##stem layer
        out=self.relu(self.bn1(self.conv1(x))) #bs,112,112,64 torch.Size([16, 6, 10, 256])
        out=self.maxpool(out) #bs,56,56,64 torch.Size([16, 64, 5, 128])

        ##layers:
        out=self.layer1(out) #bs,56,56,128*2torch.Size([16, 256, 2, 64])
        out=self.layer2(out) #bs,28,28,256*2
        out=self.layer3(out) #bs,14,14,512*2
        out=self.layer4(out) #bs,7,7,1024*2
        print(out.shape)
        ##classifier

        out=self.avgpool(out) #bs,1,1,1024*2
        print(out.shape)
        out=out.reshape(out.shape[0],-1) #bs,1024*2
        print(out.shape)
        out=self.classifier(out) #bs,1000
        print(out.shape)
        out=self.softmax(out)
        print(out.shape)
        return out

        
    
    def _make_layer(self,block,channel,blocks,stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和x的channel维度不匹配问题，即对残差结构的输入进行升维，在做残差相加的时候，必须保证残差的纬度与真正的输出维度（宽、高、以及深度）相同
        # 比如步长！=1 或者 in_channel!=channel&self.expansion
        downsample = None
        if(stride!=1 or self.in_channel!=channel*block.expansion):
            self.downsample=nn.Conv2d(self.in_channel,channel*block.expansion,stride=stride,kernel_size=1,bias=False)
        #第一个conv部分，可能需要downsample
        layers=[]
        layers.append(block(self.in_channel,channel,downsample=self.downsample,stride=stride))
        self.in_channel=channel*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channel,channel))
        return nn.Sequential(*layers)


def ResNeXt50(num_classes=1000):
    return ResNeXt(BottleNeck,[3,4,6,3],num_classes=num_classes)


def ResNeXt101(num_classes=1000):
    return ResNeXt(BottleNeck,[3,4,23,3],num_classes=num_classes)


def ResNeXt152(num_classes=1000):
    return ResNeXt(BottleNeck,[3,8,36,3],num_classes=num_classes)

""" 
if __name__ == '__main__':
    input=torch.randn(16, 6, 10, 256)
    resnext50=ResNeXt50(1000)
    # resnext101=ResNeXt101(1000)
    # resnext152=ResNeXt152(1000)
    out=resnext50(input)
    print(out.shape) """

    
class PosFormer(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.posdecoder = PosDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage
        
        )
        self.resnext = ResNeXt50(1000)
        self.save_path = 'attn_PosFormer'

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, logger
    ) -> Tuple[FloatTensor,FloatTensor,FloatTensor]:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]torch.Size([8, 6, 10, 256])
        feature = torch.cat((feature, feature), dim=0)  # torch.Size([4, 1, 58, 60])[2b, t, d] torch.Size([16, 6, 10, 256])
        mask = torch.cat((mask, mask), dim=0)#torch.Size([16, 6, 10])torch.Size([8, 4, 4])
        #feature = self.resnext(feature)  # [2b, t, d] torch.Size([16, 6, 10, 256])
        tgt_list=tgt.cpu().numpy().tolist()
        muti_labels=label_make_muti.tgt2muti_label(tgt_list)#[[1, 77, 4, 12, 5, 0], [1, 94, 22, 13, 0, 0], [1, 45, 4, 86, 5, 0], [1, 84, 82, 110, 105, 112], [1, 18, 14, 9, 14, 0], [1, 8, 16, 24, 0, 0], [1, 13, 11, 0, 0, 0], [1, 109, 4, 40, 5, 0], [2, 5, 12, 4, 77, 0], [2, 13, 22, 94, 0, 0], [2, 5, 86, 4, 45, 0], [2, 112, 105, 110, 82, 84], [2, 14, 9, 14, 18, 0], [2, 24, 16, 8, 0, 0], [2, 11, 13, 0, 0, 0], [2, 5, 40, 4, 109, 0]]

        muti_labels_tensor=torch.FloatTensor(muti_labels)   #[2b,l,5]
        #muti_labels_tensor=muti_labels_tensor.cuda()
        print("改")#torch.Size([8, 4])

        print(tgt.shape)#8, 4  [16, 6 torch.Size([2, 25])
        print(feature.shape)#torch.Size([16, 6, 10, 256]) torch.Size([2, 16, 46, 256])8, 4, 4, 256
        out, _ = self.decoder(feature, mask, tgt)
        print("aaa")
        print(out.shape)
         # 16 6 113 _ t orch.Size([128, 6, 60])
        out_layernum , out_pos, _ =self.posdecoder(feature, mask,tgt,muti_labels_tensor)
        print(out_layernum.shape)#torch.Size([16, 6, 5])
        print(out_pos.shape)#torch.Size([16, 6, 6])
        return out, out_layernum, out_pos   # [2b,l,vocab_size], [2b,l,5] and[2b,l,6]

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        seq_out= self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )

        return seq_out