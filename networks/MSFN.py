import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inFe):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inFe, inFe, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(inFe, inFe, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x

class ResGroup(nn.Module):
    def __init__(self,numFe,numBK=4,k=3):
        super(ResGroup,self).__init__()
        G=numFe
        C=numBK
        convs_residual=[]
        for c in range(C):
            convs_residual.append(ResBlock(G))
        self.convs_residual=nn.Sequential(*convs_residual)
        self.last_conv=nn.Conv2d(G,G,k,padding=(k-1)//2, stride=1)

    def forward(self, x):
        x = self.last_conv(self.convs_residual(x)) + x
        return x

class Coder(nn.Module):
    def __init__(self,numFe):
        super(Coder, self).__init__()

        self.fe=nn.Sequential(
            nn.Conv2d(numFe, numFe, 3, 1, 1),
            nn.ReLU()
        )

        self.encoder1=nn.Sequential(*[
            nn.Conv2d(numFe,2*numFe,3,2,1),
            nn.ReLU()
        ])

        self.encoder2=nn.Sequential(*[
            nn.Conv2d(2*numFe,4*numFe,3,2,1),
            nn.ReLU()
        ])

        self.decoder1=nn.Sequential(*[
            nn.ConvTranspose2d(4*numFe,2*numFe,3,2,1,output_padding=1),
            nn.ReLU()
        ])

        self.decoder2=nn.Sequential(*[
            nn.ConvTranspose2d(2*numFe,numFe,3,2,1,output_padding=1),
            nn.ReLU()
        ])

        RG0=[ResBlock(4*numFe) for _ in range(4)]
        RG0.append(nn.Conv2d(4*numFe,4*numFe,3,1,1))

        RG1=[ResBlock(2*numFe) for _ in range(2)]
        RG1.append(nn.Conv2d(2*numFe,2*numFe,3,1,1))

        RG2=[ResBlock(numFe) for _ in range(1)]
        RG2.append(nn.Conv2d(numFe,numFe,3,1,1))

        self.RG0=nn.Sequential(*RG0)
        self.RG1 = nn.Sequential(*RG1)
        self.RG2 = nn.Sequential(*RG2)
        # self.RG1=nn.Conv2d(2*numFe,2*numFe,3,1,1)
        # self.RG2 = nn.Conv2d(numFe, numFe, 3, 1, 1)

    def forward(self,x):
        f1=self.fe(x)
        f2=self.encoder1(f1)
        f3=self.encoder2(f2)
        x1=self.RG0(f3) + f3
        x2=self.RG1(self.decoder1(x1))+f2
        x3=self.RG2(self.decoder2(x2))+f1

        return x1,x2,x3

class FB(nn.Module):
    def __init__(self,numFe):
        super(FB, self).__init__()
        self.fusion0=nn.Sequential(
            nn.Conv2d(8*numFe,numFe,1,1,0),
            nn.ReLU(),
            nn.ConvTranspose2d(numFe, numFe, 8, 4, 2)
        )

        self.fusion1=nn.Sequential(
            nn.Conv2d(4* numFe, numFe, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(numFe, numFe, 3,2,1,output_padding=1)
        )
        self.fusion2=nn.Conv2d(5*numFe, numFe, 1, 1, 0)
        self.act=nn.ReLU()

    def forward(self,x1,x2,x3,y1,y2,y3,information):
        f0=self.fusion0(torch.cat([x1,y1],dim=1))
        f1=self.fusion1(torch.cat([x2,y2],dim=1))
        f=self.act(self.fusion2(torch.cat([f0, f1,x3,y3,information], dim=1)))

        return f

class MSFN(nn.Module):
    def __init__(self, inCh =4,numFe=32,numResB=12):
        super(MSFN, self).__init__()

        self.ms_conv =nn.Conv2d(inCh, numFe, 3, 1, 1)
        self.pan_conv = nn.Conv2d(1, numFe, 3, 1, 1)
        self.mscoder=Coder(numFe)
        # self.pancoder = Coder(numFe)
        self.msf=FB(numFe)
        self.sr = nn.Sequential(*[
        ResBlock(numFe) for _ in range(numResB)
        ])

        self.re = nn.Conv2d(numFe, 4, 3, 1, 1)
        self.act=nn.ReLU()
        self.information=nn.Conv2d(5+2*numFe,numFe,3,1,1)

    def forward(self,ms,pan):
        ms = F.interpolate(ms, scale_factor=4, mode='bicubic', align_corners=False)
        ms_fe = self.ms_conv(ms)
        ms_fe_act=self.act(ms_fe)
        pan_fe = self.pan_conv(pan)
        pan_fe_act = self.act(pan_fe)
        ms1,ms2,ms3=self.mscoder(ms_fe_act)
        pan1,pan2,pan3=self.mscoder(pan_fe_act)
        information=self.information(torch.cat([ms,pan,ms_fe,pan_fe],dim=1))
        fusion=self.msf(ms1,ms2,ms3,pan1,pan2,pan3,information)
        res=self.sr(fusion)
        res=self.re(res)

        return res+ms
