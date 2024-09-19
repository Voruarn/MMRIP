import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from network.DFormer import DFormer_Tiny, DFormer_Small
from network.MLPDecoder import DecoderHead


class SimSOD(nn.Module):

    def __init__(self, dec_dim=512):
        super(SimSOD, self).__init__()

        self.encoder=DFormer_Small()
        enc_dims=[64, 128, 256, 512]
        self.decoder=DecoderHead(in_channels=enc_dims, num_classes=1, embed_dim=dec_dim)

    def forward(self, image, depth):
        H,W=image.shape[2:]
        outs=self.encoder(image, depth)
        out=self.decoder(outs)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        return out, torch.sigmoid(out)



if __name__ == '__main__':
    rgb = torch.randn([1, 3, 224, 224])
    depth = torch.randn([1, 1, 224, 224])
