import torch
from torch import nn
import torch.nn.functional as F



class MaskDecoder(nn.Module):
    def __init__(self, embed_dim=768, mask_predictor_hidden_dim=256, mask_output_size=(256, 256)):
        super(MaskDecoder, self).__init__()
        self.mask_output_size = mask_output_size
        self.embed_dim = embed_dim

        self.feat_adapt = nn.Sequential(
            nn.Linear(embed_dim, mask_predictor_hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(mask_predictor_hidden_dim, mask_predictor_hidden_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mask_predictor_hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(mask_predictor_hidden_dim, mask_predictor_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_predictor_hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mask_predictor_hidden_dim, mask_predictor_hidden_dim // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mask_predictor_hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(mask_predictor_hidden_dim // 2, mask_predictor_hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_predictor_hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(mask_predictor_hidden_dim // 2, mask_predictor_hidden_dim // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mask_predictor_hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(mask_predictor_hidden_dim // 4, mask_predictor_hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_predictor_hidden_dim // 4),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(mask_predictor_hidden_dim // 4, mask_predictor_hidden_dim // 8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mask_predictor_hidden_dim // 8),
            nn.ReLU(inplace=True)
        )

        self.final_pred = nn.Sequential(
            nn.ConvTranspose2d(mask_predictor_hidden_dim // 8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self.init_weights()

    def forward(self, feat):
        B, C, N = feat.shape
        assert N == 49

        feat_2d = feat.transpose(1, 2)
        feat_2d = self.feat_adapt(feat_2d)
        feat_2d = feat_2d.transpose(1, 2)
        feat_2d = feat_2d.view(B, -1, 7, 7)

        x = self.up1(feat_2d)
        x = self.fusion1(x)

        x = self.up2(x)
        x = self.fusion2(x)

        x = self.up3(x)
        x = self.fusion3(x)

        x = self.up4(x)
        x = self.final_pred(x)
        mask_pred = F.interpolate(x, size=self.mask_output_size, mode='bilinear', align_corners=False)

        return mask_pred

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)