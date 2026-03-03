import torch
from torch import nn, einsum
import torchvision.models as models
from einops import rearrange
import clip
from transformers import MambaConfig
from model.mamba_block import CaMambaModel
# from transformers.models.mamba.modeling_mamba import MambaRMSNorm
class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network == 'HyberCLIPCNN':  # 混合CNN架构
            # 初始化ResNet50 backbone
            #self.feature_fusion = FeatureFusionModule()
            resnet = models.resnet50(pretrained=True)
            # 移除最后的池化层和分类层
            self.cnn = nn.Sequential(*list(resnet.children())[:-2])
            
            modules = list(self.cnn.children())[:-1]
            resnet_block1 = list(self.cnn.children())[:5]
            self.resnet_block1 = nn.Sequential(*resnet_block1)
            resnet_block2 = list(self.cnn.children())[5]
            self.resnet_block2 = nn.Sequential(*resnet_block2)
            resnet_block3 = list(self.cnn.children())[6]
            self.resnet_block3 = nn.Sequential(*resnet_block3)
            resnet_block4 = list(self.cnn.children())[7]
            self.resnet_block4 = nn.Sequential(*resnet_block4)
            
            #clip_model_type = self.network.replace('CLIP-', '')
            self.clip_model, preprocess = clip.load("ViT-B/32", jit=False)  #
            self.clip_model = self.clip_model.to(dtype=torch.float32)
            self.cace_fusionA = CACE(clip_dim=768, cnn_dim=2048)
            self.cace_fusionB = CACE(clip_dim=768, cnn_dim=2048)

        elif 'CLIP-' in self.network:
            clip_model_type = self.network.replace('CLIP-', '')
            self.clip_model, preprocess = clip.load(clip_model_type, jit=False)  #
            self.clip_model = self.clip_model.to(dtype=torch.float32)

        # self.cnn_list = nn.ModuleList(modules)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        if "CLIP-" in self.network:
            img_A = imageA.to(dtype=torch.float32)
            img_B = imageB.to(dtype=torch.float32)
            clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
            clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)

        elif "HyberCLIPCNN" in self.network:
            img_A = imageA.to(dtype=torch.float32)
            img_B = imageB.to(dtype=torch.float32)
            clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
            clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)
            feat1_list = []
            feat2_list = []
            
            outA1 = self.resnet_block1(img_A)  # 256
            outA2 = self.resnet_block2(outA1)  # 512
            outA3 = self.resnet_block3(outA2)  # 1024
            outA4 = self.resnet_block4(outA3)  # 2048
            
            outB1 = self.resnet_block1(img_B)  # 256
            outB2 = self.resnet_block2(outB1)  # 512
            outB3 = self.resnet_block3(outB2)  # 1024
            outB4 = self.resnet_block4(outB3)  # 2048
            
            #image_feat_A = self.cace_fusionA(img_feat_A, outA4) 
            #image_feat_B = self.cace_fusionB(img_feat_B, outB4) 
            #img_feat_A1 = self.feature_fusion(clip_emb_A, outA3, outA4)  # [B, 49, 768]
            #img_feat_B1 = self.feature_fusion(clip_emb_B, outB3, outB4)  # [B, 49, 768]
            
            return img_feat_A, img_feat_B,outA1,outA2,outA3,outA4,outB1,outB2,outB3,outB4
        
        
        else:
            # feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            # feat2 = self.cnn(imageB)
            feat1 = imageA
            feat2 = imageB
            feat1_list = []
            feat2_list = []
            cnn_list = list(self.cnn.children())
            for module in cnn_list:
                feat1 = module(feat1)
                feat2 = module(feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
            feat1_list = feat1_list[-4:]
            feat2_list = feat2_list[-4:]

        return img_feat_A, img_feat_B

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 3 through 4
        if 'CLIP-' in self.network and fine_tune:
            for p in self.clip_model.parameters():
                p.requires_grad = True
            # If fine-tuning, only fine-tune last 2 trans and ln_post
            children_list = list(self.clip_model.visual.transformer.resblocks.children())[-6:]
            children_list.append(self.clip_model.visual.ln_post)
            for c in children_list:
                for p in c.parameters():
                    p.requires_grad = True
        elif 'HyberCLIPCNN' in self.network and fine_tune:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune last 2 trans and ln_post
            children_list = list(self.clip_model.visual.transformer.resblocks.children())[-6:]
            children_list.append(self.clip_model.visual.ln_post)
            for c in children_list:
                for p in c.parameters():
                    p.requires_grad = True

            for r in list(self.cnn.children())[:]:
                for p in r.parameters():
                    p.requires_grad = fine_tune
            for p in self.cace_fusionA.parameters():
                p.requires_grad = True
            for p in self.cace_fusionB.parameters():
                p.requires_grad = True
        elif 'CLIP-' not in self.network and fine_tune:
            for c in list(self.cnn.children())[:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


from einops import rearrange

class CACE(nn.Module):
    """
    Consistency-Aware Cross-Modal Enhancement (CACE)
    - Input:
        clip_tokens: [B, 49, 768]  (from CLIP ViT)
        cnn_feat:    [B, 2048, 7, 7] (from ResNet outA4)
    - Output:
        fused_tokens: [B, 49, 768]
    """
    def __init__(self, clip_dim=768, cnn_dim=2048):
        super().__init__()
        # Project CNN to same dim as CLIP for inner product
        self.cnn_proj = nn.Conv2d(cnn_dim, clip_dim, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, clip_tokens, cnn_feat):
        B, L, D = clip_tokens.shape  # L=49, D=768
        assert L == 49, "Assumes 7x7=49 patches"

        # Step 1: Project CNN feature to 768-d and reshape to tokens
        cnn_proj = self.cnn_proj(cnn_feat)  # [B, 768, 7, 7]
        cnn_tokens = rearrange(cnn_proj, 'b d h w -> b (h w) d')  # [B, 49, 768]

        # Step 2: Compute inner-product correlation map (semantic consistency)
        # This is a [B, 49] map: high value = both modalities agree on change
        corr = torch.sum(clip_tokens * cnn_tokens, dim=-1, keepdim=True)  # [B, 49, 1]
        corr = corr / (D ** 0.5)  # scale for stability (like in attention)

        # Step 3: Apply symmetric, ReLU-gated enhancement
        # Use original features for both sides (symmetric!)
        enhanced_clip = clip_tokens + cnn_tokens * corr
        enhanced_cnn  = cnn_tokens  + clip_tokens * corr

        # Step 4: Apply ReLU (as in your CMCE)
        enhanced_clip = self.relu(enhanced_clip)
        enhanced_cnn  = self.relu(enhanced_cnn)

        # Step 5: Fuse — since captioning uses CLIP-side, we return enhanced CLIP
        # (You could also average, but CLIP is language-aligned)
        fused_tokens = enhanced_clip  # [B, 49, 768]

        return fused_tokens

class FeatureFusionModule(nn.Module):

    def __init__(self):
        super(FeatureFusionModule, self).__init__()

        self.outA3_proj = nn.Sequential(
            nn.Conv2d(1024, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.outA4_proj = nn.Sequential(
            nn.Conv2d(2048, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.clip_proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(inplace=True)
        )

        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, 1),  # 融合outA3和outA4
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.final_fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, 1),  # 融合多尺度特征和CLIP特征
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

    def forward(self, clip_feat, outA3, outA4):

        batch_size = clip_feat.size(0)

        projected_outA3 = self.outA3_proj(outA3)  # [B, 768, H/16, W/16]
        projected_outA3 = nn.functional.interpolate(
            projected_outA3, size=outA4.shape[2:], mode='bilinear', align_corners=False
        )  # [B, 768, H/32, W/32]

        projected_outA4 = self.outA4_proj(outA4)  # [B, 768, H/32, W/32]


        multiscale_feat = self.multiscale_fusion(
            torch.cat([projected_outA3, projected_outA4], dim=1)
        )

        # 将CLIP全局特征扩展到空间维度
        clip_spatial = self.clip_proj(clip_feat)  # [B, 768]
        clip_spatial = clip_spatial.view(batch_size, 768, 1, 1)  # [B, 768, 1, 1]
        clip_spatial = clip_spatial.expand(-1, -1, multiscale_feat.size(2),
                                           multiscale_feat.size(3))  # [B, 768, H/32, W/32]


        final_feat = self.final_fusion(
            torch.cat([multiscale_feat, clip_spatial], dim=1)
        )  # [B, 768, H/32, W/32]

        batch, c, h, w = final_feat.shape
        final_feat = final_feat.view(batch, c, -1).transpose(-1, 1)  # [B, HW, C] -> [B, 49, 768]

        return final_feat

class resblock(nn.Module):
    '''
    module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, int(outchannel / 1), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), int(outchannel / 1), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return self.act(out)


class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, dropout=0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.n_layers = n_layers
        self.channels = channels
        # position embedding
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        # Mamba
        config_1 = MambaConfig(num_hidden_layers=1, conv_kernel=3,hidden_size=channels)
        config_2 = MambaConfig(num_hidden_layers=1, conv_kernel=3,hidden_size=channels)
        self.CaMalayer_list = nn.ModuleList([])
        self.fuselayer_list = nn.ModuleList([])
        self.fuselayer_list_2 = nn.ModuleList([])
        self.linear_dif = nn.ModuleList([])
        self.linear_img1 = nn.ModuleList([])
        self.linear_img2 = nn.ModuleList([])
        self.Dyconv_img1_list = nn.ModuleList([])
        self.Dyconv_img2_list = nn.ModuleList([])
        embed_dim = channels
        self.Conv1_list = nn.ModuleList([])
        self.LN_list = nn.ModuleList([])
        for i in range(n_layers):
            self.CaMalayer_list.append(nn.ModuleList([
                CaMambaModel(config_1),
                CaMambaModel(config_1),
            ]))
            self.fuselayer_list.append(nn.ModuleList([
                CaMambaModel(config_2),
                CaMambaModel(config_2),
            ]))
            # self.linear_dif.append(nn.Sequential(
            #     nn.Linear(channels, channels),
            #     # nn.SiLU(),
            # ))
            # self.Dyconv_img1_list.append(Dynamic_conv(channels))
            # self.Dyconv_img2_list.append(Dynamic_conv(channels))
            # self.Dyconv_dif_list.append(Dynamic_conv(channels))
            # self.linear_img1.append(nn.Linear(2*channels, channels))
            # self.linear_img2.append(nn.Linear(2*channels, channels))
            self.Conv1_list.append(nn.Conv2d(channels * 2, embed_dim, kernel_size=1))
            self.LN_list.append(resblock(embed_dim, embed_dim))
        self.act = nn.Tanh()
        self.layerscan = CaMambaModel(config_1)
        self.LN_norm = nn.LayerNorm(channels)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.alpha2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Fusion bi-temporal feat for captioning decoder
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos_embedding(self, x):
        if len(x.shape) == 3: # NLD
            b = x.shape[0]
            c = x.shape[-1]
            x = x.transpose(-1, 1).view(b, c, self.h_feat, self.w_feat)
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        # reshape back to NLD
        x = x.view(b, c, -1).transpose(-1, 1)  # NLD (b,hw,c)
        return x

    def forward(self, img_A, img_B):
        h, w = self.h_feat, self.w_feat

        # 1. A B feature from backbone  NLD
        img_A = self.add_pos_embedding(img_A)
        img_B = self.add_pos_embedding(img_B)

        # captioning
        batch, c = img_A.shape[0], img_A.shape[-1]
        img_sa1, img_sa2 = img_A, img_B

        # Method: Mamba
        # self.CaMalayer_list.train()
        img_list = []
        N, L, D = img_sa1.shape
        for i in range(self.n_layers):
            # SD-SSM:
            dif = img_sa2 - img_sa1
            img_sa1 = self.CaMalayer_list[i][0](inputs_embeds=img_sa1, inputs_embeds_2=dif).last_hidden_state
            img_sa2 = self.CaMalayer_list[i][1](inputs_embeds=img_sa2, inputs_embeds_2=dif).last_hidden_state

            # TT-SSM:
            scan_mode = 'TT-SSM'
            if scan_mode == 'TT-SSM':
                img_sa1 = self.LN_norm(img_sa1)#+img_sa1_res
                img_sa2 = self.LN_norm(img_sa2)#+img_sa2_res
                img_sa1_res = img_sa1
                img_sa2_res = img_sa2
                img_fuse1 = img_sa1.permute(0, 2, 1).unsqueeze(-1) # (N,D,L,1)
                img_fuse2 = img_sa2.permute(0, 2, 1).unsqueeze(-1)
                img_fuse = torch.cat([img_fuse1, img_fuse2], dim=-1).reshape(N, D, -1) # (N,D,L*2)
                img_fuse = self.fuselayer_list[i][0](inputs_embeds=img_fuse.permute(0, 2, 1)).last_hidden_state.permute(0, 2, 1) # (N,D,L*2)
                img_fuse = img_fuse.reshape(N, D, L, -1)

                img_sa1 = img_fuse[..., 0].permute(0, 2, 1)#[...,:D] # (N,L,D)
                img_sa2 = img_fuse[..., 1].permute(0, 2, 1)#[...,:D]
                #
                img_sa1 = self.LN_norm(img_sa1) + img_sa1_res*self.alpha
                img_sa2 = self.LN_norm(img_sa2) + img_sa2_res*self.alpha

            # # bitemporal fusion
            if i == self.n_layers-1:
                img1_cap = img_sa1.transpose(-1, 1).view(batch, c, h, w)
                img2_cap = img_sa2.transpose(-1, 1).view(batch, c, h, w)
                feat_cap = torch.cat([img1_cap, img2_cap], dim=1)
                feat_cap = self.LN_list[i](self.Conv1_list[i](feat_cap))
                # feat_cap = self.Conv1_list[i](feat_cap)
                img_fuse = feat_cap.view(batch, c, -1).transpose(-1, 1)#.unsqueeze(-1) # (batch_size, L, D)
                img_fuse = self.LN_norm(img_fuse).unsqueeze(-1)
                img_list.append(img_fuse)

        # Out
        feat_cap = img_list[-1][..., 0]
        feat_cap = feat_cap.transpose(-1, 1)
        return feat_cap

if __name__ == '__main__':
    # test
    img_A = torch.randn(16, 49, 768).cuda()
    img_B = torch.randn(16, 49, 768).cuda()
    encoder = AttentiveEncoder(n_layers=3, feature_size=(7, 7, 768), heads=8).cuda()
    feat_cap = encoder(img_A, img_B)
    print(feat_cap.shape)
    print(feat_cap)
    print('Done')
