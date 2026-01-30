import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import*

class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, num_levels=4, blocks_per_stage=None):
        super().__init__()
        assert num_levels >= 2
        self.num_levels = num_levels
        
        if blocks_per_stage is None:
            blocks_per_stage = [2] * (num_levels - 1)
        assert len(blocks_per_stage) == (num_levels - 1)

        self.chs = [base_channels * (2 ** i) for i in range(num_levels)]

        self.initial = ConvLayer(in_channels, self.chs[0], kernel_size=3, stride=1, activation=True)

        self.stages = nn.ModuleList()
        in_ch = self.chs[0]
        for i in range(num_levels - 1):
            out_ch = self.chs[i+1]
            stage = nn.Sequential(
                ConvLayer(in_ch, out_ch, kernel_size=3, stride=2, activation=True),
                *[ResidualBlock(out_ch) for _ in range(blocks_per_stage[i])]
            )
            self.stages.append(stage)
            in_ch = out_ch

    def forward(self, x):
        features = []
        out = self.initial(x)
        features.append(out)

        for stage in self.stages:
            out = stage(out)
            features.append(out)

        return features

class ContrastAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(channels)

    def forward(self, fA, fB):
        fA_enhanced = self.ca(fA)
        fB_enhanced = self.ca(fB)

        attn_map_A = self.sa(fA_enhanced)
        attn_map_B = self.sa(fB_enhanced)

        fA_out = fA * (1 - attn_map_B + attn_map_A)
        fB_out = fB * (1 - attn_map_A + attn_map_A)

        return fA_out, fB_out

class DecisionNetwork(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.num_levels = len(encoder_channels)
        chs = encoder_channels
        chs_rev = chs[::-1]

        self.contrast_modules = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()


        for i in range(self.num_levels - 1):
            deep_ch = chs_rev[i]
            shallow_ch = chs_rev[i+1]

            self.contrast_modules.append(ContrastAttentionModule(deep_ch))

            self.fusion_convs.append(
                ConvLayer(in_channels=shallow_ch + deep_ch, out_channels=shallow_ch)
            )

        self.output_conv = nn.Sequential(
            ConvLayer(chs[0] * 2, chs[0]),
            ConvLayer(chs[0], chs[0] // 2),
            nn.Conv2d(chs[0] // 2, 2, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features_A, features_B):
        features_A_rev = features_A[::-1]
        features_B_rev = features_B[::-1]

        out_A = features_A_rev[0]
        out_B = features_B_rev[0]

        for i in range(self.num_levels - 1):
            out_A, out_B = self.contrast_modules[i](out_A, out_B)

            skip_A = features_A_rev[i+1]
            skip_B = features_B_rev[i+1]

            target_size = skip_A.shape[2:]

            up_A = F.interpolate(out_A, size=target_size, mode='bilinear', align_corners=True)
            up_B = F.interpolate(out_B, size=target_size, mode='bilinear', align_corners=True)

            out_A = self.fusion_convs[i](torch.cat([skip_A, up_A], dim=1))
            out_B = self.fusion_convs[i](torch.cat([skip_B, up_B], dim=1))

        final_features = torch.cat([out_A, out_B], dim=1)
        final_out = self.output_conv(final_features)
        masks = self.sigmoid(final_out)
        
        mask_A, mask_B = masks.chunk(2, dim=1)

        return mask_A, mask_B
    
class ComplementaryAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.self_attn_A = SelfAttention(channels)
        self.self_attn_B = SelfAttention(channels)
        self.cross_attn = CrossAttention(channels)

    def forward(self, feat_A, feat_B):
        enhanced_A = self.self_attn_A(feat_A)
        enhanced_B = self.self_attn_B(feat_B)
        
        fused_A, fused_B = self.cross_attn(enhanced_A, enhanced_B)
        
        return fused_A, fused_B

class ReconstructionNet(nn.Module):
    def __init__(self, encoder_channels, blocks_per_stage=None):
        super().__init__()
        chs = encoder_channels
        chs_rev = chs[::-1]
        self.num_levels = len(chs)
        
        if blocks_per_stage is None:
            blocks_per_stage = [1] * (self.num_levels - 1)

        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        self.bottleneck_attn = ComplementaryAttentionModule(chs_rev[0])

        self.cross_attn_modules = nn.ModuleList()

        self.bottleneck_conv = ConvLayer(chs_rev[0] * 2, chs_rev[0])

        for i in range(self.num_levels - 1):
            deep_ch, shallow_ch = chs_rev[i], chs_rev[i+1]
            
            self.up_convs.append(
                ConvLayer(deep_ch, deep_ch // 2, kernel_size=1) 
            )

            self.cross_attn_modules.append(CrossAttention(shallow_ch))

            self.conv_blocks.append(
                nn.Sequential(
                    ConvLayer(deep_ch // 2 + shallow_ch * 2, shallow_ch),
                    *[ResidualBlock(shallow_ch) for _ in range(blocks_per_stage[i])]
                )
            )
        
        self.output_conv = nn.Sequential(
            ConvLayer(chs[0], chs[0] // 2),
            ConvLayer(chs[0] // 2, 3, activation=False)
        )

    def forward(self, features_A, features_B):
        features_A_rev, features_B_rev = features_A[::-1], features_B[::-1]

        skip_A, skip_B = features_A_rev[0], features_B_rev[0]
        fused_A, fused_B = self.bottleneck_attn(skip_A, skip_B)
        x = self.bottleneck_conv(torch.cat([fused_A, fused_B], dim=1))

        for i in range(self.num_levels - 1):
            skip_A, skip_B = features_A_rev[i+1], features_B_rev[i+1]
            target_size = skip_A.shape[2:]

            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = self.up_convs[i](x)
            
            fused_A, fused_B = self.cross_attn_modules[i](skip_A, skip_B)
            
            x = torch.cat([x, fused_A, fused_B], dim=1)
            x = self.conv_blocks[i](x)

        imageC = self.output_conv(x)
        return imageC
    
class CSNet(nn.Module): 
    def __init__(self, in_channels=3, base_channels=16, num_levels=4):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, num_levels)
        self.decision_network = DecisionNetwork(self.encoder.chs)
        self.reconstruction_network = ReconstructionNet(self.encoder.chs)

    def forward(self, img_A, img_B):
        featsA = self.encoder(img_A)
        featsB = self.encoder(img_B)
        mask_A, mask_B = self.decision_network(featsA, featsB)
        imageC = self.reconstruction_network(featsA, featsB)
        return mask_A, mask_B, imageC

    
    