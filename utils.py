import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from typing import Tuple
import torch.nn.functional as F 
from numpy.linalg import eig
from scipy.signal import convolve2d, windows
from skimage.morphology import remove_small_holes
import math

def normalize1(img):
    img = np.asarray(img, dtype=np.float64)
    min_val, max_val = np.min(img), np.max(img)
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
    normalized_img = (img - min_val) / (max_val - min_val)
    return (normalized_img * 255).astype(np.uint8)

def mutual_info(im1, im2):
    hist_2d, _, _ = np.histogram2d(im1.ravel(), im2.ravel(), bins=256, range=[[0, 255], [0, 255]])
    pmf_2d = hist_2d / np.sum(hist_2d)
    pmf_2d_nz = pmf_2d[pmf_2d > 0]
    H_xy = -np.sum(pmf_2d_nz * np.log2(pmf_2d_nz))
    pmf_x = np.sum(pmf_2d, axis=1)
    pmf_y = np.sum(pmf_2d, axis=0)
    pmf_x_nz = pmf_x[pmf_x > 0]
    pmf_y_nz = pmf_y[pmf_y > 0]
    H_x = -np.sum(pmf_x_nz * np.log2(pmf_x_nz))
    H_y = -np.sum(pmf_y_nz * np.log2(pmf_y_nz))
    I_xy = H_x + H_y - H_xy
    return I_xy, H_xy, H_x, H_y

def metricNMI(im1, im2, fim):
    im1_n, im2_n, fim_n = normalize1(im1), normalize1(im2), normalize1(fim)
    I_fx, _, H_x, H_f1 = mutual_info(im1_n, fim_n)
    I_fy, _, H_y, H_f2 = mutual_info(im2_n, fim_n)
    term1 = I_fx / (H_f1 + H_x) if (H_f1 + H_x) != 0 else 0
    term2 = I_fy / (H_f2 + H_y) if (H_f2 + H_y) != 0 else 0
    return 2 * (term1 + term2)

def _ssim_for_yang(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    M, N = img1.shape
    if M < 11 or N < 11:
        return -np.inf, -np.inf, None, None
    win_1d = windows.gaussian(7, 1.5)
    window = np.outer(win_1d, win_1d)
    window /= np.sum(window)
    C1, C2 = 2e-16, 2e-16
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = convolve2d(img1**2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2**2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    ssim_map = np.clip(ssim_map, 0.0, 1.0)
    mssim = np.mean(ssim_map)
    return mssim, ssim_map, sigma1_sq, sigma2_sq

def metricYang(im1: np.ndarray, im2: np.ndarray, fim: np.ndarray) -> float:
    im1, im2, fim = im1.astype(np.float64), im2.astype(np.float64), fim.astype(np.float64)
    _, ssim_map1, sigma1_sq1, sigma2_sq1 = _ssim_for_yang(im1, im2)
    _, ssim_map2, _, _ = _ssim_for_yang(im1, fim)
    _, ssim_map3, _, _ = _ssim_for_yang(im2, fim)
    bin_map = ssim_map1 >= 0.75
    buffer = sigma1_sq1 + sigma2_sq1
    is_zero = (buffer == 0)
    sigma1_sq1[is_zero] += 0.5
    sigma2_sq1[is_zero] += 0.5
    buffer_no_nan = sigma1_sq1 + sigma2_sq1
    ramda = sigma1_sq1 / buffer_no_nan
    Q1 = (ramda * ssim_map2 + (1 - ramda) * ssim_map3) * bin_map
    Q2 = np.maximum(ssim_map2, ssim_map3) * (~bin_map)
    Q = np.mean(Q1 + Q2)
    return Q

def _calculate_gradient_map(img, flt1, flt2):
    imgX = cv2.filter2D(img, -1, flt1, borderType=cv2.BORDER_CONSTANT)
    imgY = cv2.filter2D(img, -1, flt2, borderType=cv2.BORDER_CONSTANT)
    G = np.sqrt(imgX**2 + imgY**2)
    buffer = (imgX == 0) * 1e-5
    imgX_plus_buffer = imgX + buffer
    A = np.arctan(imgY / imgX_plus_buffer)
    return G, A

def metricXydeas(img1, img2, fuse):
    img1, img2, fuse = img1.astype(np.float64), img2.astype(np.float64), fuse.astype(np.float64)
    flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    fuseG, fuseA = _calculate_gradient_map(fuse, flt1, flt2)
    img1G, img1A = _calculate_gradient_map(img1, flt1, flt2)
    img2G, img2A = _calculate_gradient_map(img2, flt1, flt2)
    bimap_1 = img1G > fuseG
    img1G_plus_buffer = img1G + (img1G == 0) * 1e-5
    fuseG_plus_buffer = fuseG + (fuseG == 0) * 1e-5
    buffer1_1 = fuseG / img1G_plus_buffer
    buffer2_1 = img1G / fuseG_plus_buffer
    Gaf = np.where(bimap_1, buffer1_1, buffer2_1)
    Aaf = 1 - np.abs(img1A - fuseA) / (np.pi / 2)
    bimap_2 = img2G > fuseG
    img2G_plus_buffer = img2G + (img2G == 0) * 1e-5
    buffer1_2 = fuseG / img2G_plus_buffer
    buffer2_2 = img2G / fuseG_plus_buffer
    Gbf = np.where(bimap_2, buffer1_2, buffer2_2)
    Abf = 1 - np.abs(img2A - fuseA) / (np.pi / 2)
    gama1, gama2, k1, k2, delta1, delta2 = 0.9994, 0.9879, -15, -22, 0.5, 0.8
    Qg_AF = gama1 / (1 + np.exp(k1 * (Gaf - delta1)))
    Qalpha_AF = gama2 / (1 + np.exp(k2 * (Aaf - delta2)))
    Qaf = Qg_AF * Qalpha_AF
    Qg_BF = gama1 / (1 + np.exp(k1 * (Gbf - delta1)))
    Qalpha_BF = gama2 / (1 + np.exp(k2 * (Abf - delta2)))
    Qbf = Qg_BF * Qalpha_BF
    L = 1.0
    Wa, Wb = img1G ** L, img2G ** L
    numerator, denominator = np.sum(Qaf * Wa + Qbf * Wb), np.sum(Wa + Wb)
    return numerator / denominator if denominator != 0 else 0

def _calculate_ncc_wang(im1, im2):
    im1, im2 = im1.astype(np.uint8), im2.astype(np.uint8)
    b = 256.0
    h, _, _ = np.histogram2d(im1.ravel(), im2.ravel(), bins=int(b), range=[[0, 255], [0, 255]])
    h = h / np.sum(h)
    im1_marg, im2_marg = np.sum(h, axis=1), np.sum(h, axis=0)
    h_nz, im1_marg_nz, im2_marg_nz = h[h > 0], im1_marg[im1_marg > 0], im2_marg[im2_marg > 0]
    H_x = -np.sum(im1_marg_nz * np.log2(im1_marg_nz))
    H_y = -np.sum(im2_marg_nz * np.log2(im2_marg_nz))
    H_xy = -np.sum(h_nz * np.log2(h_nz))
    H_x, H_y, H_xy = H_x / np.log2(b), H_y / np.log2(b), H_xy / np.log2(b)
    return H_x + H_y - H_xy

def metricWang(im1, im2, fim):
    im1_norm, im2_norm, fim_norm = normalize1(im1), normalize1(im2), normalize1(fim)
    NCCxy = _calculate_ncc_wang(im1_norm, im2_norm)
    NCCxf = _calculate_ncc_wang(im1_norm, fim_norm)
    NCCyf = _calculate_ncc_wang(im2_norm, fim_norm)
    R = np.array([[1, NCCxy, NCCxf], [NCCxy, 1, NCCyf], [NCCxf, NCCyf, 1]])
    r = np.maximum(eig(R)[0], 1e-12)
    HR = -np.sum(r * np.log2(r / 3.0) / 3.0) / np.log2(256.0)
    return 1 - HR

def metricMI_standard(A, B, F):
    A_n, B_n, F_n = normalize1(A), normalize1(B), normalize1(F)
    return mutual_info(A_n, F_n)[0] + mutual_info(B_n, F_n)[0]

def remove_holes_batch(masks, area_threshold_ratio=0.05):
    batch_size, _, h, w = masks.shape
    device = masks.device
    
    area_threshold = int(h * w * area_threshold_ratio)
    
    output_masks = []
    for i in range(batch_size):
        mask_i_np = masks[i, 0].cpu().numpy()
        mask_i_filled = remove_small_holes(mask_i_np.astype(bool), area_threshold=area_threshold)
        output_masks.append(torch.from_numpy(mask_i_filled.astype(np.float32)))
    
    return torch.stack(output_masks, dim=0).unsqueeze(1).to(device)

class LPIPSFeatureExtractor(nn.Module):
    def __init__(self):
        super(LPIPSFeatureExtractor, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        x = self.normalize_input(x)
        
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        
        outputs = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        
        normalized_outputs = []
        for feat in outputs:
            norm_feat = feat / (torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-10)
            normalized_outputs.append(norm_feat)
            
        return normalized_outputs

def similarity_selection(pred_maskA, pred_maskB, pred_imageC, imgA, imgB, mask_threhold =0.7, similarity_threshold=1.0,
                         metric='perceptual', feature_extractor=LPIPSFeatureExtractor().cuda()):

    binary_maskA = (pred_maskA > mask_threhold).float()
    binary_maskB = (pred_maskB > mask_threhold).float()
    binary_maskA = remove_holes_batch(binary_maskA)
    binary_maskB = remove_holes_batch(binary_maskB)

    overlap_map = (binary_maskA == 1) & (binary_maskB == 1)
    
    final_maskA = binary_maskA.clone(); final_maskA[overlap_map] = 0
    final_maskB = binary_maskB.clone(); final_maskB[overlap_map] = 0
    final_maskC = 1.0 - final_maskA - final_maskB

    fused_image_clear_parts = final_maskA * imgA + final_maskB * imgB

    patch_size = 7
    padding = patch_size // 2
    
    def calculate_distance(source, target, method):
        if method == 'perceptual':
            diff = torch.sum(torch.abs(source - target), dim=1, keepdim=True)
            dist = F.avg_pool2d(diff, kernel_size=patch_size, stride=1, padding=padding)
            return dist
        
        elif method == 'l2':
            diff_sq = torch.sum((source - target) ** 2, dim=1, keepdim=True)
            mse_map = F.avg_pool2d(diff_sq, kernel_size=patch_size, stride=1, padding=padding)
            dist = torch.sqrt(mse_map + 1e-12)
            return dist
            
        elif method == 'cosine':
            cos_sim = F.cosine_similarity(source, target, dim=1).unsqueeze(1)
            dist_pixel = 1.0 - cos_sim
            dist = F.avg_pool2d(dist_pixel, kernel_size=patch_size, stride=1, padding=padding)
            return dist
            
        elif method == 'ssim':
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu1 = F.avg_pool2d(source, patch_size, stride=1, padding=padding)
            mu2 = F.avg_pool2d(target, patch_size, stride=1, padding=padding)
            mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(source * source, patch_size, stride=1, padding=padding) - mu1_sq
            sigma2_sq = F.avg_pool2d(target * target, patch_size, stride=1, padding=padding) - mu2_sq
            sigma12 = F.avg_pool2d(source * target, patch_size, stride=1, padding=padding) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_avg = torch.mean(ssim_map, dim=1, keepdim=True)
            
            return 1.0 - ssim_avg
            
        elif method == 'l1':
            if feature_extractor is None:
                raise ValueError("Input feature_extractor!")
            
            with torch.no_grad():
                feats_src = feature_extractor(source) 
                feats_tgt = feature_extractor(target)
            
            total_dist = 0
            
            for f_src, f_tgt in zip(feats_src, feats_tgt):
                layer_dist = torch.sum((f_src - f_tgt) ** 2, dim=1, keepdim=True)
                
                if layer_dist.shape[2:] != source.shape[2:]:
                    layer_dist = F.interpolate(
                        layer_dist, 
                        size=source.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )

                total_dist += layer_dist

            dist = F.avg_pool2d(total_dist, kernel_size=patch_size, stride=1, padding=padding)
            return dist
            
        else:
            raise ValueError(f"Unsupported metric: {method}")

    dist_to_A = calculate_distance(pred_imageC, imgA, metric)
    dist_to_B = calculate_distance(pred_imageC, imgB, metric)
    similarity_to_A = 1.0 / (dist_to_A + 1e-8)
    similarity_to_B = 1.0 / (dist_to_B + 1e-8)

    decision_map_CA = (similarity_to_A > similarity_threshold *similarity_to_B).float()
    decision_map_CB = (similarity_to_B >= similarity_threshold * similarity_to_A).float() 
    decision_map_C = 1.0 - decision_map_CA - decision_map_CB

    fused_image_C_part = decision_map_CA * imgA + decision_map_CB * imgB + decision_map_C * pred_imageC
    imageF = fused_image_clear_parts + final_maskC * fused_image_C_part
    
    return imageF