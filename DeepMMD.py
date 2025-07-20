import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import inspect
from utils import downsample
import cv2
from PIL import Image
import argparse
from torchvision.transforms import ToTensor, ToPILImage
import math
from skimage import color
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torchvision import models, transforms
from PIL import Image
import numpy as np
import networkx as nx
import math
from sklearn.manifold import Isomap
from concurrent.futures import ThreadPoolExecutor 
import contextlib
from cuml.manifold import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]

        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()

def mmd_loss(x, y, kernels=None, bandwidths=None, weights=None):
    if kernels is None:
        kernels = ['gaussian']
    if bandwidths is None:
        bandwidths = [18.0]
    if weights is None:
        weights = [1.0 / len(kernels)] * len(kernels)
    
    total_mmd = 0.0
    
    for kernel, bandwidth, weight in zip(kernels, bandwidths, weights):
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        

        x_sqnorms = torch.diag(xx)
        y_sqnorms = torch.diag(yy)
        sq_dist_xx = -2 * xx + x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0)
        sq_dist_yy = -2 * yy + y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
        sq_dist_xy = -2 * xy + x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
        
        if kernel == 'gaussian':
            k_xx = torch.exp(-sq_dist_xx / (2 * bandwidth ** 2))
            k_yy = torch.exp(-sq_dist_yy / (2 * bandwidth ** 2))
            k_xy = torch.exp(-sq_dist_xy / (2 * bandwidth ** 2))
        elif kernel == 'laplacian':
            k_xx = torch.exp(-torch.sqrt(sq_dist_xx) / bandwidth)
            k_yy = torch.exp(-torch.sqrt(sq_dist_yy) / bandwidth)
            k_xy = torch.exp(-torch.sqrt(sq_dist_xy) / bandwidth)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        
        total_mmd += weight * mmd
    
    l2 = ((x - y) ** 2).mean()
    combined = 0.5 * total_mmd + 0.5* l2  
    return combined

def process_tensor(image_tensor, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if image_tensor.ndim == 3 and image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)

    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)

    if image_tensor.ndim == 4 and image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    return transform(image_tensor)

def extract_patch_features(feat_map, patch_size=(4, 4)):
    B, C, H, W = feat_map.shape
    ph, pw = patch_size
    patches = feat_map.unfold(2, ph, ph).unfold(3, pw, pw)  
    patches = patches.contiguous().view(B, C, -1, ph, pw) 
    patches = patches.mean(dim=[3, 4]) 
    patches = patches.permute(0, 2, 1)  
    return patches.squeeze(0)  


def extract_patch_features_from_image(image_tensor, model, patch_size=(4, 4), device='cpu'):
    image_tensor = process_tensor(image_tensor).to(device)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError(f"[extract_patch_features_from_image] image_tensor shape must be 4D, got {image_tensor.shape}")

    feats = model.forward_once(image_tensor)
    feature_maps = feats[1:]  

    patch_features_list = []
    for i, feat_map in enumerate(feature_maps):
        if feat_map.ndim == 4:
            patch_feat = extract_patch_features(feat_map, patch_size=patch_size) 
        elif feat_map.ndim == 3:
            patch_feat = feat_map.squeeze(0)  
        else:
            raise ValueError(f"[extract_patch_features_from_image] Unsupported feature map shape: {feat_map.shape}")
        patch_features_list.append(patch_feat)

    return patch_features_list



def compute_euclidean_umap_distance(feat_list1, feat_list2, n_components=2):
    distances = []

    for i, (data1, data2) in enumerate(zip(feat_list1[:5], feat_list2[:5])):
        x1 = data1.detach().cpu().numpy()
        x2 = data2.detach().cpu().numpy()

        N = min(x1.shape[0], x2.shape[0])
        x1 = x1[:N]
        x2 = x2[:N]

        if np.isnan(x1).any() or np.isinf(x1).any() or np.isnan(x2).any() or np.isinf(x2).any():
            distances.append(np.nan)
            continue

        scaler = StandardScaler()
        x1 = scaler.fit_transform(x1)
        x2 = scaler.transform(x2)

        n_neighbors = min(15, max(2, N - 1))

        try:
            umap = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.1,
                        metric='euclidean', random_state=42)
            x1_embed = umap.fit_transform(x1)
            x2_embed = umap.transform(x2)

            if np.isnan(x1_embed).any() or np.isnan(x2_embed).any():
                raise ValueError("UMAP embedding contains NaN")



        except Exception as e:
            pca = PCA(n_components=n_components)
            x1_embed = pca.fit_transform(x1)
            x2_embed = pca.transform(x2)


        diff = x1_embed - x2_embed
        if np.isnan(diff).any():
            distances.append(np.nan)
            continue

        dist = np.linalg.norm(diff, axis=1)
        mean_dist = np.mean(dist)
        distances.append(mean_dist)

    return distances


class DeepMMD(torch.nn.Module):
    def __init__(self, channels=3, load_weights=True, decom_net=None):

        assert channels == 3
        super(DeepMMD, self).__init__()
        self.window = 4
        self.kernels = ['gaussian', 'laplacian'] 
        self.bandwidths = [40, 40]  
        self.weights = [0.5, 0.5]    

        self.vit = models.vit_b_16(pretrained=True)
        

        self.vit.heads = nn.Identity()

        self.layers = [0, 3, 6, 9, 12]  

        for param in self.vit.parameters():
            param.requires_grad = False

        self.chns = [3, 768, 768, 768, 768, 768]
        
        self.decom_net = decom_net

    def forward_once(self, x):
        features = []
        x = self.vit._process_input(x)
        features.append(x)
        
        for i, layer in enumerate(self.vit.encoder.layers):
            x = layer(x)
            if i + 1 in self.layers:
                features.append(x)
        
        return features

    def forward(self, x, y, as_loss=True, resize=True, device=None):
        assert x.shape == y.shape
        
        if resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        
        score = 0 
        layer_score = []

        for k in range(len(feats0)):
            f0 = feats0[k]
            f1 = feats1[k]

            if f0.dim() > 2:
                f0 = f0.contiguous().view(f0.size(0), -1)
            if f1.dim() > 2:
                f1 = f1.contiguous().view(f1.size(0), -1)

            tmp = mmd_loss(f0, f1, kernels=self.kernels, bandwidths=self.bandwidths, weights=self.weights)

            layer_score.append(torch.log(tmp + 1))
            score += tmp
        
        score = score / (k + 1)
        mmd = torch.log(torch.tensor(score + 1))
        mmd = torch.pow(10,torch.tensor(mmd))
        

        if self.decom_net is not None:
            R_x, I_x = self.decom_net(x)
            R_y, I_y = self.decom_net(y)
            
            R_feat_list1 = extract_patch_features_from_image(R_x, self, device=device)
            R_feat_list2 = extract_patch_features_from_image(R_y, self, device=device)
            I_feat_list1 = extract_patch_features_from_image(I_x, self, device=device)
            I_feat_list2 = extract_patch_features_from_image(I_y, self, device=device)


            R_umap_distances = compute_euclidean_umap_distance(R_feat_list1, R_feat_list2)
            I_umap_distances = compute_euclidean_umap_distance(I_feat_list1, I_feat_list2)
            dist1=0
            dist2=0
            for i, R_dist in enumerate(R_umap_distances):
                dist1+=R_dist

            for i, I_dist in enumerate(I_umap_distances):
                dist2+=I_dist
            dist1 = dist1 / 5
            dist2 = dist2 / 5

            dist1 = torch.log(torch.tensor(dist1 + 1))
            dist2 = torch.log(torch.tensor(dist2 + 1))

            score2 = 0.1*dist1+ 0.9*dist2  
            total_score = 0.5*mmd + 0.5 * score2 
            print(f"mmd Score: {mmd.item()}")
            print(f"R_dist: {dist1}")
            print(f"I_dist: {dist2}")
            print(f"score2: {score2}")

        else:
            print("[DEBUG] 未传入分解网络 decom_net == None")
            total_score = torch.log(torch.tensor(score + 1))
            
        return total_score

if __name__ == '__main__':
    from PIL import Image
    import argparse
    from utils import prepare_image
    from models import DecomNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/005.png')
    parser.add_argument('--dist', type=str, default='images/005_AFBEM.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)
    
    decom_net = DecomNet().to(device)

    checkpoint_path = "./weights/decom_net.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    decom_net.load_state_dict(state_dict)
    decom_net.eval()

    model = DeepMMD(decom_net=decom_net).to(device)

    score = model(ref, dist, as_loss=False)
    print('score: %.4f' % score.item())




