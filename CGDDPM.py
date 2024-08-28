#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:10:55 2023

@author: Nicholas Sung
@paper: Cooling-Guide Diffusion Model for Battery Cell Arrangement
@arXiv: https://arxiv.org/abs/2403.10566


Code adapted from https://www.kaggle.com/code/grishasizov/simple-denoising-diffusion-model-toy-1d-example
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing as PP
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

# Utility functions
def timestep_embedding(timesteps, dim, max_period=10000, device=torch.device('cuda:0')):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def extract_features(positions, num_circles=20, radius=10.5/125):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = positions.to(device)
    batch_size = positions.size(0)
    positions = positions.view(batch_size, num_circles, 2)
    
    num_features = num_circles * (num_circles - 1) // 2
    features = torch.zeros(batch_size, num_features, device=device)
    
    k = 0
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            xi, yi = positions[:, i, 0], positions[:, i, 1]
            xj, yj = positions[:, j, 0], positions[:, j, 1]
            distance = ((xi - xj) ** 2 + (yi - yj) ** 2).sqrt() - 2 * radius
            features[:, k] = distance
            k += 1
    
    return features


class PositionBasedClassifier(nn.Module):
    def __init__(self, DDPM_Dict, Class_Dict, num_circles=20):
        super().__init__()
        self.DDPM_Dict = DDPM_Dict
        self.xdim = Class_Dict['xdim']
        self.cdim = Class_Dict['cdim']
        self.num_circles = num_circles
        self.output_dim = Class_Dict['output_dim']
        self.radius = 10.5 / 125
        self.num_features = num_circles * (num_circles - 1) // 2
        self.aggregation_layer = nn.Linear(self.num_features, self.output_dim)

    def forward(self, positions):
        features = extract_features(positions)
        negative_distances = F.relu(-features)
        return torch.sigmoid(self.aggregation_layer(negative_distances))

class RegressionResNetModel(nn.Module):
    def __init__(self, Reg_Dict):
        super().__init__()
        self.xdim = Reg_Dict['xdim']
        self.ydim = Reg_Dict['ydim']
        self.tdim = Reg_Dict['tdim']
        self.net = Reg_Dict['net']
        
        self.fc = nn.ModuleList([self._lin_layer(self.tdim, self.net[0])])
        self.fc.extend([self._lin_layer(self.net[i-1], self.net[i]) for i in range(1, len(self.net))])
        self.fc.append(self._lin_layer(self.net[-1], self.tdim))
        self.final_layer = nn.Sequential(nn.Linear(self.tdim, self.ydim))
        self.X_embed = nn.Linear(self.xdim, self.tdim)
    
    def _lin_layer(self, dimi, dimo):
        return nn.Sequential(
            nn.Linear(dimi, dimo),
            nn.SiLU(),
            nn.LayerNorm(dimo),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.X_embed(x)
        res_x = x
        for layer in self.fc:
            x = layer(x)
        x = torch.add(x, res_x)
        return self.final_layer(x)

# Model definitions
class DenoiseResNetModel(nn.Module):
    def __init__(self, DDPM_Dict):
        super().__init__()
        self.xdim = DDPM_Dict['xdim']
        self.ydim = DDPM_Dict['ydim']
        self.tdim = DDPM_Dict['tdim']
        self.cdim = DDPM_Dict['cdim']
        self.net = DDPM_Dict['net']
        
        self.fc = nn.ModuleList([self._lin_layer(self.tdim, self.net[0])])
        self.fc.extend([self._lin_layer(self.net[i-1], self.net[i]) for i in range(1, len(self.net))])
        self.fc.append(self._lin_layer(self.net[-1], self.tdim))
        self.final_layer = nn.Sequential(nn.Linear(self.tdim, self.xdim))
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim)
        )
    
    def _lin_layer(self, dimi, dimo):
        return nn.Sequential(
            nn.Linear(dimi, dimo),
            nn.SiLU(),
            nn.BatchNorm1d(dimo),
            nn.Dropout(p=0.1)
        )

    def forward(self, x, timesteps):
        x = self.X_embed(x) + self.time_embed(timestep_embedding(timesteps, self.tdim))
        res_x = x
        for layer in self.fc:
            x = layer(x)
        x = torch.add(x, res_x)
        return self.final_layer(x)


# Exponential Moving Average Class
class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

# Data normalizer class
class DataNormalizer:
    def __init__(self, X_LL_Scaled, X_UL_Scaled, datalength):
        self.normalizer = PP.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(datalength // 30, 1000), 10),
            subsample=int(1e9)
        )
        self.X_LL_Scaled = X_LL_Scaled
        self.X_UL_Scaled = X_UL_Scaled

    def fit_data(self, X):
        x = 2.0 * (X - self.X_LL_Scaled) / (self.X_UL_Scaled - self.X_LL_Scaled) - 1.0
        return x
    
    def transform_data(self, X):
        return 2.0 * (X - self.X_LL_Scaled) / (self.X_UL_Scaled - self.X_LL_Scaled) - 1.0
    
    def scale_X(self, z):
        return (z + 1.0) * 0.5 * (self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled

# Convert NumPy arrays to lists in dictionaries
def convert_np_arrays_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_np_arrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays_to_lists(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class GuidedDiffusionEnv:
    def __init__(self, DDPM_Dict, Class_Dict, Reg_Dict, X, X_reg, Y_reg, X_pos, Cons, X_neg, Cons_neg):

        
        self.DDPM_Dict = DDPM_Dict
        self.Class_Dict = Class_Dict
        self.Reg_Dict = Reg_Dict
        
        self.device =torch.device(self.DDPM_Dict['device_name'])

        self.diffusion = DenoiseResNetModel(self.DDPM_Dict)
        self.classifier = PositionBasedClassifier(DDPM_Dict=self.DDPM_Dict, Class_Dict=self.Class_Dict, num_circles=20)
        self.regressor = RegressionResNetModel(self.Reg_Dict)
        
        self.diffusion.to(self.device)
        self.classifier.to(self.device)
        self.regressor.to(self.device)

        self.dataLength = self.DDPM_Dict['datalength']
        self.batch_size = self.DDPM_Dict['batch_size']
        self.gamma = self.DDPM_Dict['gamma']
      
        self.data_norm = DataNormalizer(np.array(self.DDPM_Dict['X_LL']),np.array(self.DDPM_Dict['X_UL']),self.dataLength)

        self.X = torch.from_numpy(X.astype('float32')) 
        self.X = self.X.to(self.device)

        self.X_reg = torch.from_numpy(X_reg.astype('float32')) 
        self.X_reg = self.X_reg.to(self.device)

        self.Y_reg = torch.from_numpy(Y_reg.astype('float32'))
        self.Y_reg = self.Y_reg.to(self.device)

        self.X_pos = torch.from_numpy(X_pos.astype('float32')) 
        self.X_pos = self.X_pos.to(self.device)

        self.X_neg = torch.from_numpy(X_neg.astype('float32')) 
        self.X_neg = self.X_neg.to(self.device)

        self.Cons = torch.from_numpy(Cons.astype('float32'))
        self.Cons = self.Cons.to(self.device)

        self.Cons_neg = torch.from_numpy(Cons_neg.astype('float32'))
        self.Cons_neg = self.Cons_neg.to(self.device)
        
        self.eps = 1e-8
        
        self.ema = EMA(0.99)
        self.ema.register(self.diffusion)
        
        
        #set up optimizer 
        self.timesteps = self.DDPM_Dict['Diffusion_Timesteps']
        self.num_diffusion_epochs = self.DDPM_Dict['Training_Epochs']
        
        self.num_classifier_epochs = self.Class_Dict['Training_Epochs']
        self.num_regressor_epochs = self.Reg_Dict['Training_Epochs']
        
        lr = self.DDPM_Dict['lr']
        self.init_lr = lr
        weight_decay = self.DDPM_Dict['weight_decay']
        
        self.optimizer_diffusion = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_classifier = torch.optim.AdamW(self.classifier.parameters(),lr=.001, weight_decay=weight_decay)
        self.optimizer_regressor = torch.optim.AdamW(self.regressor.parameters(), lr=self.Reg_Dict['lr'], weight_decay=self.Reg_Dict['weight_decay'])
        
        self.log_every = 100
        self.print_every = 5000
        self.loss_history = []
               
        self.betas = torch.linspace(0.001, 0.2, self.timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.log_alpha = torch.log(self.alphas)
        self.log_cumprod_alpha = np.cumsum(self.log_alpha.cpu().numpy())
        self.log_cumprod_alpha = torch.tensor(self.log_cumprod_alpha,device=self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],[1,0],'constant', 0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod =  torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        a = torch.clone(self.posterior_variance)
        a[0] = a[1]                 
        self.posterior_log_variance_clipped = torch.log(a)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)* torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))
        
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)

    def _anneal_lr(self, epoch_step):
        frac_done = epoch_step / self.DDPM_Dict['Training_Epochs']
        lr = self.DDPM_Dict['lr'] * (1 - frac_done)
        for param_group in self.optimizer_diffusion.param_groups:
            param_group["lr"] = lr

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_loss(self, x_start, t, noise=None, loss_type='l2'):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.diffusion(x_noisy, t)
        if loss_type == 'l1':
            return F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            return F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            return F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

    def run_classifier_step(self, x, cons):
        self.optimizer_classifier.zero_grad()
        predicted_cons = self.classifier(x)
        loss = F.binary_cross_entropy(predicted_cons, cons)
        loss.backward()
        self.optimizer_classifier.step()
        return loss

    def run_train_classifier_loop(self, batches_per_epoch=100):
        X = torch.cat((self.X_pos, self.X_neg))
        C = torch.cat((self.Cons, self.Cons_neg))
        X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=0.1)
        datalength = X_train.shape[0]

        print('Classifier Model Training...')
        self.classifier.train()
        num_batches = datalength // self.batch_size
        batches_per_epoch = min(num_batches, batches_per_epoch)

        x_batch = torch.full((batches_per_epoch, self.batch_size, self.classifier.xdim), 0, dtype=torch.float32, device=self.device)
        cons_batch = torch.full((batches_per_epoch, self.batch_size, self.classifier.cdim), 0, dtype=torch.float32, device=self.device)

        for i in tqdm(range(self.num_classifier_epochs)):
            for j in range(0, batches_per_epoch):
                A = np.random.randint(0, datalength, self.batch_size)
                x_batch[j] = X_train[A]
                cons_batch[j] = C_train[A]
            for j in range(0, batches_per_epoch):
                self.run_classifier_step(x_batch[j], cons_batch[j])

        self.classifier.eval()
        A = np.random.randint(0, X_test.shape[0], 1000)
        C_pred = self.classifier(X_test[A]).to(torch.device('cpu')).detach().numpy()
        C_pred = np.rint(C_pred)
        C_test = C_test.to(torch.device('cpu')).detach().numpy()
        F1 = f1_score(C_test[A], C_pred)
        print(f'F1 score: {F1}')
        print('Classifier Training Complete!')

    def run_regressor_step(self, x, y):
        self.optimizer_regressor.zero_grad()
        predicted_y = self.regressor(x)
        loss = F.mse_loss(predicted_y.reshape(-1, 1), y.reshape(-1, 1))
        loss.backward()
        self.optimizer_regressor.step()
        return loss

    def run_train_regressor_loop(self, k_folds=5, batches_per_epoch=16):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_train_r2 = []
        all_test_r2 = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(self.X_reg)):
            print(f"Training on fold {fold+1}/{k_folds}...")
            X_reg_train, X_reg_test = self.X_reg[train_idx], self.X_reg[test_idx]
            Y_reg_train, Y_reg_test = self.Y_reg[train_idx], self.Y_reg[test_idx]

            datalength = X_reg_train.shape[0]
            num_batches = datalength // self.batch_size
            batches_per_epoch = min(num_batches, batches_per_epoch)

            self.regressor.train()
            for _ in tqdm(range(self.Reg_Dict['Training_Epochs']), desc=f"Epochs (Fold {fold+1})"):
                for _ in range(batches_per_epoch):
                    batch_indices = np.random.randint(0, datalength, self.batch_size)
                    x_train_batch = X_reg_train[batch_indices]
                    y_train_batch = Y_reg_train[batch_indices]
                    self.run_regressor_step(x_train_batch, y_train_batch)

            self.regressor.eval()
            Y_train_pred = self.regressor(X_reg_train).detach().cpu().numpy()
            train_r2 = r2_score(Y_reg_train.cpu().numpy(), Y_train_pred)
            all_train_r2.append(train_r2)

            Y_test_pred = self.regressor(X_reg_test).detach().cpu().numpy()
            test_r2 = r2_score(Y_reg_test.cpu().numpy(), Y_test_pred)
            all_test_r2.append(test_r2)

            print(f"Fold {fold+1}: Train R2 = {train_r2:.4f}, Test R2 = {test_r2:.4f}")

        avg_train_r2 = np.mean(all_train_r2)
        avg_test_r2 = np.mean(all_test_r2)
        print(f"Average Train R2: {avg_train_r2:.4f}, Average Test R2: {avg_test_r2:.4f}")
        print('Regressor K-Fold Training Complete!')

    def run_diffusion_step(self, x):
        self.optimizer_diffusion.zero_grad()
        t = torch.randint(0, self.DDPM_Dict['Diffusion_Timesteps'], (self.batch_size,), device=self.device)
        loss = self.p_loss(x, t, loss_type='l2')
        loss.backward()
        self.optimizer_diffusion.step()
        return loss

    def run_train_diffusion_loop(self, batches_per_epoch=100):
        print('Denoising Model Training...')
        self.diffusion.train()
        num_batches = self.dataLength // self.batch_size
        batches_per_epoch = min(num_batches, batches_per_epoch)

        x_batch = torch.full((batches_per_epoch, self.batch_size, self.diffusion.xdim), 0, dtype=torch.float32, device=self.device)

        for i in tqdm(range(self.num_diffusion_epochs)):
            for j in range(0, batches_per_epoch):
                A = np.random.randint(0, self.dataLength, self.batch_size)
                x_batch[j] = self.X[A]
            for j in range(0, batches_per_epoch):
                loss = self.run_diffusion_step(x_batch[j])
            self._anneal_lr(i)
            if (i + 1) % 100 == 0:
                self.loss_history.append([i + 1, float(loss.to('cpu').detach().numpy())])
            if (i + 1) % 5000 == 0:
                print(f'Step {(i + 1)}/{self.num_diffusion_epochs} Loss: {loss}')
            self.ema.update(self.diffusion)
        self.loss_history = np.array(self.loss_history)
        print('Denoising Model Training Complete!')

    def cond_fn(self, x, t, cons):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            pred_cons = self.classifier(x_in)
            error = F.binary_cross_entropy(pred_cons, cons)
            grad = torch.autograd.grad(error.sum(), x_in, allow_unused=True)[0]
            return -grad

    def perf_fn(self, x):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            perf = self.regressor(x_in)
            grad = torch.autograd.grad(perf.sum(), x_in)[0]
            return grad

    @torch.no_grad()
    def p_sample(self, x, t, cons):
        time = torch.full((x.size(dim=0),), t, dtype=torch.int64, device=self.device)
        X_diff = self.diffusion(x, time)

        betas_t = self.extract(self.betas, time, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, time, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * X_diff / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
        cons_grad = self.cond_fn(x, time, cons)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x, device=self.device)
            return model_mean + torch.sqrt(posterior_variance_t) * (noise * (1.0 - self.gamma) + self.gamma * cons_grad.float())

    @torch.no_grad()
    def performance_p_sample(self, x, t, cons, perf_weight):
        time = torch.full((x.size(dim=0),), t, dtype=torch.int64, device=self.device)
        X_diff = self.diffusion(x, time)

        betas_t = self.extract(self.betas, time, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, time, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * X_diff / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
        perf_guidance = self.perf_fn(model_mean) * perf_weight
        cons_grad = self.cond_fn(model_mean, time, cons)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x, device=self.device)
            return model_mean + torch.sqrt(posterior_variance_t) * (noise * (1.0 - self.gamma) + self.gamma * cons_grad.float() - perf_guidance)

    @torch.no_grad()
    def gen_samples(self, cons):
        num_samples = len(cons)
        cons = torch.from_numpy(cons.astype('float32')).to(self.device)
        x_gen = torch.randn((num_samples, self.diffusion.xdim), device=self.device)

        self.diffusion.eval()
        self.classifier.eval()

        for i in tqdm(range(self.DDPM_Dict['Diffusion_Timesteps'] - 1, 0, -1)):
            x_gen = self.p_sample(x_gen, i, cons)

        return x_gen.cpu().detach().numpy()

    @torch.no_grad()
    def gen_perf_samples(self, cons, perf_weight):
        num_samples = len(cons)
        perf_time_ratio = 1.0 - 0.8
        cons = torch.from_numpy(cons.astype('float32')).to(self.device)
        x_gen = torch.randn((num_samples, self.diffusion.xdim), device=self.device)

        self.diffusion.eval()
        self.classifier.eval()

        for i in tqdm(range(self.DDPM_Dict['Diffusion_Timesteps'] - 1, int(perf_time_ratio * self.DDPM_Dict['Diffusion_Timesteps']), -1)):
            x_gen = self.performance_p_sample(x_gen, i, cons, perf_weight)

        for i in tqdm(range(int(perf_time_ratio * self.DDPM_Dict['Diffusion_Timesteps']), 0, -1)):
            x_gen = self.p_sample(x_gen, i, cons)

        return x_gen.cpu().detach().numpy()

    def predict_perf_numpy(self, X):
        X = torch.from_numpy(X.astype('float32')).to(self.device)
        Y_pred = self.regressor(X).to(self.device)
        return Y_pred.to('cpu').detach().numpy()

    def load_trained_diffusion_model(self, PATH):
        self.diffusion.load_state_dict(torch.load(PATH))

    def save_diffusion_model(self, PATH, name):
        torch.save(self.diffusion.state_dict(), PATH + name + '.pth')
        ddpm_dict_preprocessed = convert_np_arrays_to_lists(self.DDPM_Dict)
        with open(PATH + name + '.json', 'w') as f:
            json.dump(ddpm_dict_preprocessed, f)

    def load_trained_classifier_model(self, PATH):
        self.classifier.load_state_dict(torch.load(PATH))

    def save_classifier_model(self, PATH, name):
        torch.save(self.classifier.state_dict(), PATH + name + '.pth')
        with open(PATH + name + '.json', 'w') as f:
            json.dump(self.Class_Dict, f)

    def load_trained_regressor_model(self, PATH):
        self.regressor.load_state_dict(torch.load(PATH))

    def save_regressor_model(self, PATH, name):
        torch.save(self.regressor.state_dict(), PATH + name + '.pth')
        with open(PATH + name + '.json', 'w') as f:
            json.dump(self.Reg_Dict, f)
