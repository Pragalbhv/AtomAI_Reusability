import torch, numpy as np
import torch.optim as optim
from   torch.optim import lr_scheduler 
from   torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn

from random import sample
from copy import copy, deepcopy
from Mat2Spec.utils import *
from Mat2Spec.SinkhornDistance import SinkhornDistance
from Mat2Spec.pytorch_stats_loss import torch_wasserstein_loss

device = set_device()
torch.cuda.empty_cache()
kl_loss_fn = torch.nn.KLDivLoss()
sinkhorn = SinkhornDistance(eps=0.1, max_iter=50, reduction='mean').to(device)

class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*16*16, 128)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class Mat2Spec(nn.Module):
    def __init__(self, args, NORMALIZER):
        super(Mat2Spec, self).__init__()
        n_heads = args.num_heads
        number_neurons = args.num_neurons
        number_layers = args.num_layers
        concat_comp = args.concat_comp
        self.cnn = CNN(in_channels=1, out_channels=64)

        self.loss_type = args.Mat2Spec_loss_type
        self.NORMALIZER = NORMALIZER
        self.input_dim = args.Mat2Spec_input_dim
        self.latent_dim = args.Mat2Spec_latent_dim
        self.emb_size = args.Mat2Spec_emb_size
        self.label_dim = args.Mat2Spec_label_dim
        self.scale_coeff = args.Mat2Spec_scale_coeff
        self.keep_prob = args.Mat2Spec_keep_prob
        self.K = args.Mat2Spec_K
        self.args = args

        self.fx1 = nn.Linear(self.input_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, self.latent_dim*self.K)
        self.fx_logvar = nn.Linear(256, self.latent_dim*self.K)
        self.fx_mix_coeff = nn.Linear(256, self.K)

        self.fe_mix_coeff = nn.Sequential(
            nn.Linear(self.label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.label_dim)
        )

        self.fd_x1 = nn.Linear(self.input_dim + self.latent_dim, 512)
        self.fd_x2 = torch.nn.Sequential(
            nn.Linear(512, self.emb_size)
        )
        self.feat_mp_mu = nn.Linear(self.emb_size, self.label_dim)

        # label layers
        self.fe0 = nn.Linear(self.label_dim, self.emb_size)
        self.fe1 = nn.Linear(self.label_dim, 512)
        self.fe2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, self.latent_dim)
        self.fe_logvar = nn.Linear(256, self.latent_dim)

        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2
        self.label_mp_mu = self.feat_mp_mu

        self.bias = nn.Parameter(torch.zeros(self.label_dim))

        assert id(self.fd_x1) == id(self.fd1)
        assert id(self.fd_x2) == id(self.fd2)

        self.dropout = nn.Dropout(p=self.keep_prob)
        self.emb_proj = nn.Linear(args.Mat2Spec_emb_size, 1024)
        self.W = nn.Linear(args.Mat2Spec_label_dim, args.Mat2Spec_emb_size) # linear transformation for label

    def label_encode(self, x):
        h1 = self.dropout(F.relu(self.fe1(x)))  # [label_dim, 512]
        h2 = self.dropout(F.relu(self.fe2(h1)))  # [label_dim, 256]
        mu = self.fe_mu(h2) * self.scale_coeff  # [label_dim, latent_dim]
        logvar = self.fe_logvar(h2) * self.scale_coeff  # [label_dim, latent_dim]

        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff  # [bs, latent_dim]
        logvar = self.fx_logvar(h3) * self.scale_coeff
        mix_coeff = self.fx_mix_coeff(h3)  # [bs, K]

        if self.K > 1:
            mu = mu.view(x.shape[0], self.K, self.args.Mat2Spec_latent_dim) # [bs, K, latent_dim]
            logvar = logvar.view(x.shape[0], self.K, self.args.Mat2Spec_latent_dim) # [bs, K, latent_dim]

        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar,
            'fx_mix_coeff': mix_coeff
        }
        return fx_output

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def feat_reparameterize(self, mu, logvar, coeff=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def label_decode(self, z):
        d1 = F.relu(self.fd1(z))
        d2 = F.leaky_relu(self.fd2(d1))
        return d2

    def feat_decode(self, z):
        d1 = F.relu(self.fd_x1(z))
        d2 = F.leaky_relu(self.fd_x2(d1))
        return d2

    def label_forward(self, x, feat):  # x is label
        n_label = x.shape[1]  # label_dim
        all_labels = torch.eye(n_label).to(x.device)  # [label_dim, label_dim]
        fe_output = self.label_encode(all_labels)  # map each label to a Gaussian mixture.
        mu = fe_output['fe_mu']
        logvar = fe_output['fe_logvar']
        fe_output['fe_mix_coeff'] = self.fe_mix_coeff(x)
        mix_coeff = F.softmax(fe_output['fe_mix_coeff'], dim=-1)

        if self.args.train:
            z = self.label_reparameterize(mu, logvar) # [label_dim, latent_dim]
        else:
            z = mu
        z = torch.matmul(mix_coeff, z)

        label_emb = self.label_decode(torch.cat((feat, z), 1))
        fe_output['label_emb'] = label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']  # [bs, latent_dim]
        logvar = fx_output['fx_logvar']  # [bs, latent_dim]
        # print("Mu value : ", mu.shape)
        if self.args.train:
            z = self.feat_reparameterize(mu, logvar)
        else:
            z = mu
        if self.K > 1:
            mix_coeff = fx_output['fx_mix_coeff']  # [bs, K]
            mix_coeff = F.softmax(mix_coeff, dim=-1)
            mix_coeff = mix_coeff.unsqueeze(-1).expand_as(z)
            z = z * mix_coeff
            z = torch.sum(z, dim=1)  # [bs, latent_dim]

        feat_emb = self.feat_decode(torch.cat((x, z), 1))  # [bs, emb_size]
        fx_output['feat_emb'] = feat_emb
        return fx_output

    def forward(self, data, label):
        label = label
        feature = self.cnn(data)

        fe_output = self.label_forward(label.float(), feature)
        label_emb = fe_output['label_emb'] # [bs, emb_size]
        fx_output = self.feat_forward(feature)
        feat_emb  = fx_output['feat_emb'] # [bs, emb_size]
        W = self.W.weight # [emb_size, label_dim]
        label_out = torch.matmul(label_emb, W)  # [bs, emb_size] * [emb_size, label_dim] = [bs, label_dim]
        feat_out = torch.matmul(feat_emb, W)  # [bs, label_dim]

        label_proj = self.emb_proj(label_emb)
        feat_proj = self.emb_proj(feat_emb)
        fe_output.update(fx_output)
        output = fe_output

        if self.args.label_scaling == 'normalized_max':
            label_out = F.relu(label_out)
            feat_out = F.relu(feat_out)
            maxima, _ = torch.max(label_out, dim=1)
            label_out = label_out.div(maxima.unsqueeze(1)+1e-8)
            maxima, _ = torch.max(feat_out, dim=1)
            feat_out = feat_out.div(maxima.unsqueeze(1)+1e-8)

        output['label_out'] = label_out
        output['feat_out'] = feat_out
        output['label_proj'] = label_proj
        output['feat_proj'] = feat_proj
        return output

def kl(fx_mu, fe_mu, fx_logvar, fe_logvar):
    kl_loss = 0.5 * torch.sum(
        (fx_logvar - fe_logvar) - 1 + torch.exp(fe_logvar - fx_logvar) + (fx_mu - fe_mu)**2 / (
                torch.exp(fx_logvar) + 1e-8), dim=-1)
    return kl_loss

def compute_c_loss(BX, BY, tau=1):
    BX = F.normalize(BX, dim=1)
    BY = F.normalize(BY, dim=1)
    b = torch.matmul(BX, torch.transpose(BY, 0, 1)) # [bs, bs]
    b = torch.exp(b/tau)
    b_diag = torch.diagonal(b, 0).unsqueeze(1) # [bs, 1]
    b_sum = torch.sum(b, dim=-1, keepdim=True) # [bs, 1]
    c = b_diag/(b_sum-b_diag)
    c_loss = -torch.mean(torch.log(c))
    return c_loss

def compute_loss(input_label, output, NORMALIZER, args):
    fe_out, fe_mu, fe_logvar, label_emb, label_proj = output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb'], output['label_proj']
    fx_out, fx_mu, fx_logvar, feat_emb, feat_proj = output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb'], output['feat_proj']

    fx_mix_coeff = output['fx_mix_coeff']  # [bs, K]
    fe_mix_coeff = output['fe_mix_coeff']
    fx_mix_coeff = F.softmax(fx_mix_coeff, dim=-1)
    fe_mix_coeff = F.softmax(fe_mix_coeff, dim=-1)
    fe_mix_coeff = fe_mix_coeff.repeat(1, args.Mat2Spec_K)
    fx_mix_coeff = fx_mix_coeff.repeat(1, args.Mat2Spec_label_dim)
    mix_coeff = fe_mix_coeff * fx_mix_coeff
    fx_mu = fx_mu.repeat(1, args.Mat2Spec_label_dim, 1)
    fx_logvar = fx_logvar.repeat(1, args.Mat2Spec_label_dim, 1)
    fe_mu = fe_mu.squeeze(0).expand(fx_mu.shape[0], fe_mu.shape[0], fe_mu.shape[1])
    fe_logvar = fe_logvar.squeeze(0).expand(fx_mu.shape[0], fe_logvar.shape[0], fe_logvar.shape[1])
    fe_mu = fe_mu.repeat(1, args.Mat2Spec_K, 1)
    fe_logvar = fe_logvar.repeat(1, args.Mat2Spec_K, 1)
    kl_all = kl(fx_mu, fe_mu, fx_logvar, fe_logvar)
    kl_all_inv = kl(fe_mu, fx_mu, fe_logvar, fx_logvar)
    kl_loss = torch.mean(torch.sum(mix_coeff * (0.5*kl_all + 0.5*kl_all_inv), dim=-1))
    c_loss = compute_c_loss(label_proj, feat_proj)

    if args.label_scaling == 'normalized_sum':
        assert args.Mat2Spec_loss_type == 'KL' or args.Mat2Spec_loss_type == 'WD'
        input_label_normalize = input_label / (torch.sum(input_label, dim=1, keepdim=True)+1e-8)
        pred_e = F.softmax(fe_out, dim=1)
        pred_x = F.softmax(fx_out, dim=1)
        P = input_label_normalize
        Q_e = pred_e
        Q_x = pred_x
        c1, c2, c3 = 1, 1.1, 0.1
        if args.ablation_LE:
            c2 = 0.0
        if args.ablation_CL:
            c3 = 0.0

        if args.Mat2Spec_loss_type == 'KL':
            nll_loss = torch.mean(torch.sum(P*(torch.log(P+1e-8)-torch.log(Q_e+1e-8)),dim=1)) \
            nll_loss_x = torch.mean(torch.sum(P*(torch.log(P+1e-8)-torch.log(Q_x+1e-8)),dim=1)) \
        elif args.Mat2Spec_loss_type == 'WD':
            nll_loss = torch_wasserstein_loss(Q_e, P)
            nll_loss_x = torch_wasserstein_loss(Q_x, P)
        total_loss = (nll_loss + nll_loss_x) * c1 + kl_loss * c2 + c_loss * c3

        return total_loss, nll_loss, nll_loss_x, kl_loss, c_loss, pred_e, pred_x

    else: # standardized or normalized_max
        assert args.Mat2Spec_loss_type == 'MAE' or args.Mat2Spec_loss_type == 'MSE'
        pred_e = fe_out
        pred_x = fx_out
        c1, c2, c3 = 1, 1.1, 0.1
        if args.ablation_LE:
            c2 = 0.0
        if args.ablation_CL:
            c3 = 0.0

        if args.Mat2Spec_loss_type == 'MAE':
            nll_loss = torch.mean(torch.abs(pred_e-input_label))
            nll_loss_x = torch.mean(torch.abs(pred_x-input_label))
        elif args.Mat2Spec_loss_type == 'MSE':
            nll_loss = torch.mean((pred_e-input_label)**2)
            nll_loss_x = torch.mean((pred_x-input_label)**2)
        total_loss = (nll_loss + nll_loss_x) * c1 + kl_loss * c2 + c_loss * c3

        return total_loss, nll_loss, nll_loss_x, kl_loss, c_loss, pred_e, pred_x
        
