import torch
import torch.nn as nn
from torch import Tensor
import math
import time
import numpy as np
import IPython

from typing import Type, Any, Callable, Union, List, Optional

from .utils import get_dim_mix_masks

#####################################
# Ratio Critic Models
#####################################

class RatioCritic(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.25):
        # Setting parameters as attributes
        super(RatioCritic, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()

        # Define model
        self.critic = nn.Sequential(
                            nn.Linear(dim_input, 500),
                            self.activation,
                            self.dropout,
                            nn.Linear(500, 100),
                            self.activation,
                            self.dropout, 
                            nn.Linear(100, dim_output)
        )
        
    
    def forward(self, x):
        return self.critic(x)
    
class RatioCriticNN1D(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.25):
        # Setting parameters as attributes
        super(RatioCriticNN1D, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()

        # Define model
        self.critic = nn.Sequential(
                            nn.Linear(dim_input, 100),
                            self.activation,
                            self.dropout,
                            nn.Linear(100, 100),
                            self.activation,
                            self.dropout, 
                            nn.Linear(100, dim_output)
        )
        
    
    def forward(self, x):
        return self.critic(x)    
    
class RatioCriticImage(nn.Module):
    def __init__(self, dim_input, dim_output, num_input_channels, dropout=0.25, alphas=None):
        # Setting parameters as attributes
        super(RatioCriticImage, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()
        if alphas is None:
            self.register_buffer('alphas', torch.Tensor([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        else: 
            self.register_buffer('alphas', alphas)
        print('Alphas set to {}'.format(self.alphas))
        
        self.g = resnet14(pretrained=False, use_fc=False, num_input_channels=num_input_channels)
        self.f = resnet14(pretrained=False, use_fc=False, num_input_channels=num_input_channels)
        
        # Define model
        self.critic = nn.Sequential(
                            nn.Linear(self.g.fc_in_features*2, 500),
                            self.activation,
                            self.dropout,
                            nn.Linear(500, 100),
                            self.activation,
                            self.dropout, 
                            nn.Linear(100, dim_output)
        )
        
    
    def forward(self, u, v):
        # Create m_samples using dimension-wise mixing
        # Mixing u and v
#         m_samples = self.get_m_samples(u.clone().detach(), v.clone().detach())
        # Mixing p and q

        m_samples = self.get_m_samples(v.clone().detach())
        
        u = self.g(u)
        v = self.f(v)
        v_m = self.f(m_samples)
        
        v_shuff = v[torch.randperm(v.shape[0])]
        
        # p (joint)
        p = self.critic(torch.cat((u, v), dim=-1))
        
        # q (marginal)
        q = self.critic(torch.cat((u, v_shuff), dim=-1))
        
        # m (linear combination)
        m = self.critic(torch.cat((u, v_m), dim=-1))
#         m = self.critic(torch.cat((u, (1-torch.tile(self.alphas, (math.ceil(u.shape[0] / len(self.alphas)),)).unsqueeze(1)[:u.shape[0]]*v
#                                        + torch.tile(self.alphas, (math.ceil(v.shape[0] / len(self.alphas)),)).unsqueeze(1)[:u.shape[0]]*v_shuff)), dim=-1))

        # m (dimension-wise mixing, where we mix 1 character/image in the grid at a time)
#     get_dim_mix_masks

        
        return p, q, m

    def get_m_samples(self, v):
        v_m = []
        v_shuff = v[torch.randperm(v.shape[0])]
        
        # Create all samples of m with p & q mixing
        for i in range(len(v)):
            masks_idx = np.random.randint(0, len(self.alphas))
            p_mask, q_mask = self.alphas[masks_idx]
            m = v[i]*p_mask + v_shuff[i]*q_mask
            v_m.append(m)
        
        v_m_all = torch.stack(v_m, dim=0)
        return v_m_all

    def test_forward(self, u, v):
        start_t = time.time()
        m_samples = self.get_m_samples(v.clone().detach())
        end_t = time.time()
        print('get m samples time: {}'.format(end_t-start_t))
        
        return m_samples
    
class RatioCriticImageBinary(nn.Module):
    def __init__(self, dim_input, dim_output, num_input_channels, num_classes, dropout=0.25, alphas=None):
        # Setting parameters as attributes
        super(RatioCriticImageBinary, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()
        if alphas is None:
            self.register_buffer('alphas', torch.Tensor([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        else: 
            self.register_buffer('alphas', alphas)
        print('Alphas set to {}'.format(self.alphas))
        
        self.g = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        self.f = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        
        # Define critic parameters
        self.critic = nn.Sequential(
                            nn.Linear(self.g.fc_out_features*2, dim_output)
        )
        

    def forward(self, u, v):
        v_q = v.clone().detach()[torch.randperm(v.shape[0])]
        
        u = self.g(u)
        v = self.f(v)

        # v for q
        v_q = self.f(v_q)        
        
        # p (joint)
        p = self.critic(torch.cat((u, v), dim=-1))
        # q (marginal)
        q = self.critic(torch.cat((u, v_q), dim=-1))
 
        return p, q

class RatioCriticImageBinaryNonLinear(nn.Module):
    def __init__(self, dim_input, dim_output, num_input_channels, num_classes, dropout=0.25, alphas=None):
        # Setting parameters as attributes
        super(RatioCriticImageBinaryNonLinear, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()
        if alphas is None:
            self.register_buffer('alphas', torch.Tensor([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        else: 
            self.register_buffer('alphas', alphas)
        print('Alphas set to {}'.format(self.alphas))
        
        self.g = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        self.f = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        
        # Define critic parameters        
#         self.critic = nn.Sequential(
#                             nn.Linear(self.g.fc_out_features*2, 500),
#                             self.activation,
#                             self.dropout,
#                             nn.Linear(500, 100),
#                             self.activation,
#                             self.dropout, 
#                             nn.Linear(100, dim_output)
#         )
        
        self.critic = nn.Sequential(
                            nn.Linear(self.g.fc_out_features*2, 500),
                            self.activation,
                            self.dropout,
                            nn.Linear(500, 250),
                            self.activation,
                            self.dropout, 
                            nn.Linear(250, 100),
                            self.activation,
                            self.dropout, 
                            nn.Linear(100, dim_output)
        )
        

    def forward(self, u, v):
        u_q = u.clone().detach()
        v_q = v.clone().detach()[torch.randperm(v.shape[0])]
        
        u = self.g(u)
        v = self.f(v)

        # v for q
        u_q = self.g(u_q)
        v_q = self.f(v_q)        
        
        # p (joint)
        p = self.critic(torch.cat((u, v), dim=-1))
        # q (marginal)
        q = self.critic(torch.cat((u_q, v_q), dim=-1))
 
        return p, q
    
class RatioCriticImageBilinear(nn.Module):
    def __init__(self, dim_input, dim_output, num_input_channels, num_classes, dropout=0.25, alphas=None, use_scale_shift=False):
        # Setting parameters as attributes
        super(RatioCriticImageBilinear, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()
        if alphas is None:
            self.register_buffer('alphas', torch.Tensor([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        else: 
            self.register_buffer('alphas', alphas)
        print('Alphas set to {}'.format(self.alphas))
        
        self.g = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes, norm_layer=ScaleShift if use_scale_shift else None)

        self.f = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes, norm_layer=ScaleShift if use_scale_shift else None)
        
        self.c = create_parameter((self.g.fc_out_features, self.g.fc_out_features))
    
    def forward(self, u, v, u_q, v_q):
#         u_q = u.clone().detach()
        # Shuffling v_q
#         v_q = v_q.clone().detach()[torch.randperm(v_q.shape[0])]
        
        # Forward passes through encoder
        u = self.g(u)
        v = self.f(v)
        
        u_q = self.g(u_q) # B x F
        v_q = self.f(v_q) # B x F
        
        gW1 = torch.matmul(u.unsqueeze(-1).permute(0,2,1), self.c) # B x 1 x F 
        p = torch.matmul(gW1, v.unsqueeze(-1)) # B x 1 x 1
        
        gW2 = torch.matmul(u_q.unsqueeze(-1).permute(0,2,1), self.c) # B x 1 x F 
        q = torch.matmul(gW2, v_q.unsqueeze(-1)) # B x 1 x 1
 
        return p.squeeze(-1), q.squeeze(-1)

class RatioCriticImageBilinearSameEnc(nn.Module):
    def __init__(self, dim_input, dim_output, num_input_channels, num_classes, dropout=0.25, alphas=None, use_scale_shift=False):
        # Setting parameters as attributes
        super(RatioCriticImageBilinearSameEnc, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()
        if alphas is None:
            self.register_buffer('alphas', torch.Tensor([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        else: 
            self.register_buffer('alphas', alphas)
        print('Alphas set to {}'.format(self.alphas))
        
        self.g = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes, norm_layer=ScaleShift if use_scale_shift else None)
        self.f = self.g
#         self.f = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes, norm_layer=ScaleShift if use_scale_shift else None)
        
        self.c = create_parameter((self.g.fc_out_features, self.g.fc_out_features))
    
    def forward(self, u, v, u_q, v_q):
#         u_q = u.clone().detach()
        # Shuffling v_q
#         v_q = v_q.clone().detach()[torch.randperm(v_q.shape[0])]
        
        # Forward passes through encoder
        u = self.g(u)
        v = self.f(v)
        
        u_q = self.g(u_q) # B x F
        v_q = self.f(v_q) # B x F
        
        gW1 = torch.matmul(u.unsqueeze(-1).permute(0,2,1), self.c) # B x 1 x F 
        p = torch.matmul(gW1, v.unsqueeze(-1)) # B x 1 x 1
        
        gW2 = torch.matmul(u_q.unsqueeze(-1).permute(0,2,1), self.c) # B x 1 x F 
        q = torch.matmul(gW2, v_q.unsqueeze(-1)) # B x 1 x 1
 
        return p.squeeze(-1), q.squeeze(-1)

class RatioCriticImageBilinearCoB(nn.Module):
    def __init__(self, dim_input, dim_output, num_input_channels, num_classes, dropout=0.25, alphas=None, same_enc=False):
        # Setting parameters as attributes
        super(RatioCriticImageBilinearCoB, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()
        if alphas is None:
            self.register_buffer('alphas', torch.Tensor([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        else: 
            self.register_buffer('alphas', alphas)
        print('Alphas set to {}'.format(self.alphas))
        
        self.g = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        if same_enc:
            self.f = self.g
        else:
            self.f = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        
        # Encoder weight init
        self.g.apply(weights_init)
        self.f.apply(weights_init)
        
        # Define critic parameters
#         self.critic = Critic(dim_input=self.g.fc_out_features*2, dim_output=dim_output)
        self.c = create_parameter((dim_output, 1, self.g.fc_out_features, self.g.fc_out_features))
        
    
    def forward(self, u, v, u_q=None, v_q=None):
        # Create m samples (on v)
        v_m = self.get_m_samples(v.clone().detach(), v_q.clone().detach() if v_q is not None else None)

        # Create q samples (on v) by shuffling, if not provided
        if v_q is None:
            v_q = v.clone().detach()[torch.randperm(v.shape[0])]
        
        # Forward passes through encoder
        u = self.g(u) # B x F
        v = self.f(v)
        v_m = self.f(torch.cat(v_m, dim=0))
        v_q = self.f(v_q)
        
        v_ms = torch.chunk(v_m, self.dim_output-2)
        
        # Discriminator
        gW1 = torch.matmul(u.unsqueeze(-1).permute(0,2,1), self.c) # (B x 1 x F) x (5 x 1 x F x F) => (5 x B x 1 x F)
        p = torch.matmul(gW1, v.unsqueeze(-1)).squeeze().permute(1, 0) # (5 x B x 1 x F) x (B x F x 1) => (5 x B x 1 x 1) => (B x 5)
        
        ms = [torch.matmul(gW1, v_other.unsqueeze(-1)).squeeze().permute(1, 0) for v_other in v_ms] # (5 x B x 1 x F) x (B x F x 1) => (5 x B x 1 x 1) => (B x 5)
        
        q = torch.matmul(gW1, v_q.unsqueeze(-1)).squeeze().permute(1, 0) # (5 x B x 1 x F) x (B x F x 1) => (5 x B x 1 x 1) => (B x 5)
 
        return p, q, ms

    def get_m_samples(self, v_p, v_q=None):
        v_m = []
        # If v_q not provided, then shuffle v
        if v_q is None:
            v_q = v_p[torch.randperm(v_p.shape[0])]
        
        # K = 1
        if (self.dim_output - 2) == 1:
            for i in range(len(v_p)):
                masks_idx = np.random.randint(0, len(self.alphas))
                p_mask, q_mask = self.alphas[masks_idx]
                m = v_p[i]*p_mask + v_q[i]*q_mask
                v_m.append(m)
            
            v_m = torch.stack(v_m, dim=0)
            return [v_m]
        # K > 1
        else: 
            # Create all samples of m with p & q mixing
            for i in range(len(self.alphas)):
                p_mask, q_mask = self.alphas[i]
                m = v_p*p_mask + v_q*q_mask
                v_m.append(m)
        
        return v_m

    def test_forward(self, u_p, v_p, u_q, v_q):
        start_t = time.time()
        m_samples = self.get_m_samples(v_p.clone().detach(), v_q.clone().detach())
        end_t = time.time()
        print('get m samples time: {}'.format(end_t-start_t))
        
        return m_samples    

class RatioCriticImageQuad(nn.Module):
    def __init__(self, dim_input, dim_output, num_input_channels, num_classes, dropout=0.25, alphas=None):
        # Setting parameters as attributes
        super(RatioCriticImageQuad, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Softplus()
        if alphas is None:
            self.register_buffer('alphas', torch.Tensor([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        else: 
            self.register_buffer('alphas', alphas)
        print('Alphas set to {}'.format(self.alphas))
        
        self.g = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        self.f = resnet14(pretrained=False, use_fc=True, num_input_channels=num_input_channels, num_classes=num_classes)
        
        # Encoder weight init
        self.g.apply(weights_init)
        self.f.apply(weights_init)
        
        # Define critic parameters
        self.critic = Critic(dim_input=self.g.fc_out_features*2, dim_output=dim_output)
        
    
    def forward(self, u, v, u_q=None, v_q=None):
        # Create m samples (on v)
        m_samples = self.get_m_samples(v.clone().detach())

        # Create q samples (on v) by shuffling, if not provided
        if v_q is None:
            v_q = v.clone().detach()[torch.randperm(v.shape[0])]
        
        # Forward passes through encoder
        u = self.g(u)
        v = self.f(v)
        v_m = self.f(m_samples)
        v_q = self.f(v_q)
        
        # Discriminator
        p = self.critic(torch.cat((u, v), dim=-1))
        q = self.critic(torch.cat((u, v_q), dim=-1))
        m = self.critic(torch.cat((u, v_m), dim=-1))
 
        return p, q, m

    def get_m_samples(self, v_p, v_q=None):
        v_m = []
        # If v_q not provided, then shuffle v
        if v_q is None:
            v_q = v_p[torch.randperm(v_p.shape[0])]
        
        # Create all samples of m with p & q mixing
        for i in range(len(v_p)):
            masks_idx = np.random.randint(0, len(self.alphas))
            p_mask, q_mask = self.alphas[masks_idx]
            m = v_p[i]*p_mask + v_q[i]*q_mask
            v_m.append(m)
        
        v_m_all = torch.stack(v_m, dim=0)
        return v_m_all

    def test_forward(self, u_p, v_p, u_q, v_q):
        start_t = time.time()
        m_samples = self.get_m_samples(v_p.clone().detach(), v_q.clone().detach())
        end_t = time.time()
        print('get m samples time: {}'.format(end_t-start_t))
        
        return m_samples    

class RatioCritic1D(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.25, tre=False):
        # Setting parameters as attributes
        super(RatioCritic1D, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activation = nn.Softplus()
        self.tre = tre
        
        # Parameters for quadratic head p
        self.q1 = create_parameter((1,1))
        self.q2 = create_parameter((1,1))
        self.s1 = create_parameter((1,1))
        self.t1 = create_parameter((1,1))
        self.b1 = create_parameter((1,1))

        # Parameters for quadratic head q
        self.q3 = create_parameter((1,1))
        self.q4 = create_parameter((1,1))
        self.s2 = create_parameter((1,1))
        self.t2 = create_parameter((1,1))
        self.b2 = create_parameter((1,1))
        
        # Parameters for linear head m
        self.q5 = create_parameter((1,1))
        self.t3 = create_parameter((1,1))
        self.b3 = create_parameter((1,1))
        
        
        # Define model for m
#         self.critic_m = nn.Sequential(
#                             nn.Linear(dim_input*2, 1),
#                             self.activation,
#                             nn.Linear(1, dim_output//3)
#         )
        
    
    def forward(self, x):
        h1 = 1.0*(x-self.q1)*(x-self.q1)*self.s1 + (x-self.q2)*self.t1 + self.b1
        h2 = (x-self.q3)*(x-self.q3)*self.s2 + (x-self.q4)*self.t2 + self.b2
#         h3 = self.critic_m(torch.cat([h1, h2], dim=-1))
        h3 = self.t3*(x-self.q5) + self.b3
    
        if not self.tre:
            logits = torch.cat([h1, h2, h3], dim=-1)
        else:
            logits = h1
        
        return logits
    
class RatioCritic1D_K3(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.25, tre=False):
        # Setting parameters as attributes
        super(RatioCritic1D_K3, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activation = nn.Softplus()
        self.tre = tre
        
        # Parameters for quadratic head p
        self.q1 = create_parameter((1,1))
        self.q2 = create_parameter((1,1))
        self.s1 = create_parameter((1,1))
        self.t1 = create_parameter((1,1))
        self.b1 = create_parameter((1,1))

        # Parameters for quadratic head q
        self.q3 = create_parameter((1,1))
        self.q4 = create_parameter((1,1))
        self.s2 = create_parameter((1,1))
        self.t2 = create_parameter((1,1))
        self.b2 = create_parameter((1,1))
        
        # Parameters for linear head m1
        self.q5 = create_parameter((1,1))
        self.q6 = create_parameter((1,1))
        self.s3 = create_parameter((1,1))
        self.t3 = create_parameter((1,1))
        self.b3 = create_parameter((1,1))
        
        # Parameters for linear head m2
        self.q7 = create_parameter((1,1))
        self.q8 = create_parameter((1,1))
        self.s4 = create_parameter((1,1))
        self.t4 = create_parameter((1,1))
        self.b4 = create_parameter((1,1))
        
        # Parameters for linear head m3
        self.q9 = create_parameter((1,1))
        self.q10 = create_parameter((1,1))
        self.s5 = create_parameter((1,1))
        self.t5 = create_parameter((1,1))
        self.b5 = create_parameter((1,1))
    
    def forward(self, x):
        h1 = 1.0*(x-self.q1)*(x-self.q1)*self.s1 + (x-self.q2)*self.t1 + self.b1
        h2 = (x-self.q3)*(x-self.q3)*self.s2 + (x-self.q4)*self.t2 + self.b2
        h3 = (x-self.q5)*(x-self.q5)*self.s3 + (x-self.q6)*self.t3 + self.b3
        h4 = (x-self.q7)*(x-self.q7)*self.s4 + (x-self.q8)*self.t4 + self.b4
        h5 = (x-self.q9)*(x-self.q9)*self.s5 + (x-self.q10)*self.t5 + self.b5
    
        if not self.tre:
            logits = torch.cat([h1, h2, h3, h4, h5], dim=-1)
        else:
            logits = h1
        
        return logits
    
class RatioCritic1D_overlap(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.25):
        # Setting parameters as attributes
        super(RatioCritic1D_overlap, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activation = nn.Softplus()
        
        # Parameters for quadratic head p
        self.q1 = create_parameter((1,1))
        self.q2 = create_parameter((1,1))
        self.s1 = create_parameter((1,1))
        self.t1 = create_parameter((1,1))
        self.b1 = create_parameter((1,1))

        # Parameters for quadratic head q
        self.q3 = create_parameter((1,1))
        self.q4 = create_parameter((1,1))
        self.s2 = create_parameter((1,1))
        self.t2 = create_parameter((1,1))
        self.b2 = create_parameter((1,1))
        
        # Parameters for linear head m
        self.q5 = create_parameter((1,1))
        self.t3 = create_parameter((1,1))
        self.b3 = create_parameter((1,1))
        
        
        # Define model for m
#         self.critic_m = nn.Sequential(
#                             nn.Linear(dim_input*2, 1),
#                             self.activation,
#                             nn.Linear(1, dim_output//3)
#         )
        
    
    def forward(self, x):
        h1 = 3.6e13*(x-self.q1)*(x-self.q1)*self.s1 + (x-self.q2)*self.t1 + self.b1
        h2 = (x-self.q3)*(x-self.q3)*self.s2 + (x-self.q4)*self.t2 + self.b2
#         h3 = self.critic_m(torch.cat([h1, h2], dim=-1))
        h3 = self.t3*(x-self.q5) + self.b3
    
        logits = torch.cat([h1, h2, h3], dim=-1)
        
        return logits

#####################################
# Critic Architecture
#####################################
class Critic(nn.Module):
    def __init__(self, dim_input, dim_output):
        # Setting parameters as attributes
        super(Critic, self).__init__()
        
        self.layer1 = nn.Linear(dim_input, 1, bias=False)
        self.layer2 = nn.Linear(dim_input, 1, bias=False)
        self.layer3 = nn.Sequential(
                        nn.Linear(dim_input, dim_input),
                        nn.Softplus(),
                        nn.Linear(dim_input, 1),
                        nn.ReLU()
        )
        
        self.W_psd1 = nn.Parameter(torch.empty((dim_input, dim_input)), requires_grad=True)
        self.W_psd2 = nn.Parameter(torch.empty((dim_input, dim_input)), requires_grad=True)
        torch.nn.init.eye_(self.W_psd1)
        torch.nn.init.eye_(self.W_psd2)
        
        self.b1 = nn.Parameter(torch.empty((1)), requires_grad=True)
        self.b2 = nn.Parameter(torch.empty((1)), requires_grad=True)
        torch.nn.init.constant_(self.b1, -120)
        torch.nn.init.constant_(self.b2, -120)
        
    def forward(self, x):
        h = x.unsqueeze(-1)
        
        # Quadratic + linear
        h1 = torch.matmul(h.permute(0,2,1), self.W_psd1)
        h1 = torch.matmul(h1, h)
        h1 = torch.squeeze(-0.5*h1+self.b1, -1) + self.layer1(x)
        
        # Quadratic + linear
        h2 = torch.matmul(h.permute(0,2,1), self.W_psd2)
        h2 = torch.matmul(h2, h)
        h2 = torch.squeeze(-0.5*h2+self.b2, -1) + self.layer2(x)    
        
        h3 = -150. - self.layer3(x)
        
        return torch.cat([h1, h2, h3], dim=-1)
        
#####################################
# ResNet Models
#####################################
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, dilation=dilation, bias=True)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.3)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(x)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity

        return out

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        if self.in_channels != out_channels:
            self.layers = nn.Sequential(
                                conv1x1(in_))
    
class ScaleShift(nn.Module):
    def __init__(self, width):
        super(ScaleShift, self).__init__()
        self.weight = None
        self.bias = None
    
    def init_wb(self, x):
        self.weight = create_parameter(x.shape[1:])
        with torch.no_grad():
            nn.init.normal_(self.weight, 1.0, 0.02)
        
        self.bias = create_parameter(x.shape[1:])
        with torch.no_grad():
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x):
        if self.weight == None and self.bias == None:
            self.init_wb(x)
            
        return torch.mul(x, self.weight) + self.bias


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_fc: bool = False,
        num_input_channels: int = 1
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_fc = use_fc

        self.inplanes = 32    # 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_input_channels, self.inplanes, kernel_size=5, stride=3)
#                                 padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(0.3)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2, padding=1)           # 64
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, # 128
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, # 256
                                       dilate=replace_stride_with_dilation[1], padding=1)
#         self.layer4 = self._make_layer(block, 64, layers[3], stride=1, # 512
#                                        dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if use_fc:
            self.fc = nn.Linear(64 * block.expansion, num_classes) # 512
            self.fc_in_features = self.fc.in_features
            self.fc_out_features = self.fc.out_features
        else:
            self.fc_in_features = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, padding: int = 0) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        avg_pool = nn.AvgPool2d((2,2), padding=padding)
        if dilate:
            self.dilation *= stride
            stride = 1
        if self.inplanes != planes * block.expansion:
#         if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                norm_layer(planes * block.expansion),
                nn.AvgPool2d((2,2), padding=0)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample or avg_pool, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for idx in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
#         x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.use_fc:
            x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet14(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-14 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet14', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

#####################################
# Helper functions
#####################################

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-14 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def create_parameter(param_shape):
    return nn.init.xavier_uniform_(nn.Parameter(torch.zeros(param_shape), requires_grad=True))
    
    


