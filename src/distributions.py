import numpy as np
import torch
import torch.distributions as D
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.cauchy import Cauchy
from torch.distributions.mixture_same_family import MixtureSameFamily
from scipy.linalg import block_diag

def get_dists_from_mi(mi, n_dims):
    rho = get_rho_from_mi(mi, n_dims)
    print(rho)
    mu1 = torch.zeros((n_dims), dtype=torch.float32)
    mu2 = torch.zeros((n_dims), dtype=torch.float32)
    mu3 = torch.zeros((n_dims), dtype=torch.float32)
    
    scale_p = block_diag(*[[[1, rho], [rho, 1]] for _ in range(n_dims // 2)])
    print('rho', rho)
    print('scale_p', scale_p)
    scale_p = torch.from_numpy(scale_p).float()
    scale_q = torch.eye(n_dims, dtype=torch.float32)
    scale_m = torch.eye(n_dims, dtype=torch.float32)
    
    return get_dists(mu1, mu2, mu3, scale_p, scale_q, scale_m)

def get_rho_from_mi(mi, n_dims):
    """Get correlation coefficient from true mutual information"""
    x = (4 * mi) / n_dims
    return (1 - np.exp(-x)) ** 0.5  # correlation coefficient

def get_dists(mu1=0., mu2=2., mu3=2., scale_p=0.1, scale_q=0.1, scale_m=1.):
    print(mu1.shape)
    p = MultivariateNormal(
        loc=mu1,
        covariance_matrix=scale_p,
    )

    q = MultivariateNormal(
        loc=mu2,
        covariance_matrix=scale_q,
    )

    m = MixtureSameFamily(
        D.Categorical(torch.Tensor([0.6, 0.4])),
        D.Independent(MultivariateNormal(loc=torch.stack([mu1, mu2], dim=0), covariance_matrix=torch.stack([scale_p, scale_q], dim=0)), 0))

    return p, q, m

def get_dists_1d(mu1=0., mu2=2., mu3=2., scale_p=0.1, scale_q=0.1, scale_m=1.):
    p = Normal(
        loc=mu1,
        scale=scale_p,
    )

    q = Normal(
        loc=mu2,
        scale=scale_q,
    )
    
    m = Cauchy(
        loc=mu3,
        scale=scale_m
    )
    
    return p, q, m

def get_dists_1d_tre(mu1=0., mu2=2., mu3=2., scale_p=0.1, scale_q=0.1, alphas=[0.33, 0.66]):
    p = Normal(
        loc=mu1,
        scale=scale_p,
    )

    q = Normal(
        loc=mu2,
        scale=scale_q,
    )
    
    if len(alphas) == 0:
        raise Exception('No alphas defined')
    
    m_list = []
    
    for a in alphas:
        mu_m = np.sqrt((1.0 - a**2)) * mu1 + a * mu2
        scale_m = np.sqrt((1-a**2)*(scale_p**2)+(a**2)*(scale_q**2))
    
        m = Normal(loc=mu_m, scale=scale_m)
        m_list.append(m)
    
    return p, q, m_list

def get_dists_1d_cob(mu1=0., mu2=2., mu3=2., scale_p=0.1, scale_q=0.1, alphas=[0.33, 0.66]):
    p = Normal(
        loc=mu1,
        scale=scale_p,
    )

    q = Normal(
        loc=mu2,
        scale=scale_q,
    )
    
    if len(alphas) == 0:
        raise Exception('No alphas defined')
    
    m_list = []
    
    for a in alphas:
        mu_m = (1.0 - a) * mu1 + a * mu2
        scale_m = np.sqrt(((1.0-a)**2)*(scale_p**2) + (a**2)*(scale_q**2))
    
        m = Normal(loc=mu_m, scale=scale_m)
        m_list.append(m)
    
    return p, q, m_list

def get_dists_1d_overlap(mu1=0., mu2=2., mu3=2., scale_p=0.1, scale_q=0.1, scale_m=1.):
    p = Normal(
        loc=mu1,
        scale=scale_p,
    )

    q = Normal(
        loc=mu2,
        scale=scale_q,
    )
#     mu1 = torch.Tensor([mu1])
#     mu2 = torch.Tensor([mu2])
#     scale_p = torch.Tensor([scale_p])
#     scale_q = torch.Tensor([scale_q])
    
    m1 = MixtureSameFamily(
        D.Categorical(torch.Tensor([0.75, 0.25])),
        D.Independent(D.Normal(torch.Tensor([mu1, mu2]), torch.Tensor([scale_p, scale_q])),0))
    
    m2 = MixtureSameFamily(
        D.Categorical(torch.Tensor([0.5, 0.5])),
        D.Independent(D.Normal(torch.Tensor([mu1, mu2]), torch.Tensor([scale_p, scale_q])),0))
    
    m3 = MixtureSameFamily(
        D.Categorical(torch.Tensor([0.25, 0.75])),
        D.Independent(D.Normal(torch.Tensor([mu1, mu2]), torch.Tensor([scale_p, scale_q]) ),0))

    m = [p,m1,m2,m3,q]
    
    return p, q, m

def get_gt_ratio_kl(p, q, samples, calc_true_kl=False):
    ratio = p.log_prob(samples) - q.log_prob(samples)
    kl = torch.mean(ratio)
    
    # Calculate true KL btw p and q distributions
    if calc_true_kl:
        return ratio, kl, D.kl.kl_divergence(p, q)
    
    return ratio, kl
