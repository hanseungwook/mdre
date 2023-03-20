import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import make_grid

class DistDataset(Dataset):
    def __init__(self, p, q, m=None, num_samples=100000, alphas=[]):
        self.p = p
        self.q = q
        self.m = m
    
        print('Sampling p')
        self.p_samples = p.sample((num_samples,))
        print('Sampling q')
        self.q_samples = q.sample((num_samples,))
        
        print(m)

        # Linear interpolation between p and q using given alphas, if m not defined
        if m is None:
            print('Linear mixing for m samples')
#             alphas = torch.from_numpy(np.tile(torch.Tensor([0.0,6.103515625e-05,0.0078125,0.13348388671875,1.0]), (num_samples // 5,)))
#             alphas = torch.from_numpy(np.tile(torch.Tensor([0., 0.5, 0.75, 0.75, 1.0]), (num_samples // 5,)))
            alphas = torch.from_numpy(np.tile(torch.Tensor([0.5]), (num_samples,)))
            print(alphas.shape)
            
            
#             alphas = torch.tile(torch.Tensor([0., 0.5, 0.75, 0.75, 1.0]), (num_samples // 5,)).unsqueeze(1)
            self.m_samples = torch.sqrt(1-alphas**2)*self.p_samples + alphas*self.q_samples
            
        elif isinstance(m,list):
            self.m_samples = torch.cat([dist.sample([num_samples//len(m)]) for dist in self.m])
        else:
            print('Sampling m')
            self.m_samples = m.sample((num_samples,))
        print(self.p_samples.shape)
        print(self.q_samples.shape)
        print(self.m_samples.shape)
    def __getitem__(self, idx):
        return self.p_samples[idx], self.q_samples[idx], self.m_samples[idx]
    
    def __len__(self):
        return len(self.p_samples)

    def get_p_samples(self):
        return self.p_samples
    
    def get_q_samples(self):
        return self.q_samples
    
    def get_m_samples(self):
        return self.m_samples
    

class DistDataset2Waymark(Dataset):
    def __init__(self, p, q, m, num_samples=100000):
        self.p = p
        self.q = q
        self.m = m
    
        print('Sampling p')
        self.p_samples = p.sample((num_samples,))
        print('Sampling q')
        self.q_samples = q.sample((num_samples,))
        
        assert len(self.m) == 2
        self.m_samples1 = self.m[0].sample([num_samples])
        self.m_samples2 = self.m[1].sample([num_samples])

        print(self.p_samples.shape)
        print(self.q_samples.shape)
        print(self.m_samples1.shape)
        print(self.m_samples2.shape)
    def __getitem__(self, idx):
        return self.p_samples[idx], self.q_samples[idx], self.m_samples1[idx], self.m_samples2[idx]
    
    def __len__(self):
        return len(self.p_samples)

    def get_p_samples(self):
        return self.p_samples
    
    def get_q_samples(self):
        return self.q_samples
    
class DistDataset3Waymark(Dataset):
    def __init__(self, p, q, m, num_samples=100000):
        self.p = p
        self.q = q
        self.m = m
    
        print('Sampling p')
        self.p_samples = p.sample((num_samples,))
        print('Sampling q')
        self.q_samples = q.sample((num_samples,))
        
        assert len(self.m) == 3
        self.m_samples1 = self.m[0].sample([num_samples])
        self.m_samples2 = self.m[1].sample([num_samples])
        self.m_samples3 = self.m[2].sample([num_samples])

        print(self.p_samples.shape)
        print(self.q_samples.shape)
        print(self.m_samples1.shape)
        print(self.m_samples2.shape)
        print(self.m_samples3.shape)
    def __getitem__(self, idx):
        return self.p_samples[idx], self.q_samples[idx], self.m_samples1[idx], self.m_samples2[idx], self.m_samples3[idx]
    
    def __len__(self):
        return len(self.p_samples)

    def get_p_samples(self):
        return self.p_samples
    
    def get_q_samples(self):
        return self.q_samples
    

class SpatialOmniDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        # Loading data
        data = np.load(data_path)
        self.imgs = data['data']
        self.labels = data['labels']
        self.preprocess()
        
        # Transforms
        self.transforms = transforms
    
    def preprocess(self):
        # Tensorfying imgs data
        # Actually supposed to be 255.0 to tensorfy
        self.imgs = self.imgs / 256.0
       
    def __getitem__(self, idx):
        img = self.imgs[idx].squeeze()
        
        # 1 image in the grid
        if len(img.shape) == 3:
            img_u, img_v = torch.from_numpy(img[:, :, 0]).unsqueeze(0), torch.from_numpy(img[:, :, 1]).unsqueeze(0)
        elif len(img.shape) == 4:
            img_u, img_v = img[:, :, :, 0], img[:, :, :, 1]
            
            # Make into a grid of images (collate all images into 1 image) and use only 1 channel b/c make_grid automatically copies into 3 channels
            img_u = make_grid(torch.from_numpy(img_u.transpose(2,0,1)).unsqueeze(1), padding=0, nrow=int(np.sqrt(img_u.shape[-1])))[0].unsqueeze(0)
            img_v = make_grid(torch.from_numpy(img_v.transpose(2,0,1)).unsqueeze(1), padding=0, nrow=int(np.sqrt(img_v.shape[-1])))[0].unsqueeze(0)
        else:
            raise NotImplementedError('Not supporting > 4 dim image tensors in dataset')
        # More than 1 images in the grid
        
        # Apply transform, if defined
        if self.transforms:
            img_u = self.transforms(img_u)
            img_v = self.transforms(img_v)
            print('applied tf')
        
        return img_u, img_v, self.labels[idx]
    
    def __len__(self):
        return len(self.labels)

class PairedSpatialOmniDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        j = np.random.randint(0, len(self))
        return tuple([self.datasets[0][i], self.datasets[1][j]])
#         return tuple(d[i] for d in self.datasets)

    def __len__(self):
        all_len = torch.Tensor([len(d) for d in self.datasets])
        # Check datasets are of equal length
        assert (all_len == all_len[0]).all()

        return min(len(d) for d in self.datasets)
    
    
class SpatialOmniPairDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        # Loading data
        data = np.load(data_path)
        self.imgs = data['data']
        self.labels = data['labels']
        self.preprocess()
        
        # Transforms
        self.transforms = transforms
    
    def preprocess(self):
        # Tensorfying imgs data
        # Actually supposed to be 255.0 to tensorfy
        self.imgs = self.imgs / 256.0
       
    def __getitem__(self, idx):
        img1 = self.imgs[idx].squeeze()
        img2 = self.imgs[np.random.randint(0, len(self))]
        
        # 1 image in the grid
        if len(img1.shape) == 3:
            img_p_u, img_p_v = torch.from_numpy(img1[:, :, 0]).unsqueeze(0), torch.from_numpy(img1[:, :, 1]).unsqueeze(0)
            img_q_u, img_q_v = torch.from_numpy(img2[:, :, 0]).unsqueeze(0), torch.from_numpy(img2[:, :, 1]).unsqueeze(0)
            
        elif len(img1.shape) == 4:
            img_p_u, img_p_v = img1[:, :, :, 0], img1[:, :, :, 1]
            img_q_u, img_q_v = img2[:, :, :, 0], img2[:, :, :, 1]
            
            # Make into a grid of images (collate all images into 1 image) and use only 1 channel b/c make_grid automatically copies into 3 channels
            img_p_u = make_grid(torch.from_numpy(img_p_u.transpose(2,0,1)).unsqueeze(1), padding=0, nrow=int(np.sqrt(img_p_u.shape[-1])))[0].unsqueeze(0)
            img_p_v = make_grid(torch.from_numpy(img_p_v.transpose(2,0,1)).unsqueeze(1), padding=0, nrow=int(np.sqrt(img_p_v.shape[-1])))[0].unsqueeze(0)
            
            img_q_u = make_grid(torch.from_numpy(img_q_u.transpose(2,0,1)).unsqueeze(1), padding=0, nrow=int(np.sqrt(img_q_u.shape[-1])))[0].unsqueeze(0)
            img_q_v = make_grid(torch.from_numpy(img_q_v.transpose(2,0,1)).unsqueeze(1), padding=0, nrow=int(np.sqrt(img_q_v.shape[-1])))[0].unsqueeze(0)
        else:
            raise NotImplementedError('Not supporting > 4 dim image tensors in dataset')
        # More than 1 images in the grid
        
        # Apply transform, if defined
        if self.transforms:
            img_p_u, img_p_v, img_q_u, img_q_v = self.transforms(img_p_u), self.transforms(img_p_v), self.transforms(img_q_u), self.transforms(img_q_v)
            print('applied tf')
        
        # Shuffle q_v for marginal
#         img_q_v = img_q_v[torch.randperm(img_q_v.shape[0])]
        
        return img_p_u, img_p_v, img_q_u, img_q_v, self.labels[idx]
    
    def __len__(self):
        return len(self.labels)
    
    
        