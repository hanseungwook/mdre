import torch


IMG_SIZE = 28


# Creates masks for mixing p and q (dimension mixing -- mixing images in grid)
# Returns masks of shape (num_masks, 2 (p_mask, q_mask), *** (shape of data))
def get_dim_mix_masks(shape):
    # Assumes each image is of size 28 x 28
    
    masks = []
    ones = torch.ones(shape)
    p_mask = torch.ones(shape)
    
    assert shape[-1] == shape[-2]
    n_img = shape[-1] // IMG_SIZE
    print('n_img', n_img)
    
    # Mask for p and q joint
#     masks.append(torch.stack([p_mask.clone(), ones - p_mask], dim=0))
    
    for i in range(n_img):
        for j in range(n_img):
            p_mask_copy = p_mask
            p_mask_copy[:, i*IMG_SIZE:(i+1)*IMG_SIZE, j*IMG_SIZE:(j+1)*IMG_SIZE].fill_(0)
            q_mask = ones - p_mask_copy
            
            # Skipping last element in the grid b/c then it becomes fully q distribution
            if i == (n_img - 1) and j == (n_img - 1):
                continue

            masks.append(torch.stack([p_mask_copy.clone(), q_mask], dim=0))
            
    return torch.stack(masks, dim=0)
    