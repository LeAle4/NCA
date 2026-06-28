import torch
import torch.nn.functional as F

from config import DEVICE, STD_DTYPE

SOBELX_KERNEL = (1/8) * torch.tensor([[
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]], dtype=STD_DTYPE, device=DEVICE)

SOBELY_KERNEL = torch.tensor([[
    [-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2, 1]
]], dtype=STD_DTYPE, device=DEVICE)

LAPLACIAN_KERNEL = (1/4) * torch.tensor([[
    [1,2,1],
    [2,-12,2],
    [1,2,1]
]],dtype=STD_DTYPE, device=DEVICE)

IDENTITY_KERNEL = torch.tensor([[
    [0,0,0],
    [0,1,0],
    [0,0,0]
]], dtype=STD_DTYPE, device=DEVICE)

AVERAGE_KERNEL = (1/9) * torch.tensor([[
    [1,1,1],
    [1,1,1],
    [1,1,1]
]], dtype=STD_DTYPE, device=DEVICE)

BIHARMONIC_KERNEL = F.conv2d(LAPLACIAN_KERNEL.unsqueeze(0), LAPLACIAN_KERNEL.unsqueeze(0), padding=2, groups = 1).squeeze(0)

def make_5x5(kernel):
    return F.pad(kernel, (1,1,1,1), mode="constant", value = 0)

def get_kernels(names:tuple[str]) -> torch.Tensor:
    """"Helper function to obtain certain convolutional kernels depending on their name"""
    kernels = []
    expand = True if "biharmonic" in names else False
    for name in names:
        if "biharmonic" in name:
            kernels.append(BIHARMONIC_KERNEL)
        elif "sobel" in name:
            if expand:
                kernels.append(make_5x5(SOBELX_KERNEL))
                kernels.append(make_5x5(SOBELY_KERNEL))   
            else:
                kernels.append(SOBELX_KERNEL)
                kernels.append(SOBELY_KERNEL)
        elif "laplacian" in name:
            if expand:
                kernels.append(make_5x5(LAPLACIAN_KERNEL))
            else:
                kernels.append(LAPLACIAN_KERNEL)
        elif "identity" in name:
            if expand:
                kernels.append(make_5x5(IDENTITY_KERNEL))
            else:
                kernels.append(IDENTITY_KERNEL)
        elif "average" in name:
            if expand:
                kernels.append(make_5x5(AVERAGE_KERNEL))
            else:
                kernels.append(AVERAGE_KERNEL)
    
    kernels = torch.stack(kernels)
    return kernels