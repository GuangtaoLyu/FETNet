import torch.nn.functional as F

def tensor_erode(bin_img, ksize=3):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded