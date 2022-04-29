import torch

def tensor_erode(bin_img, ksize=3):
    eroded = torch.nn.MaxPool2d(kernel_size=ksize,stride=1,padding=int((ksize-1)/2))
    return eroded
