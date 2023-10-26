
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2
import torch
import torch.nn as nn

class VICReg():
    def __init__():
        pass
    
    def augment():
        """
        1. Random Cropping with size ratio between .08 and 1.0 with resizing. RandomResizedCrop(32, scale=(0.08, 0.1)) in PyTorch.
        2. Random horizontal flip with probability 0.5.
        3. Color jittering of brightness, contrast, saturation and hue, with probability 0.8.
        ColorJitter(0.4, 0.4, 0.2, 0.1) in PyTorch.
        4. Grayscale with probability 0.2
        5. Gaussian blur with probability 0.5 and kernel size 23. (Do we keep the sample kernel size for cifar-10?)
        6. Solarization with probability 0.1.
        7. Color normalization with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224,
        0.225).
        
        https://pytorch.org/vision/main/transforms.html
        
        GAUSSIAN_BLUR:
        https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html#torchvision.transforms.GaussianBlur
            Inputs: 
            - kernel_size (int or sequence) – Size of the Gaussian kernel.
            - sigma (float or tuple of python:float (min, max)) – Standard deviation to be used for creating kernel to perform blurring.
            If float, sigma is fixed. If it is tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range.
        
        Args:
            x (_type_): image vector to transform
        """
        kernel_size=23
        sigma=(0.1, 2.0)
        solarize_threshold = .5
        transforms = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(0.4, 0.4, 0.2, 0.1),
            v2.RandomGrayscale([.2]), # [BETA] Randomly convert image or videos to grayscale with a probability of p (default 0.1).
            v2.GaussianBlur(kernel_size , sigma), # [BETA] Blurs image with randomly chosen Gaussian blur.
            v2.RandomSolarize(solarize_threshold, p = .1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
        
    def get_loss(x, self):
        """
            # f: encoder network, lambda, mu, nu: coefficients of the
            invariance, variance and covariance losses, N: batch size
            , D: dimension of the representations
            # mse_loss: Mean square error loss function, off_diagonal:
            off-diagonal elements of a matrix, relu: ReLU activation
            function
        """
        # Initialize the Weight Transforms
        weights = ResNet50_Weights.DEFAULT
        f_encoder = weights.transforms()


        for x in loader: # load a batch with N samples
            # two randomly augmented versions of x
            x_a, x_b = self.augment(x)
            # compute representations
            z_a = f_encoder(x_a) # N x D
            z_b = f_encoder(x_b) # N x D
            # invariance loss
            sim_loss = nn.MSELoss(z_a, z_b)
            # variance loss
            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            std_loss = torch.mean(nn.ReLU(1 - std_z_a)) + torch.mean(nn.ReLU(1 - std_z_b))
            # covariance loss
            z_a = z_a - z_a.mean(dim=0)
            z_b = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a.T @ z_a) / (N - 1)
            cov_z_b = (z_b.T @ z_b) / (N - 1)
            cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D
            + off_diagonal(cov_z_b).pow_(2).sum() / D
            # loss
            loss = lambda * sim_loss + mu * std_loss + nu * cov_loss
            # optimization step
            loss.backward()
            optimizer.step()



