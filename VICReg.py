
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optimizers
class VICReg(nn.Module):
    
    def __init__(self):
        self.lamb = 25 # invar loss weight
        self.mu = 25 # var loss weight
        self.nu = 1 # covar loss weight
        self.input_vector_size = 32 * 32
        self.expander_vector_size = 500
        self.encoder_vector_size = 400
        self.encoder = ResNet50_Weights.DEFAULT.transforms()
        self.expander = Expander(input_size=self.encoder_vector_size, output_size=self.expander_vector_size)
        self.optimizer = optimizers.SGD
        self.train_epochs = 50
        self.batch_size = 256
        self.base_lr = .01
        self.lr = (self.batch_size / 256) * self.base_lr# Batch_size / 256 * base_lr
    
    def augment(self):
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
        transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale([.2]), # [BETA] Randomly convert image or videos to grayscale with a probability of p (default 0.1).
            transforms.GaussianBlur(kernel_size , sigma), # [BETA] Blurs image with randomly chosen Gaussian blur.
            transforms.RandomSolarize(solarize_threshold, p = .1),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transforms
        
        
    def calculate_loss(self, z, z_prime):
        """Calculate the loss function. 
        
        The following is heavily based on the psuedo code provided on page 13 of https://arxiv.org/pdf/2105.04906.pdf
        
        Need to calculate 3 things: 
        
        1. Variance
        
        2. Invariance
        
        3. Covariance
        
        Args:
            z (_type_): batch of images transformed, encodeded, projected
            z_prime (_type_): batch of images transformed, encodeded, projected
        """
        
        # 1. Variance Loss
        var_epsilon = 1e-4
        std_z = torch.sqrt(z.var(dim=0) + var_epsilon)
        std_z_prime = torch.sqrt(z_prime.var(dim=0) + var_epsilon)
        std_loss = torch.mean(nn.relu(1 - std_z)) + torch.mean(nn.relu(1 - std_z_prime))
        
        
        # 2. Invariance Loss (Just MSE Loss)
        invar_loss = nn.MSELoss(z, z_prime)
        
        # 3. Covariance Loss
        
        N , D = z.shape
        z = z - z.mean(dim=0)
        z_prime = z_prime - z_prime.mean(dim=0)
        cov_z = (z.T @ z) / (N - 1)
        cov_z_prime = (z_prime.T @ z_prime) / (N - 1)
        
        cov_z = cov_z.pow(2)
        cov_z_prime = cov_z_prime.pow(2)
        
        loss_c_a = (cov_z.sum() - cov_z.diagonal().sum()) / D
        loss_c_b = (cov_z_prime.sum() - cov_z_prime.diagonal().sum()) / D
        
        loss_cov = loss_c_a + loss_c_b
        
        loss = self.lamb * invar_loss + self.mu * std_loss + self.nu * loss_cov
        
        return loss
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.expander(x)
        
        return x  
    
class Expander(nn.Module):
    def __init__(self, input_size, output_size=8192):
        """
        
        expander hφ:
        Composed of two fully-connected layers with batch normalization and ReLU,
        and a third linear layer. The sizes of all 3 layers were set to 8192

        Args:
            input_size (int): cifar vector size 
            output_size (int): output vector size, also size of intermediate linear layers.
        """
        super(Expander, self).__init__()

        # First fully-connected layer
        self.fc1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()

        # Second fully-connected layer
        self.fc2 = nn.Linear(output_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu2 = nn.ReLU()

        # Third linear layer
        self.fc3 = nn.Linear(output_size, output_size)

    def forward(self, x):
        # Forward pass through the layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x
   