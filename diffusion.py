import matplotlib.pyplot as plt
import numpy as np
import math

import torch
from torch import nn
from data import MNIST
from models import BasicUNet





class Diffusion(nn.Module):
    def __init__(self, T=1000, beta_limits=torch.Tensor([10**-4, 0.02]), device=None):
        super().__init__()
        self.T = T
        self.device = device if device is not None else torch.device('cpu')
        self.betas = self.construct_beta(beta_limits)
        self.alphas, self.alpha_bars = self.construct_alphas()

        # TODO: Initialise network
        self.net = BasicUNet().to(self.device)


    def construct_beta(self, beta_limits):
        """
        Paper does linear interpolation
        """

        # TODO: Write function that constructs variance schedule
        beta_1, beta_T = beta_limits
        return torch.linspace(beta_1, beta_T, self.T).to(self.device) # minus 1? Or is it fine like this?


    def construct_alphas(self):
        """
        a_t = 1 - b_t
        a_bar_t = cumprod_{s=1}^t a_s
        """

        # TODO: IMplement above
        alphas = 1 - self.betas.to(self.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(self.device)
        return alphas, alpha_bars


    def add_noise(self, x0, t):
        # TODO: Write function that implements math from paper to add noise to inputs gradually
        """
        Add noise to x0 according to difusion schedule at time t
        x0: [B, C, H, W] 
        t: int or tensor of ints [B]

        Math to compute xt from x0 is: 
        q(xt|x0) = N(xt; sqrt(a_bar_t)*x0, (1 - a_bar_t) * I)
        Instead of using that we use reparametrisation trick
        q(xt|x0) = sqrt(a_bar_t)*x0 + sqrt(1 - a_bar_t) * eps
        """
        if isinstance(t, torch.Tensor):
            t = t.reshape(-1, 1, 1, 1).to(self.device) 

        eps = torch.randn_like(x0, device=self.device)
        xt = torch.sqrt(self.alpha_bars[t])*x0 + torch.sqrt(1 - self.alpha_bars[t]) * eps
        return xt, eps


    def predict(self, x, t):
        """
        Predict noise (epsilon) from x_t and timestep t
        """
        return self.net(x, t)
    

    def sample(self, batch_size=4):
        """
        Generate samples by running the diffusion process in reverse
        """
        # Start from pure noise
        x = torch.randn(batch_size, 1, 28, 28)
        
        # Iterate backwards through timesteps
        for t in reversed(range(0, self.T)):
            # Convert to tensor and expand for batch
            t_tensor = torch.ones(batch_size, dtype=torch.long) * t
            
            # Predict noise
            eps = self.predict(x, t_tensor)
            
            # Reverse diffusion step
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            mean = 1/torch.sqrt(alpha)*(x - beta/torch.sqrt(1 - alpha_bar)*eps)

            if t > 0:
                noise = torch.randn_like(x)
                variance = torch.sqrt(beta)
                x = mean + variance*noise

            else:
                x = mean

        return x
    

    def visualize_samples(self, num_samples=4):
        """
        Generate and visualize num_samples samples in a grid
        """
        # Generate samples
        samples = self.sample(batch_size=num_samples)
        
        # Create a grid of subplots
        n = int(math.sqrt(num_samples))
        fig, axes = plt.subplots(n, n, figsize=(8, 8))
        
        # Plot each sample
        for i, ax in enumerate(axes.flatten()):
            if i < num_samples:
                ax.imshow(samples[i].squeeze().detach().numpy(), cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()     


def visualise(x):
    plt.figure(figsize=(5, 5))
    plt.imshow(x, cmap='gray')
    plt.show()


def visualize_noise_steps(model, image, num_images=10):
    """
    Visualize an image being progressively noised
    """
    # Create timesteps to visualize
    times = torch.linspace(0, model.T-1, num_images).long()
    
    # Create subplot grid
    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    
    # Plot original image first
    axes[0].imshow(image.squeeze(0), cmap='gray')
    axes[0].set_title('t=0')
    axes[0].axis('off')
    
    # Plot noised images
    for idx, t in enumerate(times[1:], 1):
        noised_image, _ = model.add_noise(image.unsqueeze(0), t)
        axes[idx].imshow(noised_image.squeeze().detach(), cmap='gray')
        axes[idx].set_title(f't={t.item()}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    trainloader, testloader = MNIST()
    model = Diffusion()

    datapoint = next(iter(trainloader))
    images, labels = datapoint
    image, label = images[0], labels[0]
    # visualise(image.squeeze(0))

    xt, eps = model.add_noise(image, 100)
    # visualise(xt[0])
    visualize_noise_steps(model, image)

    model.visualize_samples()


