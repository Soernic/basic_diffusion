import torch
import math

from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        """
        This is the time embedding for the UNet. 
        We choose a dimensionality of the time embedding, default 256
        Forward method computes embedding for a given time
        Nothing learnable here, just a convenient way to set it up

        Intuition is
        - We want something like [sin(t), cos(t)] but we want it to be a bit more expressive
        - Standard form is to do it something like this, which is what this is
            > # For t = 5, dim = 8, you'd get something like:
            > sin(5 * 1), sin(5 * 0.1), sin(5 * 0.01), sin(5 * 0.001),
            > cos(5 * 1), cos(5 * 0.1), cos(5 * 0.01), cos(5 * 0.001)
        - Why do we want that? cos(t), sin(t) is great and all, but if we want something unique we have to go further
        - The larger versions with more complexity allow for much more flexibility in the embeddings
        """
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BasicUNet(nn.Module):
    def __init__(self, time_dim=256):
        super().__init__()
        """
        This creates a basic encoder-decoder type network with convolutional layers
        The only added complexity here is the use of a time embedding
        """

        # Initialise time embedding block
        # Time embedding has dim=256 by default
        # Passes through a linear layer with bias
        # ReLU activation across vector
        # Produces 256-dimensional vector with non-negative values
        # Shape -> [B, 256]
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim), #
            nn.ReLU()
        )

        # Encoder
        # Input: [B, 1, 28, 28]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), # 1 channel to 64, 3x3 kernel, padding=1 (same size) (-> [B, 64, 28, 28])
            nn.BatchNorm2d(64), # stabilisation
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), # 64 -> 64, 3x3, padding=same (-> [B, 64, 28, 28])
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2) # [B, 64, 28, 28] -> [B, 64, 14, 14]

        # .. -> [B, 64, 14, 14]
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2) # [B, 64, 14, 14] -> [B, 64, 28, 28]

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), # -> [B, 64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1) # -> [B, 1, 28, 28]
        )


    def forward(self, x, t):
        # Encode
        x1 = self.enc1(x)
        x = self.pool1(x1)

        # Sinusoidal embedding is calculated and then passed through MLP
        t = self.time_mlp(t)

        # Get shape
        b, c, h, w = x.shape # [B, 64, 14, 14] (bc. after pooling operation)
        t = t.view(b, -1, 1, 1)            # [B, 256, 1, 1]
        time_emb = t.repeat(1, 1, h, w)    # [B, 256, 14, 14]
        x = x + time_emb[:, :c] # only use as many channels as we have features so that we can add them

        # middle
        x = self.middle(x) # bottleneck

        # decode
        x = self.up1(x)
        x = self.dec1(x) # 1 channel now, just [B, 1, 28, 28]

        return x
    

class BasicUNet_Wide(nn.Module):
    """Variant 1: More channels in intermediate layers"""
    def __init__(self, time_dim=256):
        super().__init__()
        
        # Time embedding (same as before)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder (wider)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),    # Increased from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),  # Increased from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        # Middle (wider)
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # Increased from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Decoder (wider)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)  # Increased from 64 to 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # Increased from 64 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1)     # Output still 1 channel
        )

    def forward(self, x, t):
        # Encode
        x1 = self.enc1(x)
        x = self.pool1(x1)

        # Time embedding
        t = self.time_mlp(t)
        b, c, h, w = x.shape
        time_emb = t.view(b, -1, 1, 1).repeat(1, 1, h, w)
        x = x + time_emb[:, :c]

        # Middle
        x = self.middle(x)

        # Decode
        x = self.up1(x)
        x = self.dec1(x)
        
        return x

class BasicUNet_Deep(nn.Module):
    """Variant 2: Deeper network (more levels)"""
    def __init__(self, time_dim=256):
        super().__init__()
        
        # Time embedding (same as before)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder (deeper)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder (deeper)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        # Encode
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)

        # Time embedding
        t = self.time_mlp(t)
        b, c, h, w = x.shape
        time_emb = t.view(b, -1, 1, 1).repeat(1, 1, h, w)
        x = x + time_emb[:, :c]

        # Middle
        x = self.middle(x)

        # Decode
        x = self.up2(x)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = self.dec1(x)
        
        return x

class BasicUNet_DeepWide(nn.Module):
    """Variant 3: Both deeper and wider"""
    def __init__(self, time_dim=256):
        super().__init__()
        
        # Time embedding (same as before)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder (deeper and wider)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder (deeper and wider)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        # Encode
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)

        # Time embedding
        t = self.time_mlp(t)
        b, c, h, w = x.shape
        time_emb = t.view(b, -1, 1, 1).repeat(1, 1, h, w)
        x = x + time_emb[:, :c]

        # Middle
        x = self.middle(x)

        # Decode
        x = self.up2(x)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = self.dec1(x)
        
        return x


if __name__ == '__main__':
        # Test the network
    net = BasicUNet()
    x = torch.randn(4, 1, 28, 28)  # 4 MNIST images
    t = torch.tensor([1, 2, 3, 4])  # 4 timesteps
    output = net(x, t)
    print(output.shape)  # Should be [4, 1, 28, 28]