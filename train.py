import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from data import MNIST
from diffusion import Diffusion

def train(epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu', 
         sample_dir='samples', UNet='basic', batch_size=64):
   # Create sample directory
   os.makedirs(sample_dir, exist_ok=True)

   print(f'Training on {device} with UNet={UNet}')
   
   # Setup
   trainloader, testloader = MNIST(batch_size=batch_size)
   model = Diffusion(device=device, UNet=UNet)
   loss_fn = torch.nn.MSELoss()
   optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

   # Training loop
   for epoch in range(epochs):
       model.train()
       epoch_loss = 0
       with tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
           for batch in pbar:
               # Get batch
               x0, _ = batch
               x0 = x0.to(device)
               batch_size = x0.shape[0]

               # Zero gradients
               optim.zero_grad()

               # Sample timestep
               t = torch.randint(1, model.T, (batch_size,), device=device)

               # Add noise and try to predict it
               xt, eps = model.add_noise(x0, t)
               predicted_eps = model.predict(xt, t)

               # Calculate loss
               loss = loss_fn(eps, predicted_eps)
               
               # Backprop
               loss.backward()
               optim.step()

               # Update progress bar
               epoch_loss += loss.item()
               pbar.set_postfix({'loss': epoch_loss/(pbar.n+1)})

       # Save samples after each epoch
       model.eval()
       with torch.no_grad():
           # Generate samples
           samples = model.sample(batch_size=16)  # Generate 16 samples
           
           # Save grid of samples
           samples = (samples + 1) / 2  # Denormalize from [-1,1] to [0,1]
           save_image(samples, 
                     os.path.join(sample_dir, f'samples_epoch_{epoch+1}.png'),
                     nrow=4)  # 4x4 grid
           
           # Save model checkpoint if needed
           if (epoch + 1) % 10 == 0:  # Every 10 epochs
               torch.save({
                   'epoch': epoch,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optim.state_dict(),
                   'loss': epoch_loss,
               }, os.path.join(sample_dir, f'checkpoint_epoch_{epoch+1}.pt'))

       # Step scheduler
       scheduler.step()

if __name__ == '__main__':
   train(
    epochs=30, 
    UNet='deepwide', 
    batch_size=128
    )