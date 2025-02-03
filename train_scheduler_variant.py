import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from data import MNIST
from diffusion import Diffusion

def train(epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu', 
          batch_size=128, sample_dir='samples', UNet='basic'):
    # Create sample directory
    os.makedirs(sample_dir, exist_ok=True)
    
    # Setup
    trainloader, testloader = MNIST(batch_size=batch_size)  # Update data loader with batch size
    model = Diffusion(device=device, UNet=UNet)
    loss_fn = torch.nn.MSELoss()

    # Higher initial learning rate with OneCycleLR
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # OneCycleLR scheduler
    # pct_start determines what fraction of training is in "warmup" phase
    # max_lr is peak learning rate after warmup
    # Steps per epoch = len(trainloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1000,  # Final lr = initial_lr/1000
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for batch in pbar:
                x0, _ = batch
                x0 = x0.to(device)
                batch_size = x0.shape[0]

                optim.zero_grad()

                t = torch.randint(1, model.T, (batch_size,), device=device)
                xt, eps = model.add_noise(x0, t)
                predicted_eps = model.predict(xt, t)
                loss = loss_fn(eps, predicted_eps)
                
                loss.backward()
                optim.step()
                scheduler.step()  # Step every batch instead of every epoch

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