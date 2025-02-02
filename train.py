import torch
from tqdm import tqdm
from data import MNIST
from diffusion import Diffusion

def train(epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Setup
    trainloader, testloader = MNIST()
    model = Diffusion(device=device)
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

        # Step scheduler
        scheduler.step()

        # Maybe visualize some samples periodically
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                model.visualize_samples()

if __name__ == '__main__':
    train(epochs=1, device=torch.device('mps'))