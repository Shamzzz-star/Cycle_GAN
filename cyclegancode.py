import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 120
SAVE_MODEL = True
CHECKPOINT_GEN_O = "genz.pth.tar"
CHECKPOINT_GEN_A = "genh.pth.tar"
CHECKPOINT_CRITIC_O = "criticz.pth.tar"
CHECKPOINT_CRITIC_A = "critich.pth.tar"

# Data transformations
transforms = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2(),
], additional_targets={"image0": "image"})

# Dataset
class AppleOrangeDataset(Dataset):
    def __init__(self, root_apple, root_orange, transform=None):
        self.root_apple = root_apple
        self.root_orange = root_orange
        self.transform = transform
        self.apple_images = os.listdir(root_apple)
        self.orange_images = os.listdir(root_orange)
        self.length_dataset = max(len(self.apple_images), len(self.orange_images))
        self.apple_len = len(self.apple_images)
        self.orange_len = len(self.orange_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        apple_img = self.apple_images[index % self.apple_len]
        orange_img = self.orange_images[index % self.orange_len]

        apple_path = os.path.join(self.root_apple, apple_img)
        orange_path = os.path.join(self.root_orange, orange_img)

        apple_img = np.array(Image.open(apple_path).convert("RGB"))
        orange_img = np.array(Image.open(orange_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=apple_img, image0=orange_img)
            apple_img = augmentations["image"]
            orange_img = augmentations["image0"]

        return apple_img, orange_img

# Discriminator
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

# Generator
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity(),
            )
            if down else
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity(),
            )
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList([
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        ])
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList([
            ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

# Training utilities
def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def train_fn(disc_O, disc_A, gen_A, gen_O, loader, opt_disc, opt_gen, l1, mse):
    loop = tqdm(loader, leave=True)
    for idx, (apple, orange) in enumerate(loop):
        apple = apple.to(DEVICE)
        orange = orange.to(DEVICE)

        # Train Discriminators
        fake_orange = gen_O(apple).detach()
        fake_apple = gen_A(orange).detach()

        D_real_O = disc_O(orange)
        D_real_A = disc_A(apple)
        D_fake_O = disc_O(fake_orange)
        D_fake_A = disc_A(fake_apple)

        loss_D_O = mse(D_real_O, torch.ones_like(D_real_O)) + mse(D_fake_O, torch.zeros_like(D_fake_O))
        loss_D_A = mse(D_real_A, torch.ones_like(D_real_A)) + mse(D_fake_A, torch.zeros_like(D_fake_A))
        loss_D = (loss_D_O + loss_D_A) / 2

        opt_disc.zero_grad()
        loss_D.backward()
        opt_disc.step()

        # Train Generators
        fake_orange = gen_O(apple)
        fake_apple = gen_A(orange)

        D_fake_O = disc_O(fake_orange)
        D_fake_A = disc_A(fake_apple)

        loss_G_O = mse(D_fake_O, torch.ones_like(D_fake_O))
        loss_G_A = mse(D_fake_A, torch.ones_like(D_fake_A))

        cycle_apple = gen_A(fake_orange)
        cycle_orange = gen_O(fake_apple)
        cycle_loss = l1(apple, cycle_apple) + l1(orange, cycle_orange)

        identity_apple = gen_A(apple)
        identity_orange = gen_O(orange)
        identity_loss = l1(apple, identity_apple) + l1(orange, identity_orange)

        loss_G = (
            loss_G_A + loss_G_O + cycle_loss * LAMBDA_CYCLE + identity_loss * LAMBDA_IDENTITY
        )

        opt_gen.zero_grad()
        loss_G.backward()
        opt_gen.step()

        if idx % 50 == 0:
            save_image(fake_orange * 0.5 + 0.5, f"outputs/fake_zebra_{idx}.png")  # Rescale to [0, 1]
            save_image(fake_apple * 0.5 + 0.5, f"outputs/fake_horse_{idx}.png")


        loop.set_description(f"Epoch Progress")
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

# Main execution
def main():
    print("Training on:", DEVICE)

    disc_A = Discriminator().to(DEVICE)
    disc_O = Discriminator().to(DEVICE)
    gen_A = Generator(img_channels=3).to(DEVICE)
    gen_O = Generator(img_channels=3).to(DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_O.parameters()),
        lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_O.parameters()),
        lr=LEARNING_RATE, betas=(0.5, 0.999)
    )

    dataset = AppleOrangeDataset(
        root_apple=r"C:\Users\shamm\OneDrive\Desktop\horse2zebra\trainA",
        root_orange=r"C:\Users\shamm\OneDrive\Desktop\horse2zebra\trainB",
        transform=transforms,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        train_fn(disc_O, disc_A, gen_A, gen_O, loader, opt_disc, opt_gen, L1, mse)

        if SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen_A, opt_gen, filename=CHECKPOINT_GEN_A)
            save_checkpoint(gen_O, opt_gen, filename=CHECKPOINT_GEN_O)
            save_checkpoint(disc_A, opt_disc, filename=CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_O, opt_disc, filename=CHECKPOINT_CRITIC_O)

if __name__ == "__main__":
    main()
