import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from PIL import Image
from tqdm import tqdm
import numpy as np

# === Blocks needed by Generator ===

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

# === Generator ===

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

# === Load Generator Checkpoint ===

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

# === FCN Score Evaluation ===

def calculate_fcn_score(gen, input_dir, fcn_model, fcn_transform, device):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    files = [f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    fcn_scores = []

    with torch.no_grad():
        for file in tqdm(files, desc="Evaluating FCN score"):
            img_path = os.path.join(input_dir, file)
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            fake_img = gen(input_tensor)
            fake_img = (fake_img + 1) / 2
            fake_img = fake_img.clamp(0, 1)

            # Apply FCN preprocessing
            fcn_ready_img = fcn_transform(fake_img.squeeze(0)).unsqueeze(0).to(device)

            # Pass through FCN
            fcn_output = fcn_model(fcn_ready_img)['out']  # (1, num_classes, H, W)
            confidence_score = torch.softmax(fcn_output, dim=1).max(dim=1)[0].mean().item()

            fcn_scores.append(confidence_score)

    avg_fcn_score = np.mean(fcn_scores)
    print(f"\n✅ Average FCN Confidence Score: {avg_fcn_score:.4f}")

# === Main Execution ===

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator_ckpt = "result/100eppoch/geno.pth.tar"  # path to generator checkpoint
    input_dir = "datasets/apple2orange/testA"           # path to input images

    # Load Generator
    gen = Generator(img_channels=3)
    load_checkpoint(generator_ckpt, gen, device)

    # Load FCN model with proper weights
    weights = FCN_ResNet101_Weights.DEFAULT
    fcn_model = fcn_resnet101(weights=weights).eval().to(device)
    fcn_transform = weights.transforms()

    # Run FCN score calculation
    calculate_fcn_score(gen, input_dir, fcn_model, fcn_transform, device)

if __name__ == "__main__":
    main()
