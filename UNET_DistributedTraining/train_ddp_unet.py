import os
import glob
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import time

import csv
from skimage.metrics import peak_signal_noise_ratio as psnr

import lpips

# === Data Loading and Preprocessing ===
data_path = "/home/sulaimon/EXPERIMENT/23/images_001"
image_files = glob.glob(os.path.join(data_path, '*.png'))
#print(f'Found {len(image_files)} Images.')

#=====Load Data=====
def load_xray_image(data_path):
    img = Image.open(data_path).convert('L').resize((256, 256))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

InputImages = torch.tensor(np.array([load_xray_image(f) for f in image_files]), dtype=torch.float32)


#=====Add Noise=====
def add_gaussian_noise(images, mean=0.1, stddev=0.1):
    noise = torch.normal(mean, std=stddev, size=images.shape)
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0., 1.)

noisy_images = add_gaussian_noise(InputImages)

class X_rayDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

# === Unet Model Definition ===
class UNet(nn.Module):
    def __init__(self, input_channels=1, Nc=64):
        super(UNet, self).__init__()
        self.conv1 = self.conv_block(input_channels, Nc)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = self.conv_block(Nc, Nc * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = self.conv_block(Nc * 2, Nc * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = self.conv_block(Nc * 4, Nc * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = self.conv_block(Nc * 8, Nc * 16)
        self.upconv6 = nn.ConvTranspose2d(Nc * 16, Nc * 8, kernel_size=2, stride=2)
        self.conv6 = self.conv_block(Nc * 16, Nc * 8)
        self.upconv7 = nn.ConvTranspose2d(Nc * 8, Nc * 4, kernel_size=2, stride=2)
        self.conv7 = self.conv_block(Nc * 8, Nc * 4)
        self.upconv8 = nn.ConvTranspose2d(Nc * 4, Nc * 2, kernel_size=2, stride=2)
        self.conv8 = self.conv_block(Nc * 4, Nc * 2)
        self.upconv9 = nn.ConvTranspose2d(Nc * 2, Nc, kernel_size=2, stride=2)
        self.conv9 = self.conv_block(Nc * 2, Nc)
        self.final_conv = nn.Conv2d(Nc, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))
        x6 = self.conv6(torch.cat([self.upconv6(x5), x4], dim=1))
        x7 = self.conv7(torch.cat([self.upconv7(x6), x3], dim=1))
        x8 = self.conv8(torch.cat([self.upconv8(x7), x2], dim=1))
        x9 = self.conv9(torch.cat([self.upconv9(x8), x1], dim=1))
        return self.final_conv(x9)

# === Metrics ===
def psnr_metric(y_true, y_pred):
    mse = F.mse_loss(y_pred, y_true)
    return 10 * torch.log10(1.0 / mse)

def ssim_metric(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    ssim_scores = [ssim(y_true[i,0], y_pred[i,0], data_range=1.0) for i in range(y_true.shape[0])]
    return torch.tensor(ssim_scores).mean()

# === DDP Setup ===
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

#=====Data Loader=====
def prepare_dataloaders(noisy_images, clean_images, local_rank):
    dataset = X_rayDataset(noisy_images, clean_images)
    train_size = int(0.5 * len(dataset))
    val_size = int(0.33 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=32, sampler=DistributedSampler(train_set), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, sampler=DistributedSampler(val_set, shuffle=False), num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

# === Test Evaluation ===
def test_model(model, test_loader, device):
    model.eval()
    model.to(device)

    total_psnr, total_ssim, total_lpips, num_samples = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)

            for i in range(noisy.shape[0]):
                y_true = clean[i,0].cpu().numpy()
                y_pred = outputs[i,0].cpu().numpy()

                total_psnr += psnr_metric(torch.tensor(y_true), torch.tensor(y_pred))
                total_ssim += ssim(y_true, y_pred, data_range=1.0, win_size=3)

                # LPIPS requires 3-channel images in [-1, 1]
                pred_img = outputs[i].unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
                gt_img = clean[i].unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1

                lpips_score = loss_fn_lpips(pred_img.to(device), gt_img.to(device))
                total_lpips += lpips_score.item()
                num_samples += 1
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples

    return avg_psnr.item(), avg_ssim, avg_lpips

# === Training ===
def train_model_ddp(model, train_loader, val_loader, test_loader, local_rank, epochs=50, save_path="best_model_unet_DDP_10.pth"):
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    scaler = GradScaler()
    best_loss = float('inf')
    total_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        train_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(noisy)
                loss = loss_fn(outputs, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss, total_psnr, total_ssim, total_samples = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                with autocast():
                    outputs = model(noisy)
                    loss = loss_fn(outputs, clean)
                val_loss += loss.item()
                total_samples += noisy.size(0)
                total_psnr += psnr_metric(clean, outputs).item()
                total_ssim += ssim_metric(clean, outputs).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = total_psnr / total_samples
        avg_val_ssim = total_ssim / total_samples

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | PSNR: {avg_val_psnr:.2f} dB | SSIM: {avg_val_ssim:.4f}")
            if (epoch + 1) % 10 == 0:
                epoch_time = time.time() - total_start_time
                test_psnr, test_ssim, test_lpips = test_model(model.module, test_loader, device)
                print(f"Elapsed: {epoch_time:.2f}s | Test PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f} | LPIPS: {test_lpips:.4f}")
           
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.module.state_dict(), save_path)
                print(f"Saved best model (val loss: {best_loss:.6f})")

#======== Save Metrics =========
def save_metrics_csv_from_model(model, test_loader, device="cpu", csv_path="unet_DDP_10.csv"):
    model.eval()
    model.to(device)

    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    count = 0

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index", "PSNR", "SSIM", "LPIPS"])

        with torch.no_grad():
            for noisy, clean in test_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)

                outputs = model(noisy)

                clean_np = clean.squeeze(1).cpu().numpy()     # (B, H, W)
                outputs_np = outputs.squeeze(1).cpu().numpy() # (B, H, W)

                for i in range(clean_np.shape[0]):
                    # PSNR & SSIM
                    psnr_val = psnr(clean_np[i], outputs_np[i], data_range=1.0)
                    ssim_val = ssim(clean_np[i], outputs_np[i], data_range=1.0, win_size=3)

                    # LPIPS
                    out_img = outputs[i].unsqueeze(0)
                    tgt_img = clean[i].unsqueeze(0)

                    if out_img.shape[1] == 1:
                        out_img = out_img.repeat(1, 3, 1, 1)
                        tgt_img = tgt_img.repeat(1, 3, 1, 1)

                    out_img = (out_img * 2) - 1
                    tgt_img = (tgt_img * 2) - 1

                    lpips_val = loss_fn_lpips(out_img.to(device), tgt_img.to(device)).item()

                    psnr_list.append(psnr_val)
                    ssim_list.append(ssim_val)
                    lpips_list.append(lpips_val)

                    writer.writerow([count + 1, f"{psnr_val:.4f}", f"{ssim_val:.4f}", f"{lpips_val:.4f}"])
                    count += 1

        writer.writerow([])
        writer.writerow([
            "Average",
            f"{np.mean(psnr_list):.4f}",
            f"{np.mean(ssim_list):.4f}",
            f"{np.mean(lpips_list):.4f}"
        ])
        writer.writerow([
            "Std Dev",
            f"{np.std(psnr_list):.4f}",
            f"{np.std(ssim_list):.4f}",
            f"{np.std(lpips_list):.4f}"
        ])

    print(f"Saved PSNR/SSIM/LPIPS metrics for {count} images to '{csv_path}'")

# === Main ===
if __name__ == "__main__":
    local_rank = setup_ddp()
    loss_fn_lpips = lpips.LPIPS(net='alex').to(torch.device(f"cuda:{local_rank}"))
    model = UNet(input_channels=1, Nc=64)
    train_loader, val_loader, test_loader = prepare_dataloaders(noisy_images, InputImages, local_rank)
    train_model_ddp(model, train_loader, val_loader, test_loader, local_rank, epochs=50)

    if local_rank == 0:
        model.load_state_dict(torch.load("best_model_unet_DDP_10.pth"))
        test_psnr, test_ssim, test_lpips = test_model(model, test_loader, device=torch.device("cuda"))
        print(f"Final Test PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f} | LPIPS: {test_lpips:.4f}")

        # Export all per-image metrics to CSV
        save_metrics_csv_from_model(model, test_loader, device="cuda", csv_path="unet_DDP_10.csv")
    dist.destroy_process_group()