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

# === Data Loading ===
data_path = "/home/sulaimon/EXPERIMENT/23/images_001"
image_files = glob.glob(os.path.join(data_path, '*.png'))

def load_xray_image(data_path):
    img = Image.open(data_path).convert('L').resize((256, 256))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

InputImages = torch.tensor(np.array([load_xray_image(f) for f in image_files]), dtype=torch.float32)

def add_gaussian_noise(images, mean=0.1, stddev=0.1):
    noise = torch.normal(mean, std=stddev, size=images.shape)
    return torch.clamp(images + noise, 0., 1.)

noisy_images = add_gaussian_noise(InputImages)

class X_rayDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

# === UNet++ Model Definition===
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNetplusplus(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_ch=64):
        super().__init__()
        Nc = [base_ch * (2 ** i) for i in range(5)]
        self.pool = nn.MaxPool2d(2,2)

        self.conv0_0 = ConvBlock(input_channels, Nc[0])
        self.conv1_0 = ConvBlock(Nc[0], Nc[1])
        self.conv2_0 = ConvBlock(Nc[1], Nc[2])
        self.conv3_0 = ConvBlock(Nc[2], Nc[3])
        self.conv4_0 = ConvBlock(Nc[3], Nc[4])

        self.conv0_1 = ConvBlock(Nc[0]+Nc[1], Nc[0])
        self.conv1_1 = ConvBlock(Nc[1]+Nc[2], Nc[1])
        self.conv2_1 = ConvBlock(Nc[2]+Nc[3], Nc[2])
        self.conv3_1 = ConvBlock(Nc[3]+Nc[4], Nc[3])

        self.conv0_2 = ConvBlock(Nc[0]*2+Nc[1], Nc[0])
        self.conv1_2 = ConvBlock(Nc[1]*2+Nc[2], Nc[1])
        self.conv2_2 = ConvBlock(Nc[2]*2+Nc[3], Nc[2])

        self.conv0_3 = ConvBlock(Nc[0]*3+Nc[1], Nc[0])
        self.conv1_3 = ConvBlock(Nc[1]*3+Nc[2], Nc[1])

        self.conv0_4 = ConvBlock(Nc[0]*4+Nc[1], Nc[0])
        self.final = nn.Conv2d(Nc[0], output_channels, 1)

    def forward(self, x):
        X0_0 = self.conv0_0(x)
        X1_0 = self.conv1_0(self.pool(X0_0))
        X2_0 = self.conv2_0(self.pool(X1_0))
        X3_0 = self.conv3_0(self.pool(X2_0))
        X4_0 = self.conv4_0(self.pool(X3_0))

        X0_1 = self.conv0_1(torch.cat([X0_0, F.interpolate(X1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        X1_1 = self.conv1_1(torch.cat([X1_0, F.interpolate(X2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        X2_1 = self.conv2_1(torch.cat([X2_0, F.interpolate(X3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        X3_1 = self.conv3_1(torch.cat([X3_0, F.interpolate(X4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        X0_2 = self.conv0_2(torch.cat([X0_0, X0_1, F.interpolate(X1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        X1_2 = self.conv1_2(torch.cat([X1_0, X1_1, F.interpolate(X2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        X2_2 = self.conv2_2(torch.cat([X2_0, X2_1, F.interpolate(X3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        X0_3 = self.conv0_3(torch.cat([X0_0, X0_1, X0_2, F.interpolate(X1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        X1_3 = self.conv1_3(torch.cat([X1_0, X1_1, X1_2, F.interpolate(X2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        X0_4 = self.conv0_4(torch.cat([X0_0, X0_1, X0_2, X0_3, F.interpolate(X1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        return {"final": self.final(X0_4)}

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

def test_model(model, test_loader, device):
    model.eval()
    model.to(device)

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)["final"]

            for i in range(noisy.size(0)):
                y_true = clean[i, 0].cpu().numpy()
                y_pred = outputs[i, 0].cpu().numpy()

                total_psnr += psnr_metric(torch.tensor(y_true), torch.tensor(y_pred))
                total_ssim += ssim(y_true, y_pred, data_range=1.0)

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
def train_model_ddp(model, train_loader, val_loader, test_loader, local_rank, epochs=50, save_path="best_model_unetpp_DDP_10.pth"):
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
                outputs = model(noisy)["final"]
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
                    outputs = model(noisy)["final"]
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
def save_metrics_csv_from_model(model, test_loader, device="cpu", csv_path="unetpp_DDP_10.csv"):
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

                outputs = model(noisy)["final"]

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

#======= Main =======
if __name__ == "__main__":
    local_rank = setup_ddp()
    loss_fn_lpips = lpips.LPIPS(net='alex').to(torch.device(f"cuda:{local_rank}"))
    model = UNetplusplus(input_channels=1, output_channels=1)
    train_loader, val_loader, test_loader = prepare_dataloaders(noisy_images, InputImages, local_rank)
    train_model_ddp(model, train_loader, val_loader, test_loader, local_rank, epochs=50)

    if local_rank == 0:
        model.load_state_dict(torch.load("best_model_unetpp_DDP_10.pth"))
        test_psnr, test_ssim, test_lpips = test_model(model, test_loader, device=torch.device("cuda"))
        print(f"Final Test PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f} | LPIPS: {test_lpips:.4f}")

        # Export all per-image metrics to CSV
        save_metrics_csv_from_model(model, test_loader, device="cuda", csv_path="unetpp_DDP_10.csv")
    dist.destroy_process_group()
