from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

class WeightedIoULoss(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(WeightedIoULoss, self).__init__()
        self.num_classes = num_classes
        self.weights = weights  # Class weights (e.g., inversely proportional to class frequency)

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) class indices
        pred_probs = F.softmax(pred, dim=1)  # Convert logits to probabilities
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))  # (B, C)
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection  # (B, C)

        iou = (intersection + 1e-7) / (union + 1e-7)  # (B, C)
        if self.weights is not None:
            iou = iou * self.weights  # Apply class weights

        iou_loss = 1 - iou.mean()  # Average over classes and batch
        return iou_loss
    

class LogCoshDiceLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, epsilon=1e-6):
        super(LogCoshDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.class_weights = class_weights  # Tensor of shape (num_classes,)

    def forward(self, pred, target):
        pred_probs = F.softmax(pred, dim=1)
        batch_size = pred.shape[0]

        dice_losses = []
        for cls in range(self.num_classes):
            pred_cls = pred_probs[:, cls]
            target_cls = (target == cls).float()

            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            cardinality = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))

            dice = (2. * intersection + self.epsilon) / (cardinality + self.epsilon)
            loss_cls = torch.log(torch.cosh(1. - dice))  # (B,)

            if self.class_weights is not None:
                loss_cls *= self.class_weights[cls]

            dice_losses.append(loss_cls)

        # Average over classes and batch
        loss = torch.mean(torch.stack(dice_losses, dim=1))  # (B, C) -> scalar
        return loss  

class Sentinel2SegmentationDataset(Dataset):
    def __init__(self, images, labels, patch_size=128, stride=128, transform=None):
        """
        images: Tensor or numpy array of shape (N, 12, H, W)
        labels: Tensor or numpy array of shape (N, H, W)
        patch_size: Size of the patches (default 256)
        stride: Stride for patching (default 128 for overlapping)
        transform: Optional image transformations
        """
        self.images = images
        self.labels = labels
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = []  # Store (image_patch, label_patch) pairs

        self.create_patches()

    def create_patches(self):
        """Extracts patches from the dataset."""
        N, C, H, W = self.images.shape

        for i in range(N):
            img = self.images[i]  # Shape (12, H, W)
            lbl = self.labels[i]  # Shape (H, W)

            # Divide the 1024 image dimension into 512x512 patches with no overlap
            for y in range(0, H, self.patch_size):
                for x in range(0, W, self.patch_size):
                    img_patch = img[:, y:y+self.patch_size, x:x+self.patch_size]
                    lbl_patch = lbl[y:y+self.patch_size, x:x+self.patch_size]
                    self.patches.append((img_patch, lbl_patch))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch, lbl_patch = self.patches[idx]

        # Apply transformations if any
        if self.transform:
            img_patch = self.transform(img_patch)

        return torch.as_tensor(img_patch, dtype=torch.float32), torch.as_tensor(lbl_patch, dtype=torch.long)


class RunTraining:
    def __init__(self, model, data,labels, loss_function, optimizer, scheduler, epochs, patch_size=128, stride=128, batch_size=32):
        self.model = model
        self.data = data
        self.labels = labels
        self.epochs = epochs
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.stride = stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    
    def normalize_10000(self, x):
        '''normalize all bands equally to 0-1 range'''
        return x.astype('float32') / 10000
    
    def satlas_normalization(self, tci_rgb, bands):
        '''normalize bands and tci_rgb separately, as specified for satlas model'''
        tci_norm = (tci_rgb - tci_rgb.min()) / (tci_rgb.max() - tci_rgb.min())

        # bands: list of np arrays [B05, B06, B07, B08, B11, B12], each 16-bit
        normalized_bands = []
        for band in bands:
            norm_band = np.clip(band / 8160.0, 0, 1)
            normalized_bands.append(norm_band)

        # Stack into final tensor
        normalized_bands = np.array(normalized_bands)
        print(normalized_bands.shape, tci_norm.shape)
        return np.concatenate([tci_norm, normalized_bands], axis=1)
    

    def create_dataloaders(self, val_size=0.15):
        # First split into train and temp (val + test)
        X_train, X_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=val_size, random_state=42)
        train_dataset = Sentinel2SegmentationDataset(X_train, y_train)
        val_dataset = Sentinel2SegmentationDataset(X_val, y_val)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return train_dataloader, val_dataloader
    
    def compute_class_weights(self, all_labels):
        classes = np.unique(all_labels)
        weights = compute_class_weight('balanced', classes=classes, y=all_labels)
        print("Class weights:", weights, "for classes:", classes)
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Get main output if model returns multiple outputs

                # Ensure correct shape before loss calculation
                if outputs.shape[0] != targets.shape[0]:
                    outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def train(self, model, num_epochs, filename,
                      num_warm_up= 5):
        self.model = model.to(self.device)
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.loss_function
        train_loader, val_loader = self.create_dataloaders()
        save_path = os.getcwd() + '/weights/' 

        # Warmup phase
        model.train()
        print("Starting warmup phase...")
        for epoch in range(num_warm_up):  # epochs of warm up:
            with tqdm(train_loader, desc=f'Warmup Epoch {epoch+1}/5') as pbar:
                for batch_idx, (images, targets) in enumerate(pbar):
                    images, targets = images.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Get main output if model returns multiple outputs
                    # Ensure correct shape before loss calculation
                    if outputs.shape[0] != targets.shape[0]:
                        outputs = outputs.permute(0, 2, 1)

                    loss = criterion(outputs, targets) / 4  # Accumulate gradients over 4 steps
                    loss.backward()

                    if (batch_idx + 1) % 4 == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    pbar.set_postfix({'loss': loss.item() * 4})  # Show accumulated loss
        optimizer.zero_grad()
        print("Starting main training phase...")
        # Main training phase
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            # Training loop with progress bar
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch_idx, (images, targets) in enumerate(pbar):
                    images, targets = images.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Get main output if model returns multiple outputs


                    # Ensure correct shape before loss calculation
                    if outputs.shape[0] != targets.shape[0]:
                        outputs = outputs.permute(0, 2, 1)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix({'train_loss': loss.item()})

            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            val_loss = self.validate(model, val_loader, criterion)
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path + filename)

            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)