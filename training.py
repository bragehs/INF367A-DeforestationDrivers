from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from typing import Tuple, Optional, List
import random
import albumentations as A

class SatelliteSegmentationDataset(Dataset):
    """
    Dataset class for satellite image segmentation with smart patch extraction
    and augmentation strategies.
    """

    def __init__(
        self,
        images: np.ndarray,  # Shape: (N, C, H, W)
        labels: np.ndarray,  # Shape: (N, H, W)
        patch_size: int = 256,
        patch_stride: Optional[int] = 128,
        min_valid_pixels: float = 0.05,
        augment: bool = True,
        max_patches_per_image: Optional[int] = None,
    ):
        self.images = images
        self.labels = labels
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.min_valid_pixels = min_valid_pixels
        self.augment = augment
        self.max_patches_per_image = max_patches_per_image

        # Initialize augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                ], p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ])
        self.patches = self._create_patches()

    def _create_patches(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create patches from images with smart extraction strategy.
        Returns list of (image_patch, label_patch) tuples.
        """
        patches = []
        N, C, H, W = self.images.shape

        for img_idx in range(N):
            image = self.images[img_idx]  # (C, H, W)
            label = self.labels[img_idx]  # (H, W)

            # Track valid patches for this image
            image_patches = []

            for y in range(0, H - self.patch_size + 1, self.patch_stride):
                for x in range(0, W - self.patch_size + 1, self.patch_stride):
                    img_patch = image[
                        :, y : y + self.patch_size, x : x + self.patch_size
                    ]
                    label_patch = label[
                        y : y + self.patch_size, x : x + self.patch_size
                    ]

                    # Check if patch contains enough non-background pixels
                    if self._is_valid_patch(label_patch):
                        image_patches.append((img_patch, label_patch))

            # If max_patches_per_image is set, randomly sample patches
            if (
                self.max_patches_per_image
                and len(image_patches) > self.max_patches_per_image
            ):
                image_patches = random.sample(image_patches, self.max_patches_per_image)

            patches.extend(image_patches)

        return patches

    def _is_valid_patch(self, label_patch: np.ndarray) -> bool:
        """
        Check if patch contains enough non-background pixels.
        """
        non_background = (label_patch > 0).sum()
        total_pixels = label_patch.size
        return (non_background / total_pixels) >= self.min_valid_pixels

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_patch, label_patch = self.patches[idx]

        if self.augment:
            # Transpose image to (H, W, C) for albumentation
            img_patch = np.transpose(img_patch, (1, 2, 0))
            transformed = self.transform(image=img_patch, mask=label_patch)
            img_patch = np.transpose(transformed["image"], (2, 0, 1))
            label_patch = transformed["mask"]

        return (
            torch.as_tensor(img_patch, dtype=torch.float32),
            torch.as_tensor(label_patch, dtype=torch.long),
        )



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

    def create_dataloaders(self, val_size=0.15):
        # First split into train and temp (val + test)
        X_train, X_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=val_size, random_state=42)
        train_dataset = SatelliteSegmentationDataset(X_train, y_train)
        val_dataset = SatelliteSegmentationDataset(X_val, y_val)
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