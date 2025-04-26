# sam_finetune_utils.py
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from contextlib import nullcontext
from typing import Tuple, List, Dict, Optional, Any

# Assuming sam2 library is installed and importable
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Assuming albumentations is installed
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define class names and colors (optional, but good for visualization)
CLASS_NAMES = ["background", "plantation", "grassland_shrubland", "mining", "logging"]
CLASS_COLORS = plt.cm.viridis  # Using a colormap for simplicity

# --- Data Loading and Preparation ---


def load_sam_predictor(
    model_cfg: str, checkpoint_path: str, device: str
) -> SAM2ImagePredictor:
    """Loads the SAM2 model and creates a predictor."""
    print(f"Loading SAM2 model...")
    print(f"  - Config: {model_cfg}")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Device: {device}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    # Ensure the config path is correct relative to the SAM2 library or provide absolute path
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 predictor loaded successfully.")
    return predictor


def prepare_prompts_pos_neg(
    mask: np.ndarray,
    num_points_per_component: int = 1,
    num_negative_points: int = 5,  # Total negative points per image
    min_area: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate positive points per component and negative points from background.
    Returns points (N, 2) in (x, y) format and labels (N,) [0 or 1].
    """
    points = []
    labels = []
    h, w = mask.shape

    # --- Sample Positive Points (from foreground components) ---
    try:
        num_classes = len(CLASS_NAMES)
    except NameError:
        print("Warning: CLASS_NAMES not found globally in utils, assuming 5 classes.")
        num_classes = 5  # Default
    for class_id in range(1, num_classes):  # Iterate foreground classes
        class_mask = (mask == class_id).astype(np.uint8)
        if np.any(class_mask):
            num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            for k in range(1, num_labels):  # Skip background label 0
                component_area = stats[k, cv2.CC_STAT_AREA]
                if component_area < min_area:
                    continue  # Skip small components

                coords = np.argwhere(labels_map == k)  # Get (y, x)
                num_available = len(coords)
                num_to_sample = min(num_points_per_component, num_available)

                if num_to_sample > 0:
                    sampled_indices = np.random.choice(
                        num_available, size=num_to_sample, replace=False
                    )
                    for idx in sampled_indices:
                        y, x = coords[idx]
                        points.append([x, y])  # Add (x, y) point
                        labels.append(1)  # Add positive label

    # --- Sample Negative Points (from background area) ---
    if num_negative_points > 0:
        # Find all background coordinates (where mask is 0)
        bg_coords = np.argwhere(mask == 0)  # Shape [N_bg, 2], format (y, x)
        num_bg_available = len(bg_coords)

        # Sample specified number of negative points, ensuring not to exceed available points
        num_neg_to_sample = min(num_negative_points, num_bg_available)

        if num_neg_to_sample > 0:
            sampled_bg_indices = np.random.choice(
                num_bg_available, size=num_neg_to_sample, replace=False
            )
            for idx in sampled_bg_indices:
                y, x = bg_coords[idx]
                points.append([x, y])  # Add (x, y) point
                labels.append(0)  # Add negative label (0)

    # Handle case where no points (positive or negative) were sampled
    if not points:
        # Return empty arrays - the training loop should handle skipping these
        return np.array([]), np.array([])

    return np.array(points), np.array(labels)


class DeforestationDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Deforestation Segmentation"""

    def __init__(
        self,
        images: np.ndarray,  # Input argument
        masks: np.ndarray,  # Input argument
        num_points_per_component: int = 1,
        num_negative_points: int = 5,
        min_component_area: int = 10,
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()  # Good practice to call parent __init__

        # --- ADD THESE TWO LINES ---
        self.images = images
        self.masks = masks
        # --- END ADD ---

        # Store other parameters
        self.num_points_per_component = num_points_per_component
        self.num_negative_points = num_negative_points
        self.min_component_area = min_component_area
        self.transform = transform

        # Updated print statement for confirmation
        print(
            f"Dataset initialized with {len(self.images)} samples."
        )  # Use self.images now
        print(
            f"  Points per component: {self.num_points_per_component}, Neg points: {self.num_negative_points}, Min Area: {self.min_component_area}"
        )

    # --- __len__ and __getitem__ should now work ---
    def __len__(self):
        return len(self.images)  # self.images now exists

    def __getitem__(self, idx):
        # Access data using self attributes
        image = self.images[idx]  # Shape (C, H, W), float assumed
        mask = self.masks[idx]  # Shape (H, W), int assumed

        # --- START FULL IMAGE PREP for Albumentations ---
        # 1. Prepare image format (needs HWC, RGB, uint8 for many transforms)
        img_for_aug = image.copy()

        # Assuming input is CHW, convert to HWC
        if (
            img_for_aug.ndim == 3 and img_for_aug.shape[0] == 3
        ):  # Check for 3 channels first
            # Convert BGR (common channel order) to RGB
            img_for_aug = img_for_aug[[2, 1, 0], :, :]  # CHW BGR -> CHW RGB
            img_for_aug = np.transpose(img_for_aug, (1, 2, 0))  # CHW RGB -> HWC RGB
        elif (
            img_for_aug.ndim == 3 and img_for_aug.shape[0] < img_for_aug.shape[2]
        ):  # Guess CHW if not 3 channels first
            print(
                f"Warning: Image shape {img_for_aug.shape} might be CHW but not 3 channels. Transposing."
            )
            img_for_aug = np.transpose(img_for_aug, (1, 2, 0))
        # Add checks here if your format might be different (e.g., already HWC)

        # 2. Convert dtype to uint8
        if img_for_aug.dtype != np.uint8:
            # Check if image data is normalized (0.0 to 1.0 range)
            if img_for_aug.max() <= 1.0 and img_for_aug.min() >= 0.0:
                img_for_aug = (img_for_aug * 255).astype(np.uint8)
            else:
                # If not normalized 0-1, assume it's already scaled close to 0-255
                # Clip just in case to be safe before converting type
                img_for_aug = np.clip(img_for_aug, 0, 255).astype(np.uint8)
        # Now img_for_aug should be HWC, RGB, uint8
        # --- END FULL IMAGE PREP ---

        # 3. Apply Albumentations transforms
        if self.transform:
            try:
                # Albumentations takes HWC image, HW mask
                augmented = self.transform(image=img_for_aug, mask=mask)
                processed_image = augmented["image"]  # Output is HWC uint8
                processed_mask = augmented["mask"]  # Output is HW int
            except Exception as e:
                print(f"Error during augmentation for index {idx}: {e}")
                # Fallback to original prepped image/mask if augmentation fails
                processed_image = img_for_aug
                processed_mask = mask
        else:
            # If no transform, use the prepped image/mask
            processed_image = img_for_aug
            processed_mask = mask

        # 4. Generate point prompts using the potentially augmented mask
        points, labels = prepare_prompts_pos_neg(  # Use the function with neg prompts
            processed_mask,
            self.num_points_per_component,
            self.num_negative_points,
            self.min_component_area,
        )

        # 5. Prepare GT mask for loss calculation (Binary Float Tensor [H, W])
        # Use the potentially augmented mask
        gt_mask_tensor = torch.tensor(processed_mask > 0, dtype=torch.float32)

        # Return dictionary. Note: 'image' is HWC uint8 np.array for predictor.set_image
        return {
            "image": processed_image,
            "points": torch.tensor(points, dtype=torch.float32),
            "labels": torch.tensor(
                labels, dtype=torch.long
            ),  # Labels are indices/classes -> long
            "gt_mask": gt_mask_tensor,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable numbers of points.
    Keeps images as a list, stacks GT masks, puts points/labels into lists.
    """
    images = [item["image"] for item in batch]
    points = [item["points"] for item in batch]
    labels = [item["labels"] for item in batch]
    # Stack GT masks if they have the same size (should after dataset processing)
    gt_masks = torch.stack([item["gt_mask"] for item in batch], dim=0)

    return {
        "images": images,  # List of HWC uint8 np arrays
        "points": points,  # List of Tensors [num_points, 2]
        "labels": labels,  # List of Tensors [num_points]
        "gt_masks": gt_masks,  # Tensor [B, H, W]
    }


# --- Training Loop ---


def train_sam2_decoder(
    predictor: SAM2ImagePredictor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: str = "sam_finetune_output",
    device: str = "cuda",
) -> Dict[str, List]:
    """Trains the SAM2 mask decoder using point prompts."""

    model = predictor.model
    model.to(device)  # Ensure model is on the correct device

    # Freeze parts of the model
    model.sam_mask_decoder.train(True)
    model.sam_prompt_encoder.train(False)  # Keep frozen
    model.image_encoder.train(False)  # Keep frozen

    # Optimizer for the decoder only
    optimizer = torch.optim.AdamW(
        params=filter(
            lambda p: p.requires_grad, model.parameters()
        ),  # Only optimize decoder params
        lr=lr,
        weight_decay=weight_decay,
    )
    num_decoder_params = sum(
        p.numel() for p in model.sam_mask_decoder.parameters() if p.requires_grad
    )
    print(f"Optimizing {num_decoder_params:,} parameters in the mask decoder.")

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    print(
        f"Using device: {device}. Mixed precision {'enabled' if scaler else 'disabled'}."
    )

    os.makedirs(output_dir, exist_ok=True)
    best_val_iou = -1.0
    history = {"train_loss": [], "val_iou": []}

    print(f"\n===== Starting Decoder Fine-tuning =====")
    print(f"Epochs: {epochs}, LR: {lr}, Batch Size: {train_loader.batch_size}")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()  # Set decoder to train mode (affects dropout, batchnorm etc if decoder uses them)
        model.sam_mask_decoder.train(True)  # Explicitly ensure decoder part is training
        epoch_train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in train_pbar:
            images_list_np = batch["images"]  # List of HWC uint8 numpy arrays
            points_list_t = batch["points"]  # List of Tensors [N_pts, 2]
            labels_list_t = batch["labels"]  # List of Tensors [N_pts]
            gt_masks_t = batch["gt_masks"].to(
                device
            )  # Target masks Tensor [B, H, W], float

            # Filter out samples with no valid points BEFORE converting/processing
            valid_indices = [
                i for i, pts in enumerate(points_list_t) if pts.shape[0] > 0
            ]
            if not valid_indices:
                continue

            # Keep only valid items
            images_list_np = [images_list_np[i] for i in valid_indices]
            points_list_t = [points_list_t[i] for i in valid_indices]
            labels_list_t = [labels_list_t[i] for i in valid_indices]
            gt_masks_t = gt_masks_t[valid_indices]

            current_batch_size = len(images_list_np)
            if current_batch_size == 0:
                continue

            try:
                optimizer.zero_grad()

                # --- Step 1: Get Image Embeddings ---
                predictor.set_image_batch(images_list_np)
                image_embeddings = predictor._features[
                    "image_embed"
                ]  # Shape [B_orig, C, H, W]
                high_res_features_batch = predictor._features.get(
                    "high_res_feats", None
                )  # List of [B_orig, C', H', W'] or None

                # --- Initialize lists to collect losses ---
                batch_total_loss_list = []
                # Optional: Collect IoUs if needed for reporting average
                # batch_actual_ious_list = []

                # --- Step 2 & 3: Process Each Item Individually (Prompts -> Decoder -> Loss) ---
                for i in range(
                    current_batch_size
                ):  # Loop through items in the current filtered batch
                    img_idx = valid_indices[i]  # Original index for this item
                    current_points_t = points_list_t[i].to(device)
                    current_labels_t = labels_list_t[i].to(device)
                    current_gt_mask_t = gt_masks_t[
                        i : i + 1
                    ]  # Keep batch dim [1, H, W]

                    # Prepare prompts for this single item
                    mask_input_i, unnorm_coords_i, labels_i, unnorm_box_i = (
                        predictor._prep_prompts(
                            current_points_t.cpu().numpy(),
                            current_labels_t.cpu().numpy(),
                            box=None,
                            mask_logits=None,
                            normalize_coords=True,
                            img_idx=img_idx,
                        )
                    )

                    # Skip item if prompts are invalid
                    if unnorm_coords_i is None or labels_i is None:
                        # print(f"Warning: Skipping item {i} due to missing prompts.")
                        continue  # Skip this item

                    # --- Forward pass for item i under autocast ---
                    # Note: Autocast context wraps the operations for ONE item here
                    with torch.cuda.amp.autocast() if scaler is not None else nullcontext():
                        # Get prompt embeddings for item i
                        sparse_embed_i, dense_embed_i = model.sam_prompt_encoder(
                            points=(unnorm_coords_i, labels_i),
                            boxes=unnorm_box_i,
                            masks=mask_input_i,
                        )

                        # Slice image embedding for item i from the batch embedding
                        image_embed_i = image_embeddings[i : i + 1]

                        # Prepare high-res features for item i
                        active_high_res_feats_i = None
                        if high_res_features_batch:
                            try:
                                # Slice each level of high-res features for the i-th item
                                active_high_res_feats_i = [
                                    feat_level[i : i + 1]
                                    for feat_level in high_res_features_batch
                                ]
                            except Exception as e_hrf:
                                print(
                                    f"Warning: Could not prepare high_res_feats for item {i}: {e_hrf}. Passing None."
                                )

                        # >>> Call Mask Decoder for item i <<<
                        low_res_masks_i, iou_pred_i, _, _ = model.sam_mask_decoder(
                            image_embeddings=image_embed_i,
                            image_pe=model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embed_i,
                            dense_prompt_embeddings=dense_embed_i,
                            multimask_output=True,
                            repeat_image=False,
                            high_res_features=active_high_res_feats_i,
                        )

                        # Post-process (Upscale) for item i
                        orig_hw_i = predictor._orig_hw[img_idx]
                        upscaled_masks_logits_i = (
                            predictor._transforms.postprocess_masks(
                                low_res_masks_i, orig_hw_i
                            )
                        )  # Shape [1, num_masks, H_orig, W_orig]

                        # --- Calculate Loss for item i ---
                        prd_mask_logits_i = upscaled_masks_logits_i[
                            :, 0
                        ]  # Logits for first mask [1, H, W]
                        prd_mask_prob_i = torch.sigmoid(prd_mask_logits_i)

                        seg_loss_i = torch.nn.functional.binary_cross_entropy(
                            prd_mask_prob_i, current_gt_mask_t
                        )

                        prd_mask_binary_i = (prd_mask_prob_i > 0.5).float()
                        inter_i = (current_gt_mask_t * prd_mask_binary_i).sum()
                        union_i = (
                            current_gt_mask_t.sum() + prd_mask_binary_i.sum() - inter_i
                        )
                        actual_iou_i = inter_i / (union_i + 1e-6)  # Scalar IoU

                        predicted_iou_first_mask_i = iou_pred_i[:, 0]  # Shape [1]
                        # Ensure targets for MSELoss have same shape
                        score_loss_i = torch.nn.functional.mse_loss(
                            predicted_iou_first_mask_i, actual_iou_i.unsqueeze(0)
                        )

                        total_loss_i = (
                            seg_loss_i + score_loss_i * 20.0
                        )  # Loss for this item

                        # Store the loss for this item
                        batch_total_loss_list.append(total_loss_i)
                        # Optional: store IoU if needed for reporting
                        # batch_actual_ious_list.append(actual_iou_i)

                # --- After iterating through batch ---
                if (
                    not batch_total_loss_list
                ):  # Handle case where all items were skipped
                    # print("Warning: No valid losses computed for batch.")
                    continue  # Skip backward pass

                # --- Step 4: Calculate Average Batch Loss ---
                # Average the loss over the items that were successfully processed
                final_batch_loss = torch.mean(
                    torch.stack(batch_total_loss_list)
                )  # Average loss across batch items

                # --- Step 5: Backward Pass --- (Uses the averaged loss)
                if scaler is not None:
                    scaler.scale(final_batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    final_batch_loss.backward()  # Backpropagate the average loss
                    optimizer.step()

                epoch_train_losses.append(final_batch_loss.item())
                # avg_iou_in_batch = torch.mean(torch.stack(batch_actual_ious_list)).item() if batch_actual_ious_list else 0
                train_pbar.set_postfix(
                    {"Loss": final_batch_loss.item()}
                )  # , "Batch Avg IoU": avg_iou_in_batch})

            except Exception as e:
                print(f"\nError during training batch processing: {e}")
                import traceback

                traceback.print_exc()
                continue  # Skip to next batch

        avg_train_loss = (
            np.mean(epoch_train_losses) if epoch_train_losses else float("nan")
        )
        history["train_loss"].append(avg_train_loss)
        print(f"\nEpoch {epoch+1} Train Avg Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        # (Validation loop remains the same as it processes samples individually)
        model.eval()  # Set model to evaluation mode
        epoch_val_ious = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for batch in val_pbar:
                images_list_np = batch["images"]
                points_list_t = batch["points"]
                labels_list_t = batch["labels"]
                gt_masks_np = batch[
                    "gt_masks"
                ].numpy()  # Get GT masks as numpy for comparison

                # Filter out samples with no valid points
                valid_indices = [
                    i for i, pts in enumerate(points_list_t) if pts.shape[0] > 0
                ]
                if not valid_indices:
                    continue

                images_list_np = [images_list_np[i] for i in valid_indices]
                points_list_t = [points_list_t[i] for i in valid_indices]
                labels_list_t = [labels_list_t[i] for i in valid_indices]
                gt_masks_np = gt_masks_np[valid_indices]

                if len(images_list_np) == 0:
                    continue

                # Predict per image using the standard 'predict' method for validation ease
                for i in range(len(images_list_np)):
                    try:
                        current_image = images_list_np[i]  # HWC numpy
                        current_points = (
                            points_list_t[i].cpu().numpy()
                        )  # numpy [N_pts, 2]
                        current_labels = labels_list_t[i].cpu().numpy()  # numpy [N_pts]
                        current_gt_mask_binary = (
                            gt_masks_np[i] > 0
                        )  # Binary GT mask [H, W]

                        # Use predictor's standard predict method
                        predictor.set_image(current_image)  # Set image individually
                        masks, scores, logits = predictor.predict(
                            point_coords=current_points,
                            point_labels=current_labels,
                            multimask_output=True,
                        )
                        # masks shape: (num_masks, H, W), boolean numpy array

                        # Calculate IoU for the best mask output vs GT
                        best_iou_for_sample = 0.0
                        if masks is not None and len(masks) > 0:
                            for pred_mask in masks:  # Iterate through multimask outputs
                                pred_mask_binary = (
                                    pred_mask > 0
                                )  # Already boolean usually
                                inter = np.sum(
                                    current_gt_mask_binary & pred_mask_binary
                                )
                                union = np.sum(
                                    current_gt_mask_binary | pred_mask_binary
                                )
                                iou = inter / (union + 1e-6)
                                best_iou_for_sample = max(best_iou_for_sample, iou)

                        epoch_val_ious.append(best_iou_for_sample)

                    except Exception as e:
                        print(f"\nError during validation sample processing: {e}")
                        epoch_val_ious.append(0.0)  # Assign 0 IoU on error
                        continue

                val_pbar.set_postfix(
                    {"Avg IoU": np.mean(epoch_val_ious) if epoch_val_ious else 0.0}
                )

        avg_val_iou = np.mean(epoch_val_ious) if epoch_val_ious else float("nan")
        history["val_iou"].append(avg_val_iou)
        print(f"\nEpoch {epoch+1} Validation Avg IoU: {avg_val_iou:.4f}")

        # Save best model based on validation IoU
        if not np.isnan(avg_val_iou) and avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_path = os.path.join(
                output_dir, f"sam2_decoder_best_epoch{epoch+1}_iou{avg_val_iou:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model checkpoint to {save_path}")
        # Save latest model too
        latest_save_path = os.path.join(output_dir, "sam2_decoder_latest.pt")
        torch.save(model.state_dict(), latest_save_path)

    print(f"Training finished. Best Validation IoU: {best_val_iou:.4f}")
    # Save training history
    history_path = os.path.join(output_dir, "training_history.npy")
    np.save(history_path, history)
    print(f"Training history saved to {history_path}")

    # Plot history (same as before)
    # ... [Plotting code remains the same] ...
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # Ensure losses are numeric, filter out NaNs if any occurred
        valid_losses = [loss for loss in history["train_loss"] if not np.isnan(loss)]
        if valid_losses:
            plt.plot(valid_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss")

        plt.subplot(1, 2, 2)
        valid_ious = [iou for iou in history["val_iou"] if not np.isnan(iou)]
        if valid_ious:
            plt.plot(valid_ious, label="Validation IoU")
        plt.xlabel("Epoch")
        plt.ylabel("Mean IoU")
        plt.legend()
        plt.title("Validation IoU")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_plots.png")
        plt.savefig(plot_path)
        print(f"Training plots saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not generate plots: {e}")

    return history


# --- Visualization ---


def visualize_predictions(
    predictor: SAM2ImagePredictor,
    dataset: torch.utils.data.Dataset,  # Use dataset to get samples
    num_samples: int = 5,
    output_dir: str = "sam_finetune_output",
    checkpoint_path: Optional[str] = None,  # Path to load specific weights
):
    """Visualizes model predictions on dataset samples."""
    if checkpoint_path:
        print(f"Loading model weights from: {checkpoint_path}")
        predictor.model.load_state_dict(torch.load(checkpoint_path))
        print("Weights loaded.")
    predictor.model.eval()  # Set to evaluation mode

    os.makedirs(output_dir, exist_ok=True)
    indices = np.random.choice(
        len(dataset), size=min(num_samples, len(dataset)), replace=False
    )

    plt.figure(figsize=(15, 5 * len(indices)))
    plot_idx = 1

    for i, idx in enumerate(indices):
        sample = dataset[idx]  # Get processed sample
        image_np = sample["image"]  # HWC uint8
        points = sample["points"].numpy()
        labels = sample["labels"].numpy()
        gt_mask_tensor = sample["gt_mask"]  # Binary float tensor [H, W]
        gt_mask_np = gt_mask_tensor.numpy().astype(bool)  # Boolean numpy [H, W]

        if points.shape[0] == 0:
            print(f"Skipping visualization for sample {idx} due to no prompt points.")
            continue

        # Get model prediction
        try:
            predictor.set_image(image_np)
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,  # Usually True by default
            )
            # masks are boolean numpy arrays [num_masks, H, W]
            # scores are numpy array [num_masks]
        except Exception as e:
            print(f"Error predicting for sample {idx}: {e}")
            continue

        # --- Plotting ---
        # Original Image + Points
        plt.subplot(len(indices), 3, plot_idx)
        plt.imshow(image_np)
        # Plot points used for prediction
        for pt, lbl in zip(points, labels):
            color = "green" if lbl == 1 else "red"  # Green for foreground
            plt.scatter(
                pt[0],
                pt[1],
                color=color,
                marker="*",
                s=100,
                edgecolor="white",
                linewidth=1.25,
            )
        plt.title(f"Sample {idx} - Image + Prompts")
        plt.axis("off")
        plot_idx += 1

        # Ground Truth Mask
        plt.subplot(len(indices), 3, plot_idx)
        plt.imshow(image_np)
        plt.imshow(gt_mask_np, cmap="viridis", alpha=0.6)  # Show GT mask overlay
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plot_idx += 1

        # Predicted Mask (showing best one based on score)
        plt.subplot(len(indices), 3, plot_idx)
        plt.imshow(image_np)
        if masks is not None and len(masks) > 0:
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            plt.imshow(
                best_mask, cmap="inferno", alpha=0.6
            )  # Show predicted mask overlay
            plt.title(f"Prediction (Best Score: {best_score:.3f})")
        else:
            plt.title("Prediction (No mask output)")
        plt.axis("off")
        plot_idx += 1

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "prediction_visualization.png")
    plt.savefig(viz_path)
    print(f"Prediction visualizations saved to {viz_path}")
    plt.show()
    plt.close()
