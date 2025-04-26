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
import torch
import torch.nn.functional as F

# Assuming sam2 library is installed and importable
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import remove_small_regions

# Assuming albumentations is installed
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define class names and colors (optional, but good for visualization)
CLASS_NAMES = ["background", "plantation", "grassland_shrubland", "mining", "logging"]
CLASS_COLORS = plt.cm.viridis  # Using a colormap for simplicity


# --- Loss Function ---
def dice_loss(inputs, targets, num_objects):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of shape [N, M, H, W] or [N, H, W].
                The predictions for each example. Assumes raw logits.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects represented in the inputs/targets (for averaging).
                     For per-instance loss, this will typically be 1.0.
    Returns:
        Dice loss tensor shape [N, M] or [N,].
    """
    inputs = inputs.sigmoid()  # Apply sigmoid to get probabilities
    if inputs.dim() == 4:  # Multi-mask case [N, M, H, W]
        # Flatten spatial dims H, W -> P = H * W
        inputs = inputs.flatten(2)  # Shape: [N, M, P]
        targets = targets.flatten(2)  # Shape: [N, M, P]
        numerator = 2 * (inputs * targets).sum(-1)  # Sum over P -> Shape [N, M]
        denominator = inputs.sum(-1) + targets.sum(-1)  # Shape [N, M]
    elif inputs.dim() == 3:  # Single mask case [N, H, W] - Treat as M=1
        inputs = inputs.flatten(1)  # Shape [N, P]
        targets = targets.flatten(1)  # Shape [N, P]
        numerator = 2 * (inputs * targets).sum(1)  # Sum over P -> Shape [N]
        denominator = inputs.sum(1) + targets.sum(1)  # Shape [N]
    else:
        raise ValueError(f"Unsupported input shape: {inputs.shape}")

    loss = 1 - (numerator + 1) / (denominator + 1)  # Add 1 for stability

    # Original SAM code divides by num_objects here when loss_on_multimask=True.
    # We'll perform averaging based on num_objects later if needed,
    # but return the per-mask-channel loss here.
    # If handling batch averaging later, ensure num_objects > 0.
    # return loss / max(num_objects, 1.0) # Let's return raw loss per mask channel first
    return loss


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of shape [N, M, H, W] or [N, H, W].
                The predictions for each example. Assumes raw logits.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects represented (for averaging, typically 1.0 here).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples. Default = 2.
    Returns:
        Focal loss tensor shape [N, M] or [N,]. Average loss per pixel, per mask channel.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Original SAM code averages over spatial dims here when loss_on_multimask=True.
    # We return the average loss per mask channel.
    if loss.dim() == 4:  # [N, M, H, W]
        # Average over H, W -> Shape [N, M]
        # return loss.flatten(2).mean(-1) / max(num_objects, 1.0)
        return loss.mean((2, 3))  # Average over H, W dimensions
    elif loss.dim() == 3:  # [N, H, W]
        # Average over H, W -> Shape [N]
        # return loss.flatten(1).mean(-1) / max(num_objects, 1.0)
        return loss.mean((1, 2))  # Average over H, W dimensions
    else:
        raise ValueError(f"Unsupported input shape: {loss.shape}")


def iou_loss(inputs, targets, pred_ious, num_objects, use_l1_loss=False):
    """
    Calculates the loss between predicted IoU scores and actual IoUs.
    Args:
        inputs: A float tensor of shape [N, M, H, W]. Logits.
        targets: A float tensor of shape [N, M, H, W]. Ground truth masks.
        pred_ious: A float tensor of shape [N, M] containing the predicted IoUs scores per mask.
        num_objects: Number of objects (for averaging, typically 1.0 here).
        use_l1_loss: Whether to use L1 loss instead of MSE loss.
    Returns:
        IoU loss tensor shape [N, M].
    """
    if not (inputs.dim() == 4 and targets.dim() == 4 and pred_ious.dim() == 2):
        raise ValueError(
            f"Dimension mismatch: inputs {inputs.shape}, targets {targets.shape}, pred_ious {pred_ious.shape}"
        )
    if not (inputs.shape[:2] == targets.shape[:2] == pred_ious.shape):
        raise ValueError(
            f"N, M dimension mismatch: inputs {inputs.shape}, targets {targets.shape}, pred_ious {pred_ious.shape}"
        )

    # Calculate actual IoU (using thresholded mask)
    pred_mask = (inputs.sigmoid() > 0.5).flatten(
        2
    )  # Sigmoid + Threshold + Flatten HW -> [N, M, P]
    gt_mask = targets.flatten(2) > 0.5  # Ensure binary + Flatten HW -> [N, M, P]

    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()  # Intersection area [N, M]
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()  # Union area [N, M]
    actual_ious = area_i / torch.clamp(area_u, min=1.0)  # Actual IoU [N, M]

    # Calculate loss between predicted IoU and actual IoU
    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")  # Shape [N, M]
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")  # Shape [N, M]

    # Original SAM divides by num_objects here when loss_on_multimask=True.
    # Return raw loss per mask channel.
    # return loss / max(num_objects, 1.0)
    return loss


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
    """Dataset returning images and original integer masks for per-instance training."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,  # Expect ORIGINAL integer masks here
        transform: Optional[A.Compose] = None,
        # <<< REMOVE point generation params from __init__ >>>
        # num_points_per_component: int = 1,
        # num_negative_points: int = 5,
        # min_component_area: int = 10,
    ):
        super().__init__()
        self.images = images
        # Ensure masks are integer type, not float/bool
        self.masks = masks.astype(np.int32)
        self.transform = transform
        print(
            f"Dataset initialized with {len(self.images)} samples for Per-Instance Training."
        )
        if len(self.images) > 0:
            print(f"  Image shape example: {self.images[0].shape}")
            print(
                f"  Mask shape example: {self.masks[0].shape}, Mask dtype: {self.masks[0].dtype}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape (C, H, W), float assumed
        original_mask = self.masks[idx]  # Original HW int mask

        # --- Image Prep (HWC, RGB, uint8) ---
        img_for_aug = image.copy()
        # (Your finalized image prep logic here)
        if img_for_aug.ndim == 3 and img_for_aug.shape[0] == 3:
            img_for_aug = img_for_aug[[2, 1, 0], :, :]
            img_for_aug = np.transpose(img_for_aug, (1, 2, 0))
        elif img_for_aug.ndim == 3 and img_for_aug.shape[0] < img_for_aug.shape[2]:
            img_for_aug = np.transpose(img_for_aug, (1, 2, 0))
        # Ensure uint8
        if img_for_aug.dtype != np.uint8:
            if img_for_aug.max() <= 1.0 and img_for_aug.min() >= 0.0:
                img_for_aug = (img_for_aug * 255).astype(np.uint8)
            else:
                img_for_aug = np.clip(img_for_aug, 0, 255).astype(np.uint8)

        # --- Apply Augmentations ---
        # Apply to image AND the original integer mask
        if self.transform:
            try:
                # Ensure mask interpolation uses cv2.INTER_NEAREST if resizing occurs
                augmented = self.transform(image=img_for_aug, mask=original_mask)
                processed_image = augmented["image"]
                processed_mask_int = augmented["mask"].astype(
                    np.int32
                )  # Ensure output is int
            except Exception as e:
                print(f"Error during augmentation for index {idx}: {e}")
                processed_image = img_for_aug
                processed_mask_int = original_mask
        else:
            processed_image = img_for_aug
            processed_mask_int = original_mask

        # <<< REMOVE prompt generation call from here >>>
        # points, labels = prepare_prompts_pos_neg(...) # No longer needed here

        # <<< REMOVE binary gt_mask_tensor generation from here (done in loop) >>>
        # gt_mask_tensor = torch.tensor(processed_mask_int > 0, dtype=torch.float32)

        # Return image and the potentially augmented ORIGINAL integer mask
        return {
            "image": processed_image,  # HWC uint8 numpy
            "original_mask": processed_mask_int,  # HW int numpy
            # Add original index if needed for debugging complex issues later
            # "original_index": idx
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function updated for per-instance training dataset output."""
    images = [item["image"] for item in batch]
    original_masks = [item["original_mask"] for item in batch]  # List of HW int numpy

    # No points/labels/gt_mask needed from collate_fn for the new training loop
    return {
        "images": images,  # List
        "original_masks": original_masks,  # List
    }


# --- Training Loop ---


def train_sam2_decoder_per_instance(
    predictor: SAM2ImagePredictor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_points_per_component: int = 1,
    num_negative_points: int = 5,
    min_component_area: int = 10,
    # --- Parameters for SAM Loss ---
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    iou_loss_weight: float = 1.0,  # Weight for IoU score prediction loss
    dice_loss_weight: float = 1.0,  # Weight for Dice loss
    focal_loss_weight: float = 20.0,  # Weight for Focal loss (often higher, e.g., 20x)
    iou_use_l1_loss: bool = False,  # Use L1 for IoU loss instead of MSE
    supervise_all_iou: bool = False,  # If True, calculate IoU loss for all masks, not just best
    # --- End Parameters for SAM Loss ---
    output_dir: str = "sam_finetune_output",
    device: str = "cuda",
) -> Dict[str, List]:
    """
    Trains the SAM2 mask decoder using per-instance point prompts and
    a combination of Focal, Dice, and IoU loss.
    """

    model = predictor.model
    model.to(device)
    # Setup optimizer, scaler, history, output dir (same as before)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    history = {"train_loss": [], "val_iou": []}
    best_val_iou = -1.0
    os.makedirs(output_dir, exist_ok=True)

    # Ensure CLASS_NAMES is defined globally or passed as an argument if needed
    try:
        num_total_classes = len(CLASS_NAMES)  # Including background
    except NameError:
        print(
            "Warning: CLASS_NAMES not found globally, assuming 5 classes + background."
        )
        num_total_classes = 6  # Adjust if needed

    num_decoder_params = sum(
        p.numel() for p in model.sam_mask_decoder.parameters() if p.requires_grad
    )
    print(f"Optimizing {num_decoder_params:,} parameters in the mask decoder.")
    print(
        f"Using device: {device}. Mixed precision {'enabled' if scaler else 'disabled'}."
    )
    print(f"\n===== Starting Per-Instance Decoder Fine-tuning =====")
    print(f"Epochs: {epochs}, LR: {lr}, Batch Size: {train_loader.batch_size}")
    print(
        f"Points per component: {num_points_per_component}, Neg points: {num_negative_points}, Min Area: {min_component_area}"
    )

    # Store effective weights used
    loss_weights = {
        "loss_mask": focal_loss_weight,  # Corresponds to focal loss weight
        "loss_dice": dice_loss_weight,
        "loss_iou": iou_loss_weight,
    }
    print(f"Using SAM Loss Weights: {loss_weights}")
    print(f"  Focal alpha: {focal_alpha}, Focal gamma: {focal_gamma}")
    print(f"  IoU Loss uses {'L1' if iou_use_l1_loss else 'MSE'}")
    print(f"  Supervising IoU for {'ALL' if supervise_all_iou else 'BEST'} mask(s)")

    for epoch in range(epochs):
        model.train()
        # Ensure the decoder specifically is in train mode if other parts are frozen
        model.sam_mask_decoder.train(True)
        epoch_batch_losses = []  # Collect average loss for each batch

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            images_list_np = batch["images"]  # List [B] of HWC uint8 numpy
            original_masks_list_np = batch["original_masks"]  # List [B] of HW int numpy

            current_batch_size = len(images_list_np)
            if current_batch_size == 0:
                continue

            optimizer.zero_grad()

            # --- Step 1: Get Image Embeddings (once per batch) ---
            try:
                predictor.set_image_batch(images_list_np)
                image_embeddings = predictor._features["image_embed"]
                high_res_features_batch = predictor._features.get(
                    "high_res_feats", None
                )
                batch_orig_hw = (
                    predictor._orig_hw
                )  # List of (H, W) tuples from predictor
            except Exception as e_embed:
                print(f"\nError setting image batch: {e_embed}")
                continue  # Skip batch

            # --- List to collect losses for ALL INSTANCES processed in this batch ---
            batch_instance_losses = []
            instances_processed_in_batch = 0

            # --- Step 2: Loop through each image 'i' in the batch ---
            for i in range(current_batch_size):
                image_embed_i = image_embeddings[
                    i : i + 1
                ]  # Keep batch dim: [1, C, H, W]
                original_mask_i = original_masks_list_np[i]  # Shape [H, W]
                orig_hw_i = batch_orig_hw[i]  # H, W for this image

                # Prepare high-res features for item i
                high_res_feats_i = None
                if high_res_features_batch:
                    try:
                        # Ensure high_res_feats_i maintains the batch dimension for the model
                        high_res_feats_i = [
                            feat_level[i : i + 1]
                            for feat_level in high_res_features_batch
                        ]
                    except Exception as e_hrf:
                        print(
                            f"\nWarning: Error extracting high-res features for image {i}: {e_hrf}"
                        )
                        pass  # Continue without them if extraction fails

                # --- Step 3: Find instances (components) in image 'i' ---
                num_classes = num_total_classes - 1  # Number of foreground classes

                for class_id in range(
                    1, num_total_classes
                ):  # Iterate foreground classes (1 to N)
                    class_mask_i = (original_mask_i == class_id).astype(np.uint8)
                    if not np.any(class_mask_i):
                        continue

                    # --- Step 4: Loop through instances 'k' of this class ---
                    num_labels_k, labels_map_k, stats_k, _ = (
                        cv2.connectedComponentsWithStats(class_mask_i, connectivity=8)
                    )

                    for k in range(1, num_labels_k):  # Skip background label 0
                        component_area_k = stats_k[k, cv2.CC_STAT_AREA]
                        if component_area_k < min_component_area:
                            continue

                        # --- Step 5: Prepare Prompts for specific Instance k ---
                        # (Erosion logic for positive point sampling as implemented before)
                        points_k_list = []
                        labels_k_list = []
                        instance_mask_k = (labels_map_k == k).astype(np.uint8)
                        erosion_kernel = np.ones((5, 5), np.uint8)
                        eroded_instance_mask_k = cv2.erode(
                            instance_mask_k, erosion_kernel, iterations=1
                        )
                        coords_k_eroded = np.argwhere(eroded_instance_mask_k > 0)

                        if len(coords_k_eroded) == 0:
                            coords_k = np.argwhere(labels_map_k == k)
                            # print(f"Warning: Erosion removed instance k={k} in image i={i}. Using original coords.") # Optional Warning
                            if len(coords_k) == 0:
                                continue
                        else:
                            coords_k = coords_k_eroded

                        num_pos_available = len(coords_k)
                        num_pos_to_sample = min(
                            num_points_per_component, num_pos_available
                        )
                        if num_pos_to_sample <= 0:
                            continue

                        sampled_indices_k = np.random.choice(
                            num_pos_available, size=num_pos_to_sample, replace=False
                        )
                        for idx_k in sampled_indices_k:
                            y, x = coords_k[idx_k]
                            points_k_list.append([x, y])
                            labels_k_list.append(1)

                        # Negative points sampling
                        bg_coords_i = np.argwhere(original_mask_i == 0)
                        num_neg_available = len(bg_coords_i)
                        num_neg_to_sample = min(num_negative_points, num_neg_available)
                        if num_neg_to_sample > 0:
                            sampled_indices_neg = np.random.choice(
                                num_neg_available, size=num_neg_to_sample, replace=False
                            )
                            for idx_neg in sampled_indices_neg:
                                y, x = bg_coords_i[idx_neg]
                                points_k_list.append([x, y])
                                labels_k_list.append(0)

                        points_k = np.array(points_k_list)
                        labels_k = np.array(labels_k_list)
                        if 1 not in labels_k_list:
                            continue  # Need at least one positive prompt

                        # --- Step 6: Prepare Target Mask for Instance k ---
                        # Ground truth mask for THIS instance k. Shape [H, W]
                        gt_instance_mask_k_np = labels_map_k == k
                        gt_instance_mask_k = torch.tensor(
                            gt_instance_mask_k_np, dtype=torch.float32, device=device
                        )  # Shape: [H, W]

                        # --- Step 7: Forward Pass & Loss for Instance k ---
                        try:
                            # Prepare prompts using predictor method
                            (
                                mask_input_k,
                                unnorm_coords_k,
                                labels_k_prep,
                                unnorm_box_k,
                            ) = predictor._prep_prompts(
                                points_k,
                                labels_k,
                                box=None,
                                mask_logits=None,
                                normalize_coords=True,  # Important for SAM
                                img_idx=i,  # Pass image index if needed by prep_prompts
                            )
                            # Skip if prompt preparation failed
                            if unnorm_coords_k is None or labels_k_prep is None:
                                # print(f"Skipping instance k={k} in image i={i}: Prompt prep failed.") # Optional debug
                                continue

                            # Forward pass under autocast
                            with torch.cuda.amp.autocast() if scaler is not None else nullcontext():
                                # Get sparse/dense embeddings from prompt encoder
                                sparse_embed_k, dense_embed_k = (
                                    model.sam_prompt_encoder(
                                        points=(unnorm_coords_k, labels_k_prep),
                                        boxes=unnorm_box_k,
                                        masks=mask_input_k,
                                    )
                                )

                                # Get Decoder Outputs
                                # Ensure multimask_output=True to get potentially multiple masks
                                low_res_masks_k, iou_pred_k, _, _ = (
                                    model.sam_mask_decoder(
                                        image_embeddings=image_embed_i,  # Use the embedding for THIS image
                                        image_pe=model.sam_prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=sparse_embed_k,
                                        dense_prompt_embeddings=dense_embed_k,
                                        multimask_output=True,
                                        repeat_image=False,  # We are processing image by image
                                        high_res_features=high_res_feats_i,  # Pass high-res features if available
                                    )
                                )  # low_res_masks_k shape [1, M, H_low, W_low], iou_pred_k shape [1, M]

                                # Upscale predicted masks to original size
                                # Output shape should be [1, M, H, W] (Logits)
                                upscaled_masks_logits_k = (
                                    predictor._transforms.postprocess_masks(
                                        low_res_masks_k, orig_hw_i
                                    )
                                )

                                # Predicted IoU scores
                                # Ensure shape is [1, M]
                                if iou_pred_k.dim() == 1:
                                    iou_pred_k = iou_pred_k.unsqueeze(
                                        0
                                    )  # Make it [1, M]

                                # --- Prepare inputs for loss functions ---
                                # Predicted mask logits: src_masks [1, M, H, W]
                                src_masks = upscaled_masks_logits_k
                                # Predicted iou scores: pred_ious [1, M]
                                pred_ious = iou_pred_k
                                # Ground truth mask: target [H, W] -> [1, 1, H, W] -> [1, M, H, W]
                                target_masks_prep = (
                                    gt_instance_mask_k.unsqueeze(0)
                                    .unsqueeze(1)
                                    .expand_as(src_masks)
                                )

                                num_masks_per_instance = src_masks.shape[1]  # M
                                num_instances_for_loss = 1.0  # Processing one instance

                                # --- Calculate losses per predicted mask channel ---
                                # Output shapes should be [1, M]
                                loss_multi_focal = sigmoid_focal_loss(
                                    src_masks,
                                    target_masks_prep,
                                    num_instances_for_loss,
                                    alpha=focal_alpha,
                                    gamma=focal_gamma,
                                )
                                loss_multi_dice = dice_loss(
                                    src_masks, target_masks_prep, num_instances_for_loss
                                )
                                loss_multi_iou = iou_loss(
                                    src_masks,
                                    target_masks_prep,
                                    pred_ious,
                                    num_instances_for_loss,
                                    use_l1_loss=iou_use_l1_loss,
                                )

                                # --- Select the best mask based on Focal + Dice ---
                                if num_masks_per_instance > 1:
                                    loss_combo = (
                                        loss_multi_focal * focal_loss_weight
                                        + loss_multi_dice * dice_loss_weight
                                    )  # Shape: [1, M]
                                    best_mask_idx = torch.argmin(
                                        loss_combo, dim=1
                                    )  # Shape: [1]

                                    # Select losses for the best mask index (index 0 because N=1)
                                    loss_focal = loss_multi_focal[
                                        0, best_mask_idx.item()
                                    ]
                                    loss_dice = loss_multi_dice[0, best_mask_idx.item()]

                                    if supervise_all_iou:
                                        loss_iou = (
                                            loss_multi_iou.mean()
                                        )  # Average over all M masks
                                    else:
                                        loss_iou = loss_multi_iou[
                                            0, best_mask_idx.item()
                                        ]
                                else:
                                    # If only one mask predicted (M=1)
                                    loss_focal = loss_multi_focal[0, 0]
                                    loss_dice = loss_multi_dice[0, 0]
                                    loss_iou = loss_multi_iou[0, 0]

                                # --- Combine the selected losses using weights ---
                                total_loss_k = (
                                    loss_focal * focal_loss_weight
                                    + loss_dice * dice_loss_weight
                                    + loss_iou * iou_loss_weight
                                )

                                # Append the loss for this valid instance
                                batch_instance_losses.append(total_loss_k)
                                instances_processed_in_batch += 1

                        except Exception as e_inst:
                            print(
                                f"\nError during instance processing (i={i}, k={k}, class={class_id}): {e_inst}"
                            )
                            import traceback  # Optional: import for detailed error

                            # traceback.print_exc() # Uncomment for detailed debug trace
                            continue  # Continue to next instance
                        # End instance loop k
                    # End class loop
                # End image loop i
            # <<< END PER-INSTANCE LOGIC >>>

            # --- After ALL images and instances in batch ---
            if not batch_instance_losses:
                # print("Warning: No instances processed in this batch.") # Optional debug
                continue  # Skip batch if no losses were computed

            # --- Step 8: Average Loss and Backward Pass ---
            final_batch_loss = torch.mean(torch.stack(batch_instance_losses))

            # Check if the loss requires gradients before backward pass
            if not final_batch_loss.requires_grad:
                print(
                    "Warning: Final batch loss does not require grad. Check model freezing / forward pass / loss calculation."
                )
                continue

            # Backward pass
            if scaler is not None:
                scaler.scale(final_batch_loss).backward()
                # Gradient clipping (optional but often helpful)
                # scaler.unscale_(optimizer) # Unscale gradients before clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                final_batch_loss.backward()
                # Gradient clipping (optional)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_batch_losses.append(final_batch_loss.item())
            train_pbar.set_postfix(
                {
                    "Avg Inst Loss": final_batch_loss.item(),
                    "Num Inst": instances_processed_in_batch,
                }
            )

        # --- After Epoch ---
        avg_epoch_loss = (
            np.mean(epoch_batch_losses) if epoch_batch_losses else float("nan")
        )
        history["train_loss"].append(avg_epoch_loss)
        print(f"\nEpoch {epoch+1} Train Avg Instance Loss: {avg_epoch_loss:.4f}")

        # --- Validation Phase ---
        # (Validation loop remains unchanged - still uses combined prompts/GT)
        model.eval()
        epoch_val_ious = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for batch_val in val_pbar:
                # ... (Existing Validation Logic using prepare_prompts_pos_neg) ...
                # ... (This part is NOT updated for per-instance validation) ...
                images_list_np_val = batch_val["images"]
                original_masks_list_np_val = batch_val["original_masks"]
                current_batch_size_val = len(images_list_np_val)
                if current_batch_size_val == 0:
                    continue

                batch_val_ious = []
                for i_val in range(current_batch_size_val):
                    try:
                        current_image_val = images_list_np_val[i_val]
                        current_mask_val = original_masks_list_np_val[i_val]
                        current_gt_mask_binary_val = (
                            current_mask_val > 0
                        )  # Combined GT for validation IoU

                        # Generate combined prompts using prepare_prompts_pos_neg
                        # Make sure prepare_prompts_pos_neg is defined/imported
                        points_val, labels_val = prepare_prompts_pos_neg(
                            current_mask_val,
                            num_points_per_component,
                            num_negative_points,
                            min_component_area,
                        )
                        if points_val.shape[0] == 0:
                            continue

                        # Predict using combined prompts
                        predictor.set_image(current_image_val)
                        masks_val, scores_val, _ = predictor.predict(
                            point_coords=points_val,
                            point_labels=labels_val,
                            multimask_output=True,
                        )

                        # Calculate IoU of best mask vs COMBINED GT
                        best_iou_for_sample = 0.0
                        if masks_val is not None and len(masks_val) > 0:
                            best_mask_idx_val = np.argmax(scores_val)
                            # Ensure mask is boolean for bitwise ops
                            pred_mask_binary_val = (
                                masks_val[best_mask_idx_val] > 0.0
                            )  # Use threshold if logits? Check predict output
                            inter = np.sum(
                                current_gt_mask_binary_val & pred_mask_binary_val
                            )
                            union = np.sum(
                                current_gt_mask_binary_val | pred_mask_binary_val
                            )
                            iou = inter / (union + 1e-6)
                            best_iou_for_sample = iou
                        batch_val_ious.append(best_iou_for_sample)
                    except Exception as e_val:
                        print(f"\nError during validation sample {i_val}: {e_val}")
                        batch_val_ious.append(0.0)  # Append 0 IoU on error

                # Calculate average IoU for the validation batch postfixed to pbar
                if (
                    epoch_val_ious or batch_val_ious
                ):  # Avoid division by zero if first batch fails
                    current_avg_iou = np.mean(epoch_val_ious + batch_val_ious)
                else:
                    current_avg_iou = 0.0
                val_pbar.set_postfix({"Avg IoU (Combined)": current_avg_iou})
                epoch_val_ious.extend(
                    batch_val_ious
                )  # Accumulate IoUs for epoch average

        avg_val_iou = np.mean(epoch_val_ious) if epoch_val_ious else float("nan")
        history["val_iou"].append(avg_val_iou)
        print(f"\nEpoch {epoch+1} Validation Avg IoU (Combined): {avg_val_iou:.4f}")

        # --- Model Saving ---
        # (Save best model based on combined validation IoU)
        if not np.isnan(avg_val_iou) and avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_path = os.path.join(
                output_dir, f"sam2_decoder_best_epoch{epoch+1}_iou{avg_val_iou:.4f}.pt"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model checkpoint to {save_path}")
        # Save latest model checkpoint regardless of performance
        latest_save_path = os.path.join(output_dir, "sam2_decoder_latest.pt")
        torch.save(model.state_dict(), latest_save_path)

    print(f"Training finished. Best Validation IoU (Combined): {best_val_iou:.4f}")
    # You might want to save history here
    # import json
    # history_path = os.path.join(output_dir, "training_history.json")
    # with open(history_path, 'w') as f:
    #    json.dump(history, f)
    # print(f"Training history saved to {history_path}")

    return history


# --- Visualization ---


def visualize_predictions(  # Keep signature mostly the same
    predictor: SAM2ImagePredictor,
    dataset: torch.utils.data.Dataset,
    num_samples: int = 3,  # Visualize fewer samples might be clearer
    # --- Add params needed for prompt gen ---
    num_points_per_component: int = 1,  # Use same values as training for consistency
    num_negative_points: int = 5,
    min_component_area: int = 10,
    # --- End Add ---
    output_dir: str = "sam_finetune_output",
    checkpoint_path: Optional[str] = None,
):
    """Visualizes predictions by prompting specific instances."""
    if checkpoint_path:
        print(f"Loading model weights from: {checkpoint_path}")
        try:
            device = next(predictor.model.parameters()).device
            predictor.model.load_state_dict(
                torch.load(checkpoint_path, map_location=device)
            )
            print("Weights loaded.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using current model weights.")

    predictor.model.eval()

    os.makedirs(output_dir, exist_ok=True)
    if len(dataset) == 0:
        print("Dataset is empty, cannot visualize.")
        return
    # Ensure num_samples is not larger than dataset length
    num_samples = min(num_samples, len(dataset))
    if num_samples == 0:
        print("Number of samples to visualize is 0.")
        return
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    plt.figure(figsize=(20, 6 * len(indices)))  # Made figure wider for 4 plots
    plot_idx = 1

    for i, idx in enumerate(indices):
        try:
            print(f"Visualizing sample index: {idx}")
            # Dataset now returns image and original integer mask
            sample = dataset[idx]
            image_np = sample["image"]  # HWC uint8 numpy
            original_mask_np = sample["original_mask"]  # HW int numpy
            h, w = original_mask_np.shape

            # --- Instance Identification: Find the largest foreground component ---
            largest_component_info = None
            largest_component_area = -1

            try:
                num_classes = len(CLASS_NAMES)
            except NameError:
                num_classes = 5

            all_components_stats = []
            current_label_offset = 0
            overall_labels_map = np.zeros_like(original_mask_np, dtype=np.int32)

            for class_id in range(1, num_classes):  # Iterate foreground classes
                class_mask_i = (original_mask_np == class_id).astype(np.uint8)
                if not np.any(class_mask_i):
                    continue

                num_labels_k, labels_map_k, stats_k, _ = (
                    cv2.connectedComponentsWithStats(class_mask_i, connectivity=8)
                )

                if num_labels_k > 1:  # Found components
                    for k in range(1, num_labels_k):
                        component_area_k = stats_k[k, cv2.CC_STAT_AREA]
                        if component_area_k >= min_component_area:
                            global_label = k + current_label_offset
                            component_info = {
                                "label": global_label,
                                "area": component_area_k,
                                "class_id": class_id,
                                "local_labels_map": labels_map_k,
                                "local_label": k,
                            }
                            all_components_stats.append(component_info)
                            overall_labels_map[labels_map_k == k] = (
                                global_label  # Optional map of all components
                            )

                            if component_area_k > largest_component_area:
                                largest_component_area = component_area_k
                                largest_component_info = (
                                    component_info  # Keep track of largest
                                )

                    current_label_offset += num_labels_k - 1

            if largest_component_info is None:
                print(
                    f"  Skipping: No suitable foreground components found in sample {idx}."
                )
                # Adjust plot index to maintain grid structure if skipping row
                plot_idx = (i + 1) * 4 + 1
                continue

            print(
                f"  Targeting largest component: Label={largest_component_info['label']}, Area={largest_component_info['area']}, Class={largest_component_info['class_id']}"
            )

            # --- Prompt Generation specifically for the largest instance ---
            points_k_list = []
            labels_k_list = []

            labels_map_target = largest_component_info["local_labels_map"]
            k_target = largest_component_info["local_label"]

            # Positive points from the target instance k
            coords_k = np.argwhere(labels_map_target == k_target)  # (y, x)
            num_pos_available = len(coords_k)
            num_pos_to_sample = min(num_points_per_component, num_pos_available)

            if num_pos_to_sample > 0:
                sampled_indices_k = np.random.choice(
                    num_pos_available, size=num_pos_to_sample, replace=False
                )
                for idx_k in sampled_indices_k:
                    y, x = coords_k[idx_k]
                    points_k_list.append([x, y])
                    labels_k_list.append(1)
            else:
                print(
                    f"  Skipping: Cannot sample positive points for largest component in sample {idx}."
                )
                plot_idx = (i + 1) * 4 + 1
                continue

            # Negative points (global background)
            bg_coords = np.argwhere(original_mask_np == 0)
            num_neg_available = len(bg_coords)
            num_neg_to_sample = min(num_negative_points, num_neg_available)
            if num_neg_to_sample > 0:
                sampled_indices_neg = np.random.choice(
                    num_neg_available, size=num_neg_to_sample, replace=False
                )
                for idx_neg in sampled_indices_neg:
                    y, x = bg_coords[idx_neg]
                    points_k_list.append([x, y])
                    labels_k_list.append(0)

            points_k = np.array(points_k_list)
            labels_k = np.array(labels_k_list)

            if points_k.shape[0] == 0:
                print(
                    f"  Skipping: No points generated for largest component in sample {idx}."
                )
                plot_idx = (i + 1) * 4 + 1
                continue

            # Create the single instance GT mask for visualization
            target_instance_mask_k = labels_map_target == k_target

            # --- Prediction using only points for this instance ---
            print(
                f"  Predicting using {points_k.shape[0]} points ({sum(labels_k)} positive)..."
            )
            predictor.set_image(image_np)
            masks, scores, logits = predictor.predict(
                point_coords=points_k,
                point_labels=labels_k,
                multimask_output=True,
            )
            print(
                f"  Prediction done. Got {len(masks) if masks is not None else 0} masks."
            )

            # --- Plotting (4 columns) ---

            # Plot 1: Image + Instance Prompts
            plt.subplot(len(indices), 4, plot_idx)
            plt.imshow(image_np)
            for pt, lbl in zip(points_k, labels_k):
                color = "lime" if lbl == 1 else "red"
                plt.scatter(
                    pt[0],
                    pt[1],
                    color=color,
                    marker="*",
                    s=120,
                    edgecolor="black",
                    linewidth=1.0,
                )
            plt.title(f"Sample {idx} - Instance Prompts")
            plt.axis("off")
            plot_idx += 1

            # Plot 2: Original Integer Mask (shows all components/classes)
            plt.subplot(len(indices), 4, plot_idx)
            # Use a qualitative colormap for integer labels, ensure background (0) is distinct
            cmap = plt.cm.get_cmap(
                "tab20", num_classes
            )  # Use number of classes for cmap
            cmap.set_under(
                color="black"
            )  # Explicitly set color for values < vmin (i.e., 0)
            plt.imshow(
                original_mask_np, cmap=cmap, vmin=0.1, vmax=num_classes - 0.1
            )  # vmin>0 ensures 0 is bg color
            plt.title(f"Original GT Mask (Labels)")
            plt.axis("off")
            plot_idx += 1

            # Plot 3: Target Instance Mask (highlighting the one prompted)
            plt.subplot(len(indices), 4, plot_idx)
            plt.imshow(image_np)
            # Create RGBA overlay: Green for target instance, transparent otherwise
            overlay = np.zeros((*target_instance_mask_k.shape, 4), dtype=np.float32)
            overlay[target_instance_mask_k] = [0, 1, 0, 0.6]  # RGBA Green with alpha
            plt.imshow(overlay)
            plt.title(f"Target Instance (Area: {largest_component_area})")
            plt.axis("off")
            plot_idx += 1

            # Plot 4: Predicted Mask (best score) resulting from instance prompts
            plt.subplot(len(indices), 4, plot_idx)
            plt.imshow(image_np)
            if masks is not None and len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                best_score = scores[best_mask_idx]
                # Create RGBA overlay: Yellow for prediction
                pred_overlay = np.zeros((*best_mask.shape, 4), dtype=np.float32)
                pred_overlay[best_mask > 0] = [1, 1, 0, 0.6]  # RGBA Yellow with alpha
                plt.imshow(pred_overlay)
                plt.title(f"Prediction (Best Score: {best_score:.3f})")
            else:
                plt.title("Prediction (No mask output)")
            plt.axis("off")
            plot_idx += 1

        except Exception as e_vis:
            print(f"\nError during visualization for sample index {idx}: {e_vis}")
            import traceback

            traceback.print_exc()
            # Try to advance plot index to avoid overwriting
            plot_idx = (i + 1) * 4 + 1  # Move to start of next row
            continue

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust layout rect=[left, bottom, right, top]
    plt.suptitle(
        "Per-Instance Prediction Visualization (Largest Component Prompted)",
        fontsize=16,
        y=0.99,
    )  # Adjust title pos
    viz_path = os.path.join(
        output_dir, "prediction_visualization_per_instance.png"
    )  # New name
    plt.savefig(viz_path)
    print(f"Per-instance prediction visualizations saved to {viz_path}")
    plt.show()
    plt.close()


def visualize_all_instances(
    predictor: SAM2ImagePredictor,
    dataset: torch.utils.data.Dataset,
    num_samples: int = 3,
    min_area_for_smoothing=5,
    # --- Point generation parameters (match training) ---
    num_points_per_component: int = 1,
    num_negative_points: int = 5,
    min_component_area: int = 10,
    use_erosion_for_pos: bool = True,  # Flag to use erosion like in training
    erosion_kernel_size: int = 5,
    # --- End point gen params ---
    output_dir: str = "sam_finetune_output",
    checkpoint_path: Optional[str] = None,
):
    """
    Visualizes SAM2 predictions by prompting *each* valid instance individually
    and combining the results for selected sample images.

    Args:
        predictor: The initialized SAM2ImagePredictor.
        dataset: The dataset object to sample from.
        num_samples: How many random images to visualize.
        num_points_per_component: Positive points per instance.
        num_negative_points: Negative points per image (sampled globally).
        min_component_area: Minimum area for an instance to be prompted.
        use_erosion_for_pos: Whether to erode instance mask before positive sampling.
        erosion_kernel_size: Size of the erosion kernel (e.g., 3 or 5).
        output_dir: Directory to save the visualization.
        checkpoint_path: Optional path to load model weights.
    """
    if checkpoint_path:
        print(f"Loading model weights from: {checkpoint_path}")
        try:
            device = next(predictor.model.parameters()).device
            predictor.model.load_state_dict(
                torch.load(checkpoint_path, map_location=device)
            )
            print("Weights loaded.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using current model weights.")

    predictor.model.eval()
    os.makedirs(output_dir, exist_ok=True)

    if len(dataset) == 0:
        print("Dataset is empty, cannot visualize.")
        return
    num_samples = min(num_samples, len(dataset))
    if num_samples == 0:
        print("Number of samples to visualize is 0.")
        return
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    # --- Setup Plotting ---
    num_cols = 3  # Image+Prompts, Ground Truth, Combined Predictions
    plt.figure(figsize=(num_cols * 6, num_samples * 5))  # Adjust size as needed
    plot_idx = 1

    # Determine number of classes for colormap
    try:
        num_classes = len(CLASS_NAMES)
    except NameError:
        num_classes = 6  # Default guess
    cmap = plt.cm.get_cmap("viridis", num_classes)  # Or 'tab20' etc.
    cmap.set_under(color="black")  # Ensure background (0) is black

    print(f"Starting visualization for {num_samples} samples...")
    for i, idx in enumerate(tqdm(indices, desc="Visualizing Samples")):
        try:
            sample = dataset[idx]
            image_np = sample["image"]  # HWC uint8 numpy
            original_mask_np = sample["original_mask"]  # HW int numpy
            h, w = original_mask_np.shape

            # --- 1. Find all valid components in the ground truth ---
            all_components_info = []
            current_global_label = 1  # Start labeling components from 1
            overall_gt_labels_map = np.zeros_like(
                original_mask_np, dtype=np.int32
            )  # Map with unique label per component

            for class_id in range(1, num_classes):  # Iterate foreground classes
                class_mask = (original_mask_np == class_id).astype(np.uint8)
                if not np.any(class_mask):
                    continue

                num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
                    class_mask, connectivity=8
                )

                if num_labels > 1:  # Found components for this class
                    for k in range(1, num_labels):  # k is local label within class_mask
                        component_area = stats[k, cv2.CC_STAT_AREA]
                        if component_area >= min_component_area:
                            component_info = {
                                "global_label": current_global_label,  # Assign unique global ID
                                "area": component_area,
                                "class_id": class_id,
                                "local_labels_map": labels_map,  # Map where this component is 'k'
                                "local_label": k,
                            }
                            all_components_info.append(component_info)
                            overall_gt_labels_map[labels_map == k] = (
                                current_global_label  # Store global label in GT map
                            )
                            current_global_label += 1

            if not all_components_info:
                print(
                    f"  Skipping sample {idx}: No components found meeting min area {min_component_area}."
                )
                plot_idx += num_cols  # Skip row
                continue

            # --- 2. Prepare for Iterative Prediction ---
            combined_preds_map = np.zeros_like(original_mask_np, dtype=np.int32)
            all_prompts_points = []
            all_prompts_labels = []

            # Set image ONCE for this sample
            predictor.set_image(image_np)

            # --- 3. Loop through each component, prompt, and predict ---
            for component in all_components_info:
                points_k_list = []
                labels_k_list = []
                local_labels_map = component["local_labels_map"]
                k = component["local_label"]

                # --- Positive Point Sampling (using erosion consistent w/ training) ---
                coords_k_for_sampling = None
                if use_erosion_for_pos:
                    instance_mask_k = (local_labels_map == k).astype(np.uint8)
                    kernel = np.ones(
                        (erosion_kernel_size, erosion_kernel_size), np.uint8
                    )
                    eroded_mask_k = cv2.erode(instance_mask_k, kernel, iterations=1)
                    coords_eroded = np.argwhere(eroded_mask_k > 0)
                    if len(coords_eroded) > 0:
                        coords_k_for_sampling = coords_eroded
                    else:  # Fallback if erosion removes instance
                        coords_k_for_sampling = np.argwhere(local_labels_map == k)
                else:  # Use original coords if erosion disabled
                    coords_k_for_sampling = np.argwhere(local_labels_map == k)

                num_pos_available = (
                    len(coords_k_for_sampling)
                    if coords_k_for_sampling is not None
                    else 0
                )
                num_pos_to_sample = min(num_points_per_component, num_pos_available)

                if num_pos_to_sample > 0:
                    sampled_indices = np.random.choice(
                        num_pos_available, size=num_pos_to_sample, replace=False
                    )
                    for idx_k in sampled_indices:
                        y, x = coords_k_for_sampling[idx_k]
                        points_k_list.append([x, y])
                        labels_k_list.append(1)
                # else: # Optional: Skip if no positive points? Usually want negative points anyway.
                #    continue

                # --- Negative Point Sampling (Global Background) ---
                bg_coords = np.argwhere(original_mask_np == 0)
                num_neg_available = len(bg_coords)
                num_neg_to_sample = min(num_negative_points, num_neg_available)
                if num_neg_to_sample > 0:
                    sampled_indices_neg = np.random.choice(
                        num_neg_available, size=num_neg_to_sample, replace=False
                    )
                    for idx_neg in sampled_indices_neg:
                        y, x = bg_coords[idx_neg]
                        points_k_list.append([x, y])
                        labels_k_list.append(0)

                # Combine and store prompts for final plot
                if not points_k_list:
                    continue  # Skip if no points generated at all
                points_k = np.array(points_k_list)
                labels_k = np.array(labels_k_list)
                all_prompts_points.extend(points_k.tolist())
                all_prompts_labels.extend(labels_k.tolist())

                # --- Predict for this instance ---
                masks, scores, _ = predictor.predict(
                    point_coords=points_k,
                    point_labels=labels_k,
                    multimask_output=True,
                )

                # --- Store best prediction in combined map ---
                if masks is not None and len(masks) > 0:
                    best_mask_idx = np.argmax(scores)

                    best_mask = masks[best_mask_idx] > 0.0  # Threshold the numpy array
                    processed_mask = best_mask  # Start with the original prediction
                    if min_area_for_smoothing > 0:
                        try:
                            # Ensure mask is uint8 for cv2/remove_small_regions
                            # best_mask is now boolean, so convert directly
                            mask_uint8 = processed_mask.astype(np.uint8)

                            # 1. Remove small holes
                            smoothed_mask_holes, changed_holes = remove_small_regions(
                                mask_uint8, min_area_for_smoothing, mode="holes"
                            )
                            # 2. Remove small islands from the result
                            smoothed_mask_islands, changed_islands = (
                                remove_small_regions(
                                    smoothed_mask_holes,
                                    min_area_for_smoothing,
                                    mode="islands",
                                )
                            )
                            # Final mask is boolean again
                            processed_mask = smoothed_mask_islands.astype(bool)
                        except Exception as e_smooth:
                            print(
                                f"\nWarning: Error during mask smoothing for instance {component['global_label']}: {e_smooth}. Using original mask."
                            )
                            processed_mask = best_mask

                # Assign the unique global label using the potentially PROCESSED mask
                combined_preds_map[processed_mask] = component["global_label"]

            # --- 4. Plotting for the sample ---
            # Plot 1: Image + All Prompts
            plt.subplot(num_samples, num_cols, plot_idx)
            plt.imshow(image_np)
            if all_prompts_points:  # Check if list is not empty
                all_pts = np.array(all_prompts_points)
                pos_pts = all_pts[np.array(all_prompts_labels) == 1]
                neg_pts = all_pts[np.array(all_prompts_labels) == 0]
                plt.scatter(
                    pos_pts[:, 0],
                    pos_pts[:, 1],
                    color="lime",
                    marker="*",
                    s=100,
                    edgecolor="black",
                    linewidth=0.5,
                    label="Positive",
                )
                plt.scatter(
                    neg_pts[:, 0],
                    neg_pts[:, 1],
                    color="red",
                    marker="*",
                    s=100,
                    edgecolor="black",
                    linewidth=0.5,
                    label="Negative",
                )
            plt.title(f"Sample {idx} - All Prompts ({len(all_components_info)} inst)")
            plt.axis("off")
            plot_idx += 1

            # Plot 2: Original GT Mask (using global labels)
            plt.subplot(num_samples, num_cols, plot_idx)
            # Use vmin > 0 to ensure background (0) uses cmap.set_under color
            plt.imshow(
                overall_gt_labels_map,
                cmap=cmap,
                vmin=0.1,
                vmax=current_global_label - 0.1,
            )
            plt.title(f"Original GT (Labels)")
            plt.axis("off")
            plot_idx += 1

            # Plot 3: Combined Predictions
            plt.subplot(num_samples, num_cols, plot_idx)
            plt.imshow(
                combined_preds_map, cmap=cmap, vmin=0.1, vmax=current_global_label - 0.1
            )
            plt.title(f"Combined Predictions")
            plt.axis("off")
            plot_idx += 1

        except Exception as e_vis:
            print(f"\nError during visualization for sample index {idx}: {e_vis}")
            import traceback

            traceback.print_exc()
            # Try to advance plot index to avoid messing up grid
            plot_idx = (i + 1) * num_cols + 1  # Move to start of next row
            continue

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout
    plt.suptitle(
        "All Instances Prediction Visualization (Individually Prompted)",
        fontsize=16,
        y=0.99,
    )
    viz_path = os.path.join(output_dir, "prediction_visualization_all_instances.png")
    plt.savefig(viz_path)
    print(f"All instances prediction visualizations saved to {viz_path}")
    plt.show()
    plt.close()  # Close figure to free memory
