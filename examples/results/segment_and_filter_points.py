"""
Segment ball using SAM and filter tracking points to only those on the ball.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app_3rd.sam_utils.hf_sam_predictor import get_hf_sam_predictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  SAM not available. Install transformers: pip install transformers")


def project_3d_to_2d_simple(coords_3d, intrinsics, extrinsics):
    """Project 3D points to 2D image coordinates."""
    # Convert to homogeneous coordinates
    coords_homogeneous = np.concatenate([coords_3d, np.ones((coords_3d.shape[0], 1))], axis=1)
    
    # Transform to camera space
    coords_camera = (extrinsics @ coords_homogeneous.T).T
    
    # Extract depths
    depths = coords_camera[:, 2]
    
    # Project to 2D using intrinsics
    coords_2d_homogeneous = (intrinsics @ coords_camera[:, :3].T).T
    coords_2d = coords_2d_homogeneous[:, :2] / (coords_2d_homogeneous[:, 2:3] + 1e-8)
    
    return coords_2d, depths


def segment_with_sam_auto(image, model_type='vit_h', device=None):
    """
    Automatic segmentation using SAM (segments everything in the image).
    
    Args:
        image: RGB image as numpy array [H, W, 3]
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
        device: Device to run on ('cuda', 'cpu', None for auto)
    
    Returns:
        masks: List of segmentation masks
        scores: Confidence scores for each mask
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install: pip install transformers")
    
    print(f"Loading SAM model ({model_type})...")
    predictor = get_hf_sam_predictor(model_type=model_type, device=device)
    
    print("Running automatic segmentation...")
    # Use automatic mask generation by providing grid points
    H, W = image.shape[:2]
    
    # Create a grid of points across the image
    grid_size = 32
    x_points = np.linspace(W//4, 3*W//4, grid_size)
    y_points = np.linspace(H//4, 3*H//4, grid_size)
    xx, yy = np.meshgrid(x_points, y_points)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Process with SAM
    input_points = [points.tolist()]
    input_labels = [[1] * len(points)]  # All foreground points
    
    inputs = predictor.preprocess(image, input_points, input_labels)
    
    # Get predictions
    with torch.no_grad():
        outputs = predictor.model(**inputs)
    
    # Get masks
    masks = predictor.processor.post_process_masks(
        outputs.pred_masks,
        inputs['original_sizes'],
        inputs['reshaped_input_sizes']
    )[0]
    
    # Convert to numpy
    masks = masks.cpu().numpy()
    scores = outputs.iou_scores.cpu().numpy().flatten()
    
    print(f"Generated {len(masks)} masks")
    
    return masks, scores


def segment_with_sam_click(image, click_point, model_type='vit_h', device=None):
    """
    Segment object at clicked point using SAM.
    
    Args:
        image: RGB image as numpy array [H, W, 3]
        click_point: (x, y) coordinates of click
        model_type: SAM model type
        device: Device to run on
    
    Returns:
        mask: Binary mask [H, W]
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install: pip install transformers")
    
    print(f"Loading SAM model ({model_type})...")
    predictor = get_hf_sam_predictor(model_type=model_type, device=device)
    
    print(f"Segmenting object at point {click_point}...")
    
    # Prepare inputs
    input_points = [[click_point]]
    input_labels = [[1]]  # Foreground point
    
    inputs = predictor.preprocess(image, input_points, input_labels)
    
    # Get predictions
    with torch.no_grad():
        outputs = predictor.model(**inputs)
    
    # Get best mask
    masks = predictor.processor.post_process_masks(
        outputs.pred_masks,
        inputs['original_sizes'],
        inputs['reshaped_input_sizes']
    )[0]
    
    # Get mask with highest score
    scores = outputs.iou_scores.cpu().numpy().flatten()
    best_idx = np.argmax(scores)
    mask = masks[best_idx, 0].cpu().numpy() > 0
    
    print(f"Segmentation complete! Mask size: {mask.sum()} pixels")
    
    return mask


def interactive_segment_selection(image, masks, scores):
    """
    Interactive UI to select the ball segment.
    
    Args:
        image: RGB image
        masks: List of segmentation masks
        scores: Confidence scores
    
    Returns:
        selected_mask: The selected mask
    """
    # Sort masks by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n=== Interactive Segment Selection ===")
    print("Instructions:")
    print("  - Press 'n' for next segment")
    print("  - Press 'p' for previous segment")
    print("  - Press 's' to select current segment")
    print("  - Press 'q' to quit without selection")
    print("=====================================\n")
    
    current_idx = 0
    selected_mask = None
    
    while True:
        idx = sorted_indices[current_idx]
        mask = masks[idx, 0] > 0
        score = scores[idx]
        
        # Create visualization
        vis_image = image.copy()
        
        # Overlay mask in semi-transparent color
        overlay = vis_image.copy()
        overlay[mask] = [0, 255, 0]  # Green
        vis_image = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)
        
        # Add border around mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
        # Add text
        text = f"Segment {current_idx + 1}/{len(masks)} | Score: {score:.3f} | Press 's' to select"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Show
        cv2.imshow('Select Ball Segment', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('n'):  # Next
            current_idx = (current_idx + 1) % len(masks)
        elif key == ord('p'):  # Previous
            current_idx = (current_idx - 1) % len(masks)
        elif key == ord('s'):  # Select
            selected_mask = mask
            print(f"✅ Selected segment {current_idx + 1} with score {score:.3f}")
            break
        elif key == ord('q'):  # Quit
            print("❌ Selection cancelled")
            break
    
    cv2.destroyAllWindows()
    return selected_mask


def filter_points_by_mask(coords_2d, mask, H, W):
    """
    Filter points to only those within the mask.
    
    Args:
        coords_2d: [N, 2] 2D point coordinates
        mask: [H, W] binary mask
        H, W: Image dimensions
    
    Returns:
        valid_indices: Indices of points within mask
    """
    valid_indices = []
    
    for i, (x, y) in enumerate(coords_2d):
        x_int, y_int = int(round(x)), int(round(y))
        
        # Check if within image bounds
        if 0 <= x_int < W and 0 <= y_int < H:
            # Check if within mask
            if mask[y_int, x_int]:
                valid_indices.append(i)
    
    return np.array(valid_indices)


def visualize_filtered_points(image, coords_2d, filtered_indices, mask, output_path):
    """Visualize the filtered points on the image."""
    vis_image = image.copy()
    
    # Draw mask boundary
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
    
    # Draw all points in gray
    for x, y in coords_2d:
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < image.shape[1] and 0 <= y_int < image.shape[0]:
            cv2.circle(vis_image, (x_int, y_int), 2, (128, 128, 128), -1)
    
    # Draw filtered points in color
    for idx in filtered_indices:
        x, y = coords_2d[idx]
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < image.shape[1] and 0 <= y_int < image.shape[0]:
            cv2.circle(vis_image, (x_int, y_int), 3, (255, 0, 0), -1)
            cv2.circle(vis_image, (x_int, y_int), 3, (255, 255, 255), 1)
    
    # Add text
    text = f"Filtered Points: {len(filtered_indices)} / {len(coords_2d)}"
    cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"✅ Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment ball and filter tracking points")
    parser.add_argument("npz_path", type=str, help="Path to .npz tracking file")
    parser.add_argument("--model-type", type=str, default="vit_h", choices=['vit_b', 'vit_l', 'vit_h'],
                       help="SAM model type (default: vit_h)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--click", type=int, nargs=2, metavar=('X', 'Y'), 
                       help="Click point (x, y) to segment ball directly")
    parser.add_argument("--output", type=str, default=None, help="Output path for filtered .npz")
    
    args = parser.parse_args()
    
    if not SAM_AVAILABLE:
        print("❌ SAM not available. Install transformers:")
        print("   pip install transformers torch")
        return
    
    # Load data
    print(f"Loading data from {args.npz_path}...")
    data = np.load(args.npz_path)
    
    coords = data["coords"]  # [T, N, 3]
    video = data["video"]  # [T, C, H, W]
    intrinsics = data["intrinsics"]  # [T, 3, 3] or [3, 3]
    extrinsics = data["extrinsics"]  # [T, 4, 4] or [4, 4]
    
    T, N, _ = coords.shape
    _, C, H, W = video.shape
    print(f"Loaded: {T} frames, {N} points, {H}x{W} video")
    
    # Get first frame
    first_frame = video[0].transpose(1, 2, 0)  # [H, W, C]
    first_frame = np.clip(first_frame, 0, 1)
    first_frame_uint8 = (first_frame * 255).astype(np.uint8)
    
    # Handle intrinsics/extrinsics
    if intrinsics.ndim == 2:
        intrinsics_0 = intrinsics
    else:
        intrinsics_0 = intrinsics[0]
    
    if extrinsics.ndim == 2:
        extrinsics_0 = extrinsics
    else:
        extrinsics_0 = extrinsics[0]
    
    # Project points to first frame
    print("Projecting 3D points to 2D...")
    coords_2d, depths = project_3d_to_2d_simple(coords[0], intrinsics_0, extrinsics_0)
    valid_proj = (depths > 0) & np.isfinite(coords_2d).all(axis=1)
    print(f"Valid projections: {valid_proj.sum()} / {N}")
    
    # Segment ball
    if args.click:
        # Direct click-based segmentation
        mask = segment_with_sam_click(first_frame_uint8, args.click, args.model_type, args.device)
    else:
        # Automatic segmentation with interactive selection
        import torch
        masks, scores = segment_with_sam_auto(first_frame_uint8, args.model_type, args.device)
        mask = interactive_segment_selection(first_frame_uint8, masks, scores)
        
        if mask is None:
            print("❌ No segment selected. Exiting.")
            return
    
    # Filter points by mask
    print("Filtering points by mask...")
    filtered_indices = filter_points_by_mask(coords_2d[valid_proj], mask, H, W)
    
    # Map back to original indices
    valid_original_indices = np.where(valid_proj)[0]
    ball_point_indices = valid_original_indices[filtered_indices]
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Total points: {N}")
    print(f"Points on ball: {len(ball_point_indices)}")
    print(f"Percentage: {100 * len(ball_point_indices) / N:.1f}%")
    print(f"{'='*60}\n")
    
    # Visualize
    output_dir = Path(args.npz_path).parent
    vis_path = output_dir / f"{Path(args.npz_path).stem}_filtered_points.png"
    visualize_filtered_points(first_frame_uint8, coords_2d, ball_point_indices, mask, str(vis_path))
    
    # Save filtered data
    if args.output is None:
        args.output = str(output_dir / f"{Path(args.npz_path).stem}_ball_only.npz")
    
    # Create filtered dataset
    filtered_data = {
        'coords': coords[:, ball_point_indices, :],
        'coords_score': data['coords_score'][:, ball_point_indices],
        'video': video,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'ball_point_indices': ball_point_indices,  # Save for reference
        'original_num_points': N
    }
    
    # Copy visibs if present
    if 'visibs' in data:
        filtered_data['visibs'] = data['visibs'][:, ball_point_indices]
    
    np.savez(args.output, **filtered_data)
    print(f"✅ Filtered data saved: {args.output}")
    print(f"\n🎯 You can now visualize with:")
    print(f"   python visualize_tracks.py {args.output}")


if __name__ == "__main__":
    main()

