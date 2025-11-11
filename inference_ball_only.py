"""
Run SpaTrack inference on ball points only (segmented using SAM).
"""
import pycolmap
from models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
import moviepy.editor as mp
from models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import glob
from rich import print
import argparse
import decord
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from pathlib import Path
import sys

# Add paths for SAM
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app_3rd.sam_utils.hf_sam_predictor import get_hf_sam_predictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[yellow]⚠️  SAM not available. Will use full frame if no mask provided.[/yellow]")


def segment_ball_with_sam(image, model_type='vit_h', device=None):
    """
    Segment ball using SAM with interactive selection.
    
    Args:
        image: RGB image as numpy array [H, W, 3] (0-255 uint8)
        model_type: SAM model type
        device: Device to run on
    
    Returns:
        mask: Binary mask [H, W]
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install: pip install transformers")
    
    print(f"[cyan]Loading SAM model ({model_type})...[/cyan]")
    predictor = get_hf_sam_predictor(model_type=model_type, device=device)
    
    print("[cyan]Running automatic segmentation...[/cyan]")
    H, W = image.shape[:2]
    
    # Create a grid of points across the image
    grid_size = 32
    x_points = np.linspace(W//4, 3*W//4, grid_size)
    y_points = np.linspace(H//4, 3*H//4, grid_size)
    xx, yy = np.meshgrid(x_points, y_points)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Process with SAM
    input_points = [points.tolist()]
    input_labels = [[1] * len(points)]
    
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
    
    print(f"[cyan]Generated {len(masks)} masks[/cyan]")
    
    # Interactive selection
    selected_mask = interactive_segment_selection(image, masks, scores)
    
    return selected_mask


def interactive_segment_selection(image, masks, scores):
    """
    Interactive UI to select the ball segment.
    
    Args:
        image: RGB image (0-255 uint8)
        masks: Array of masks [N, 1, H, W]
        scores: Confidence scores
    
    Returns:
        selected_mask: The selected mask [H, W]
    """
    # Sort masks by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n[bold cyan]═══ Interactive Ball Segment Selection ═══[/bold cyan]")
    print("[yellow]Instructions:[/yellow]")
    print("  • Press [bold]'n'[/bold] for next segment")
    print("  • Press [bold]'p'[/bold] for previous segment")
    print("  • Press [bold]'s'[/bold] to select current segment (the ball)")
    print("  • Press [bold]'q'[/bold] to quit without selection")
    print("[bold cyan]════════════════════════════════════════[/bold cyan]\n")
    
    current_idx = 0
    selected_mask = None
    
    while True:
        idx = sorted_indices[current_idx]
        mask = masks[idx, 0] > 0
        score = scores[idx]
        
        # Create visualization
        vis_image = image.copy()
        
        # Overlay mask in semi-transparent green
        overlay = vis_image.copy()
        overlay[mask] = [0, 255, 0]
        vis_image = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)
        
        # Add border around mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 3)
        
        # Add text overlay
        text = f"Segment {current_idx + 1}/{len(masks)} | Score: {score:.3f} | Press 's' to select"
        cv2.putText(vis_image, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(vis_image, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Show
        cv2.imshow('Select Ball Segment (Green Overlay)', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('n'):  # Next
            current_idx = (current_idx + 1) % len(masks)
        elif key == ord('p'):  # Previous
            current_idx = (current_idx - 1) % len(masks)
        elif key == ord('s'):  # Select
            selected_mask = mask
            print(f"[bold green]✅ Selected segment {current_idx + 1} with score {score:.3f}[/bold green]")
            break
        elif key == ord('q'):  # Quit
            print("[bold red]❌ Selection cancelled[/bold red]")
            break
    
    cv2.destroyAllWindows()
    return selected_mask


def segment_ball_with_click(image, click_point, model_type='vit_h', device=None):
    """
    Segment ball at clicked point using SAM.
    
    Args:
        image: RGB image (0-255 uint8)
        click_point: (x, y) coordinates
        model_type: SAM model type
        device: Device
    
    Returns:
        mask: Binary mask [H, W]
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install: pip install transformers")
    
    print(f"[cyan]Loading SAM model ({model_type})...[/cyan]")
    predictor = get_hf_sam_predictor(model_type=model_type, device=device)
    
    print(f"[cyan]Segmenting ball at point {click_point}...[/cyan]")
    
    # Prepare inputs
    input_points = [[click_point]]
    input_labels = [[1]]
    
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
    
    scores = outputs.iou_scores.cpu().numpy().flatten()
    best_idx = np.argmax(scores)
    mask = masks[best_idx, 0].cpu().numpy() > 0
    
    print(f"[green]✅ Segmentation complete! Mask size: {mask.sum()} pixels[/green]")
    
    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpaTrack inference on ball points only")
    parser.add_argument("--track_mode", type=str, default="offline", choices=['offline', 'online'])
    parser.add_argument("--data_type", type=str, default="RGBD", choices=['RGBD', 'RGB'])
    parser.add_argument("--data_dir", type=str, default="examples/results")
    parser.add_argument("--video_name", type=str, default="real_most_with_scores", help="Name without extension")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to track on ball")
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    
    # SAM options
    parser.add_argument("--sam_model", type=str, default="vit_h", choices=['vit_b', 'vit_l', 'vit_h'],
                       help="SAM model type")
    parser.add_argument("--click", type=int, nargs=2, metavar=('X', 'Y'),
                       help="Click point (x, y) to segment ball directly")
    parser.add_argument("--mask_path", type=str, default=None,
                       help="Path to pre-saved ball mask (skip SAM if provided)")
    parser.add_argument("--save_mask", action="store_true",
                       help="Save the ball mask for reuse")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = os.path.join(args.data_dir, f"{args.video_name}_ball_tracking")
    fps = int(args.fps)
    
    print("[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    print("[bold cyan]   SpaTrack Ball-Only Inference[/bold cyan]")
    print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
    
    # Load VGGT4Track for depth/pose if needed
    if args.data_type == "RGB":
        print("[cyan]Loading VGGT4Track model...[/cyan]")
        vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        vggt4track_model.eval()
        vggt4track_model = vggt4track_model.to("cuda")

    # Load data
    if args.data_type == "RGBD":
        npz_dir = os.path.join(args.data_dir, f"{args.video_name}.npz")
        print(f"[cyan]Loading RGBD data from {npz_dir}...[/cyan]")
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        video_tensor = data_npz_load["video"] * 255
        video_tensor = torch.from_numpy(video_tensor)
        video_tensor = video_tensor[::fps]
        depth_tensor = data_npz_load["depths"]
        depth_tensor = depth_tensor[::fps]
        intrs = data_npz_load["intrinsics"]
        intrs = intrs[::fps]
        extrs = np.linalg.inv(data_npz_load["extrinsics"])
        extrs = extrs[::fps]
        unc_metric = None
        
    elif args.data_type == "RGB":
        vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
        print(f"[cyan]Loading RGB video from {vid_dir}...[/cyan]")
        video_reader = decord.VideoReader(vid_dir)
        video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
        video_tensor = video_tensor[::fps].float()

        video_tensor = preprocess_image(video_tensor)[None]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = vggt4track_model(video_tensor.cuda()/255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
        
        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
        data_npz_load = {}
    
    # Get first frame for segmentation
    first_frame = video_tensor[0].permute(1, 2, 0).numpy()  # [H, W, C]
    first_frame_uint8 = first_frame.astype(np.uint8)
    frame_H, frame_W = first_frame.shape[:2]
    
    print(f"[cyan]Video shape: {video_tensor.shape}[/cyan]")
    print(f"[cyan]Frame size: {frame_H} x {frame_W}[/cyan]")
    
    # Get ball mask
    if args.mask_path and os.path.exists(args.mask_path):
        print(f"[cyan]Loading pre-saved mask from {args.mask_path}...[/cyan]")
        mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (frame_W, frame_H))
        mask = mask > 127
    else:
        if not SAM_AVAILABLE:
            print("[bold red]❌ SAM not available and no mask provided![/bold red]")
            print("[yellow]Install SAM: pip install transformers torch[/yellow]")
            sys.exit(1)
        
        if args.click:
            # Direct click-based segmentation
            mask = segment_ball_with_click(first_frame_uint8, args.click, args.sam_model, 'cuda')
        else:
            # Interactive segmentation
            mask = segment_ball_with_sam(first_frame_uint8, args.sam_model, 'cuda')
        
        if mask is None:
            print("[bold red]❌ No mask selected. Exiting.[/bold red]")
            sys.exit(1)
        
        # Save mask if requested
        if args.save_mask:
            mask_save_path = os.path.join(args.data_dir, f"{args.video_name}_ball_mask.png")
            cv2.imwrite(mask_save_path, (mask * 255).astype(np.uint8))
            print(f"[green]✅ Mask saved: {mask_save_path}[/green]")
    
    # Sample points on ball
    print(f"\n[bold cyan]Sampling {args.num_points} points on ball...[/bold cyan]")
    np.random.seed(42)
    valid_mask = mask.astype(bool)
    ys, xs = np.where(valid_mask)
    num_valid = len(xs)
    
    if num_valid < args.num_points:
        print(f"[yellow]⚠️  Only {num_valid} valid pixels found; using all of them.[/yellow]")
        selected = np.arange(num_valid)
    else:
        selected = np.random.choice(num_valid, size=args.num_points, replace=False)
    
    xy_points = np.stack([xs[selected], ys[selected]], axis=1)
    grid_pts = torch.from_numpy(xy_points).float().unsqueeze(0)
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    
    print(f"[green]✅ Sampled {len(xy_points)} points on ball surface[/green]")
    
    # Visualize sampled points
    vis_points = first_frame_uint8.copy()
    for x, y in xy_points:
        cv2.circle(vis_points, (int(x), int(y)), 2, (0, 255, 0), -1)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "ball_points_visualization.png"), 
                cv2.cvtColor(vis_points, cv2.COLOR_RGB2BGR))
    print(f"[green]✅ Point visualization saved to {out_dir}/ball_points_visualization.png[/green]")
    
    # Load SpaTrack model
    print(f"\n[bold cyan]Loading SpaTrack model ({args.track_mode} mode)...[/bold cyan]")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    model.spatrack.track_num = args.vo_points
    model.eval()
    model.to("cuda")
    
    # Setup visualizer
    viser = Visualizer(save_dir=out_dir, grayscale=True, fps=10, pad_value=0, tracks_leave_trace=5)
    
    # Run inference
    print(f"\n[bold cyan]Running SpaTrack inference...[/bold cyan]")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
        # Resize if needed
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs_out[:,:2,:] = intrs_out[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))

        # Visualize
        print("[cyan]Creating visualization...[/cyan]")
        viser.visualize(video=video[None],
                        tracks=track2d_pred[None][...,:2],
                        visibility=vis_pred[None],
                        filename="ball_tracking")

        # Save results
        print("[cyan]Saving results...[/cyan]")
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs_out.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        data_npz_load["ball_mask"] = mask  # Save the ball mask
        data_npz_load["ball_query_points"] = xy_points  # Save query points

        output_path = os.path.join(out_dir, 'result_ball_only.npz')
        np.savez(output_path, **data_npz_load)
        
    print(f"\n[bold green]{'═'*50}[/bold green]")
    print(f"[bold green]✅ Ball tracking complete![/bold green]")
    print(f"[bold green]{'═'*50}[/bold green]")
    print(f"\n[bold]Results saved to:[/bold] [cyan]{out_dir}[/cyan]")
    print(f"  • NPZ file: [cyan]{output_path}[/cyan]")
    print(f"  • Visualization: [cyan]{out_dir}/ball_tracking.mp4[/cyan]")
    print(f"\n[bold yellow]To visualize with viz.html:[/bold yellow]")
    print(f"  [cyan]python tapip3d_viz.py {output_path}[/cyan]")
    print(f"\n[bold yellow]To create side-by-side GIF:[/bold yellow]")
    print(f"  [cyan]python examples/results/visualize_tracks.py {output_path}[/cyan]\n")

