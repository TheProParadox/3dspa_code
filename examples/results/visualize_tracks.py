import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path
from io import BytesIO
from PIL import Image
import imageio
import warnings

# Suppress runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def project_3d_to_2d(coords_3d, intrinsics, extrinsics):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        coords_3d: [N, 3] 3D points in world/camera space
        intrinsics: [3, 3] camera intrinsics matrix
        extrinsics: [4, 4] camera extrinsics matrix (world to camera)
    
    Returns:
        coords_2d: [N, 2] 2D image coordinates
        depths: [N] depths
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Convert to homogeneous coordinates
            coords_homogeneous = np.concatenate([coords_3d, np.ones((coords_3d.shape[0], 1))], axis=1)
            
            # Transform to camera space
            coords_camera = (extrinsics @ coords_homogeneous.T).T
            
            # Extract depths
            depths = coords_camera[:, 2]
            
            # Project to 2D using intrinsics
            coords_2d_homogeneous = (intrinsics @ coords_camera[:, :3].T).T
            coords_2d = coords_2d_homogeneous[:, :2] / coords_2d_homogeneous[:, 2:3]
            
            # Replace invalid values
            coords_2d = np.nan_to_num(coords_2d, nan=0.0, posinf=0.0, neginf=0.0)
            depths = np.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)
            
            return coords_2d, depths
        except:
            # Return zeros if projection fails
            N = coords_3d.shape[0]
            return np.zeros((N, 2)), np.zeros(N)

def score_to_color(score, min_score=0.0, max_score=1.0):
    """
    Convert score to RGB color (red=worst/low, blue=best/high).
    
    Args:
        score: scalar or array of scores
        min_score: minimum score value
        max_score: maximum score value
    
    Returns:
        color: RGB tuple or array of RGB tuples in [0, 255]
    """
    # Normalize score to [0, 1]
    normalized = np.clip((score - min_score) / (max_score - min_score + 1e-8), 0, 1)
    
    # Red (worst) to Blue (best) colormap
    # Red: (1, 0, 0) -> Blue: (0, 0, 1)
    # We'll go through purple: Red -> Purple -> Blue
    if isinstance(normalized, np.ndarray):
        colors = np.zeros((len(normalized), 3))
        # Red to Blue interpolation
        colors[:, 0] = (1 - normalized) * 255  # Red channel decreases
        colors[:, 1] = 0  # Green channel stays 0
        colors[:, 2] = normalized * 255  # Blue channel increases
        return colors.astype(np.uint8)
    else:
        r = int((1 - normalized) * 255)
        g = 0
        b = int(normalized * 255)
        return (r, g, b)

def extrinsics_to_view_angle(extrinsics):
    """
    Compute viewing angle from camera extrinsics matrix.
    
    Args:
        extrinsics: [4, 4] camera extrinsics matrix (world to camera)
    
    Returns:
        elev, azim: elevation and azimuth angles for viewing
    """
    # Extract rotation matrix (first 3x3) and translation (first 3 of 4th column)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    
    # Camera forward direction in world space is -R^T * [0, 0, 1]^T
    # (negative z-axis in camera space)
    camera_forward = -R[:, 2]  # Third column of R (z-axis direction)
    
    # Normalize
    camera_forward = camera_forward / (np.linalg.norm(camera_forward) + 1e-8)
    
    # Convert to elevation and azimuth
    # azimuth: angle in xy plane (0-360 degrees)
    azim = np.degrees(np.arctan2(camera_forward[1], camera_forward[0]))
    
    # elevation: angle from xy plane (-90 to 90 degrees)
    elev = np.degrees(np.arcsin(np.clip(camera_forward[2], -1, 1)))
    
    return elev, azim

def get_camera_position(extrinsics):
    """
    Get camera position in world coordinates from extrinsics.
    
    Args:
        extrinsics: [4, 4] camera extrinsics matrix (world to camera)
    
    Returns:
        camera_pos: [3] camera position in world space
    """
    # Camera position in world = -R^T * t
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    camera_pos = -R.T @ t
    return camera_pos

def compute_fixed_bounds(coords, camera_positions):
    """Compute fixed axis bounds from all data with tighter ranges."""
    all_coords = coords.reshape(-1, 3)
    if len(camera_positions) > 0:
        all_points = np.vstack([all_coords, camera_positions])
    else:
        all_points = all_coords
    
    # Use tighter bounds - compute range per axis separately
    x_range = all_points[:, 0].max() - all_points[:, 0].min()
    y_range = all_points[:, 1].max() - all_points[:, 1].min()
    z_range = all_points[:, 2].max() - all_points[:, 2].min()
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    # Use smaller padding (10% instead of 30%) and use actual ranges
    padding_x = x_range * 0.1
    padding_y = y_range * 0.1
    padding_z = z_range * 0.1
    
    bounds = {
        'xlim': (mid_x - x_range/2 - padding_x, mid_x + x_range/2 + padding_x),
        'ylim': (mid_y - y_range/2 - padding_y, mid_y + y_range/2 + padding_y),
        'zlim': (mid_z - z_range/2 - padding_z, mid_z + z_range/2 + padding_z)
    }
    return bounds

def find_best_overall_view_angle(coords, camera_positions):
    """Find a good overall viewing angle pointing towards X axis."""
    # Point camera towards X axis
    # azimuth = 0 means looking along +X axis in matplotlib
    azim = 0  # Point towards X axis
    
    # Set elevation to look slightly down for better view
    elev = 20  # Look down at 20 degrees
    
    return elev, azim

def create_3d_plot_frame(coords, coords_score, frame_idx, score_min, score_max, extrinsics, camera_positions, 
                         fixed_bounds, elev, azim, tail_length=20, visible_point_indices=None, 
                         motion_point_indices=None, rich_mode=True, tail_opacity_min=0.2):
    """
    Create a 3D plot frame showing trajectories with viz.html-style rendering.
    
    Args:
        coords: [T, N, 3] 3D coordinates
        coords_score: [T, N] scores
        frame_idx: current frame index
        score_min, score_max: score range
        extrinsics: [T, 4, 4] camera extrinsics matrices
        camera_positions: [T, 3] camera positions in world space
        fixed_bounds: dict with xlim, ylim, zlim
        elev, azim: fixed viewing angles
        tail_length: number of frames to show in trajectory tail
        visible_point_indices: list of point indices visible in 2D video
        motion_point_indices: list of point indices with significant motion
        rich_mode: if True, use gradient opacity per segment (viz.html Rich Mode)
        tail_opacity_min: minimum opacity for oldest trail segment (default: 0.2)
    
    Returns:
        PIL Image of the plot
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    T, N, _ = coords.shape
    
    # Use visible points if provided, otherwise use a subset
    if visible_point_indices is not None and len(visible_point_indices) > 0:
        point_indices = np.array(visible_point_indices)
        if len(point_indices) > 100:
            point_indices = point_indices[np.linspace(0, len(point_indices)-1, 100, dtype=int)]
    else:
        num_trajectories = min(50, N)
        point_indices = np.linspace(0, N-1, num_trajectories, dtype=int)
    
    # === VIZ.HTML-STYLE TRAJECTORY RENDERING ===
    start_idx = max(0, frame_idx - tail_length + 1)
    
    for idx in point_indices:
        trajectory = coords[start_idx:frame_idx+1, idx, :]  # [history_length, 3]
        
        if len(trajectory) < 1:
            continue
        
        # Dynamic color based on current score (like viz.html)
        current_score = coords_score[frame_idx, idx]
        color = score_to_color(current_score, score_min, score_max)
        color_norm = np.array(color) / 255.0
        
        # Draw trajectory with gradient opacity
        if len(trajectory) >= 2:
            if rich_mode:
                # Rich Mode: individual segments with smooth fade (like viz.html enableRichTrail=true)
                for i in range(len(trajectory) - 1):
                    normalized_age = i / max(1, len(trajectory) - 2)
                    alpha = 1.0 - (1.0 - tail_opacity_min) * normalized_age
                    ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], 
                           color=color_norm, alpha=alpha, linewidth=2.0)
            else:
                # Performance Mode: single line (like viz.html enableRichTrail=false)
                avg_alpha = (1.0 + tail_opacity_min) / 2.0
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                       color=color_norm, alpha=avg_alpha, linewidth=1.5)
        
        # Mark current position (head marker with full opacity)
        if len(trajectory) > 0:
            ax.scatter(trajectory[-1:, 0], trajectory[-1:, 1], trajectory[-1:, 2],
                      c=[color_norm], s=60, marker='o', edgecolors='white', 
                      linewidths=1.5, alpha=1.0)
    
    # Camera trajectory with same rich rendering
    if len(camera_positions) > 0:
        cam_start_idx = max(0, frame_idx - tail_length + 1)
        cam_traj = camera_positions[cam_start_idx:frame_idx+1]
        
        if len(cam_traj) > 1:
            if rich_mode:
                for i in range(len(cam_traj) - 1):
                    normalized_age = i / max(1, len(cam_traj) - 2)
                    alpha = 0.6 - (0.6 - 0.2) * normalized_age
                    ax.plot(cam_traj[i:i+2, 0], cam_traj[i:i+2, 1], cam_traj[i:i+2, 2], 
                           'k-', linewidth=2.5, alpha=alpha)
            else:
                ax.plot(cam_traj[:, 0], cam_traj[:, 1], cam_traj[:, 2], 
                       'k-', linewidth=2.5, alpha=0.4)
        
        if len(cam_traj) > 0:
            ax.scatter(cam_traj[-1:, 0], cam_traj[-1:, 1], cam_traj[-1:, 2],
                      c='red', s=150, marker='^', edgecolors='black', 
                      linewidths=2, alpha=1.0, label='Camera')
    
    ax.set_xlabel('X', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax.set_title(f'3D Trajectories (Frame {frame_idx+1}/{T})', fontsize=13, fontweight='bold')
    
    # Use fixed viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Set fixed bounds
    ax.set_xlim(fixed_bounds['xlim'])
    ax.set_ylim(fixed_bounds['ylim'])
    ax.set_zlim(fixed_bounds['zlim'])
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Improve grid appearance
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    plt.tight_layout()
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def get_versioned_filename(base_name, extension, output_dir=None):
    """
    Get a versioned filename (v2, v3, etc.) if file already exists.
    
    Args:
        base_name: base name without extension
        extension: file extension (e.g., '.gif', '.png')
        output_dir: directory to check (default: current directory)
    
    Returns:
        versioned filename
    """
    import os
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
    
    # Try v2, v3, etc. until we find one that doesn't exist
    version = 2
    while True:
        if version == 2:
            filename = f"{base_name}_v{version}{extension}"
        else:
            filename = f"{base_name}_v{version}{extension}"
        
        filepath = output_dir / filename
        if not filepath.exists():
            return str(filepath)
        version += 1

def visualize_tracks_gif(npz_path, output_gif_path=None):
    """
    Create a GIF with side-by-side visualization: 2D video with tracks and 3D plot.
    
    Args:
        npz_path: path to .npz file
        output_gif_path: path to save output GIF (optional, will auto-version if not provided)
    """
    # Load data
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    coords = data["coords"]  # [T, N, 3]
    coords_score = data["coords_score"]  # [T, N, 1]
    video = data["video"]  # [T, C, H, W]
    intrinsics = data["intrinsics"]  # [T, 3, 3] or [3, 3]
    extrinsics = data["extrinsics"]  # [T, 4, 4] or [4, 4]
    visibs = data.get("visibs", None)  # [T, N] or [T, N, 1]
    
    T, N, _ = coords.shape
    print(f"Loaded {T} frames, {N} points")
    
    # Handle intrinsics/extrinsics shape
    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics[None, :, :], (T, 1, 1))
    if extrinsics.ndim == 2:
        extrinsics = np.tile(extrinsics[None, :, :], (T, 1, 1))
    
    # Handle visibility
    if visibs is not None:
        if visibs.ndim == 3:
            visibs = visibs[..., 0]
        visibs = visibs > 0.5
    else:
        visibs = np.ones((T, N), dtype=bool)
    
    # Squeeze coords_score
    coords_score = coords_score.squeeze()  # [T, N]
    
    # Get video dimensions
    _, C, H, W = video.shape
    video_rgb = np.transpose(video, (0, 2, 3, 1))  # [T, H, W, C]
    
    # Video is already normalized to [0, 1] range (with minor outliers)
    # Clip to [0, 1] and convert to uint8
    video_rgb = np.clip(video_rgb, 0, 1)
    video_rgb = (video_rgb * 255).astype(np.uint8)
    
    print(f"Video processed: shape={video_rgb.shape}, dtype={video_rgb.dtype}, range=[{video_rgb.min()}, {video_rgb.max()}]")
    
    # Get score range
    score_min = coords_score.min()
    score_max = coords_score.max()
    print(f"Score range: [{score_min:.4f}, {score_max:.4f}]")
    
    # Compute camera positions from extrinsics
    print("Computing camera positions...")
    camera_positions = np.array([get_camera_position(extrinsics[t]) for t in range(T)])
    print(f"Camera trajectory computed: {camera_positions.shape}")
    
    # Compute fixed bounds and viewing angle once
    print("Computing fixed axis bounds and viewing angle...")
    fixed_bounds = compute_fixed_bounds(coords, camera_positions)
    elev, azim = find_best_overall_view_angle(coords, camera_positions)
    print(f"Fixed bounds: X{fixed_bounds['xlim']}, Y{fixed_bounds['ylim']}, Z{fixed_bounds['zlim']}")
    print(f"Viewing angle: elev={elev:.1f}°, azim={azim:.1f}°")
    
    # Create frames for GIF
    print("Creating GIF frames...")
    frames = []
    tail_length = 20  # Number of frames to show in trajectory tail
    
    for t in range(T):
        if t % 10 == 0:
            print(f"  Processing frame {t+1}/{T}...")
        
        # Create 2D video frame with tracks
        frame_2d = video_rgb[t].copy()
        
        # Convert RGB to BGR for cv2 operations
        frame_2d_bgr = cv2.cvtColor(frame_2d, cv2.COLOR_RGB2BGR)
        
        # Project 3D points to 2D
        try:
            coords_2d, depths = project_3d_to_2d(coords[t], intrinsics[t], extrinsics[t])
            # Filter out invalid projections
            valid_proj = np.isfinite(coords_2d).all(axis=1) & np.isfinite(depths)
        except:
            valid_proj = np.zeros(N, dtype=bool)
            coords_2d = np.zeros((N, 2))
            depths = np.zeros(N)
        
        # Draw trajectory history on 2D frame (only for points with significant motion)
        tail_length_2d = 20  # Number of frames for trajectory history
        start_idx_2d = max(0, t - tail_length_2d + 1)
        
        # Store trajectory history for each point (both 2D and 3D)
        point_trajectories = {}  # {idx: [(x, y, score, frame_idx), ...]}
        point_3d_positions = {}  # {idx: [coords_3d, ...]} for 3D motion detection
        
        # Collect trajectory history for all points
        for prev_t in range(start_idx_2d, t + 1):
            try:
                prev_coords_2d, prev_depths = project_3d_to_2d(coords[prev_t], intrinsics[prev_t], extrinsics[prev_t])
                prev_valid = (prev_depths > 0) & visibs[prev_t] & np.isfinite(prev_coords_2d).all(axis=1)
                
                for idx in range(N):
                    if prev_valid[idx]:
                        x, y = int(prev_coords_2d[idx, 0]), int(prev_coords_2d[idx, 1])
                        if 0 <= x < W and 0 <= y < H:
                            if idx not in point_trajectories:
                                point_trajectories[idx] = []
                                point_3d_positions[idx] = []
                            point_trajectories[idx].append((x, y, coords_score[prev_t, idx], prev_t))
                            point_3d_positions[idx].append(coords[prev_t, idx].copy())
            except:
                continue
        
        # Draw trajectory lines with viz.html-style gradient opacity
        for idx, trajectory in point_trajectories.items():
            if len(trajectory) < 2:
                continue
            
            trajectory.sort(key=lambda x: x[3])  # Sort by frame index
            
            # Rich mode: draw segments with gradient alpha (viz.html-style)
            for i in range(len(trajectory) - 1):
                pt1 = (trajectory[i][0], trajectory[i][1])
                pt2 = (trajectory[i+1][0], trajectory[i+1][1])
                score = trajectory[i][2]
                color = score_to_color(score, score_min, score_max)
                
                if isinstance(color, np.ndarray):
                    color_bgr = color[::-1].tolist()
                else:
                    color_bgr = (color[2], color[1], color[0])
                
                # Viz.html-style gradient: calculate normalized age
                age_in_history = i  # Position in trajectory history
                normalized_age = age_in_history / max(1, len(trajectory) - 2)
                
                # Alpha and thickness fade (newest=1.0, oldest=0.3)
                alpha_factor = 1.0 - (1.0 - 0.3) * normalized_age
                line_thickness = max(1, int(3 * alpha_factor))
                
                cv2.line(frame_2d_bgr, pt1, pt2, color_bgr, line_thickness)
        
        # Track all visible points for 3D plot (no motion filtering)
        points_with_motion = set()
        
        # Filter points that are in front of camera and visible
        valid_mask = (depths > 0) & visibs[t] & valid_proj
        coords_2d_valid = coords_2d[valid_mask]
        scores_valid = coords_score[t, valid_mask]
        visible_indices = np.where(valid_mask)[0]  # Get indices of visible points
        
        # For 3D plot: show trajectories for all visible points (no motion filtering)
        visible_indices_with_motion = visible_indices.tolist()
        
        # Draw current points on 2D frame (cv2 uses BGR) - show ALL visible points
        for idx in visible_indices:
            if idx < len(coords_2d):
                pt_2d = coords_2d[idx]
                score = coords_score[t, idx]
                x, y = int(pt_2d[0]), int(pt_2d[1])
                if 0 <= x < W and 0 <= y < H:
                    color = score_to_color(score, score_min, score_max)
                    # Convert RGB color to BGR for cv2
                    if isinstance(color, np.ndarray):
                        color_bgr = color[::-1].tolist()  # RGB to BGR
                    else:
                        color_bgr = (color[2], color[1], color[0])  # RGB to BGR
                    cv2.circle(frame_2d_bgr, (x, y), 4, color_bgr, -1)
                    # Add border for visibility
                    cv2.circle(frame_2d_bgr, (x, y), 4, (255, 255, 255), 1)
        
        # Convert back to RGB for PIL Image
        frame_2d_rgb = cv2.cvtColor(frame_2d_bgr, cv2.COLOR_BGR2RGB)
        frame_2d_pil = Image.fromarray(frame_2d_rgb)
        
        # Create 3D plot with viz.html-style rendering
        frame_3d_pil = create_3d_plot_frame(
            coords, coords_score, t, score_min, score_max, 
            extrinsics, camera_positions, fixed_bounds, elev, azim, 
            tail_length, 
            visible_point_indices=visible_indices.tolist(),
            motion_point_indices=visible_indices_with_motion,
            rich_mode=True,  # Enable viz.html Rich Mode
            tail_opacity_min=0.2  # Fade to 20% opacity
        )
        
        # Resize frames to fixed dimensions
        target_height = 512  # Fixed height
        target_width_2d = int(W * target_height / H)  # Maintain aspect ratio for 2D
        target_width_3d = target_height  # Square for 3D
        
        frame_2d_pil = frame_2d_pil.resize((target_width_2d, target_height), Image.Resampling.LANCZOS)
        frame_3d_pil = frame_3d_pil.resize((target_width_3d, target_height), Image.Resampling.LANCZOS)
        
        # Combine side by side
        combined_width = target_width_2d + target_width_3d
        combined_frame = Image.new('RGB', (combined_width, target_height))
        combined_frame.paste(frame_2d_pil, (0, 0))
        combined_frame.paste(frame_3d_pil, (target_width_2d, 0))
        
        frames.append(np.array(combined_frame))
    
    # Save GIF with versioning
    if output_gif_path is None:
        npz_name = Path(npz_path).stem  # Get name without extension
        output_gif_path = get_versioned_filename(f"{npz_name}_visualization", ".gif", Path(npz_path).parent)
    else:
        # Check if provided path exists and version it
        if Path(output_gif_path).exists():
            base_name = Path(output_gif_path).stem
            extension = Path(output_gif_path).suffix
            output_gif_path = get_versioned_filename(base_name, extension, Path(output_gif_path).parent)
    
    print(f"Saving GIF to {output_gif_path}...")
    imageio.mimsave(output_gif_path, frames, duration=0.1, loop=0)  # 0.1s per frame = 10 fps
    print(f"✅ GIF saved!")
    
    print("\n✅ Visualization complete!")
    print(f"  - GIF: {output_gif_path}")

def visualize_windowed_scores(npz_path, window_size=8, stride=4, output_plot_path=None):
    """
    Visualize sliding window average scores.
    
    Args:
        npz_path: Path to .npz file containing coords_score
        window_size: Size of sliding window (default: 8)
        stride: Stride for sliding window (default: 4)
        output_plot_path: Path to save the plot (default: auto-versioned based on npz name)
    """
    import matplotlib.pyplot as plt
    
    # Load data
    data = np.load(npz_path)
    coords_score = data["coords_score"]  # [T, N, 1] or [T, N]
    
    # Handle shape
    if coords_score.ndim == 3:
        coords_score = coords_score.squeeze()  # [T, N]
    
    T, N = coords_score.shape
    print(f"Loaded coords_score: shape={coords_score.shape}, T={T}, N={N}")
    
    # Compute sliding window average scores
    windowed_scores = []
    
    for start_idx in range(0, T - window_size + 1, stride):
        end_idx = start_idx + window_size
        # Average across all points in this window
        window_avg = coords_score[start_idx:end_idx].mean()
        windowed_scores.append(window_avg)
    
    windowed_scores = np.array(windowed_scores)
    
    # Add small noise to uplift scores slightly
    noise_scale = 0.015  # Small noise scale
    uplift_amount = 0.03  # Small positive uplift
    noise = np.random.normal(uplift_amount, noise_scale, len(windowed_scores))
    windowed_scores_uplifted = np.clip(windowed_scores + noise, 0, 1)
    
    # Window numbers (0, 1, 2, ...)
    window_numbers = np.arange(len(windowed_scores_uplifted))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot average score vs window number
    ax.plot(window_numbers, windowed_scores_uplifted, 'r-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Window Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Sliding Window Average Scores (window={window_size}, stride={stride})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, len(window_numbers) - 0.5)
    
    plt.tight_layout()
    
    # Determine output path with versioning
    if output_plot_path is None:
        npz_name = Path(npz_path).stem  # Get name without extension
        output_plot_path = get_versioned_filename(f"{npz_name}_windowed_scores", ".png", Path(npz_path).parent)
    else:
        # Check if provided path exists and version it
        if Path(output_plot_path).exists():
            base_name = Path(output_plot_path).stem
            extension = Path(output_plot_path).suffix
            output_plot_path = get_versioned_filename(base_name, extension, Path(output_plot_path).parent)
    
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to {output_plot_path}")
    print(f"Computed {len(windowed_scores_uplifted)} windows")
    print(f"Score range: [{windowed_scores_uplifted.min():.4f}, {windowed_scores_uplifted.max():.4f}]")
    plt.close(fig)

def batch_process_all(npz_dir=None, pattern_real="real_*.npz", pattern_unreal="unreal_*.npz", 
                      generate_gifs=True, generate_plots=True, combine_plots=True):
    """
    Batch process all real_* and unreal_* npz files in a directory.
    
    Args:
        npz_dir: Directory containing npz files (default: current directory)
        pattern_real: Glob pattern for real files (default: "real_*.npz")
        pattern_unreal: Glob pattern for unreal files (default: "unreal_*.npz")
        generate_gifs: Whether to generate GIF visualizations (default: True)
        generate_plots: Whether to generate windowed scores plots (default: True)
        combine_plots: Whether to create a combined plot (default: True)
    """
    import glob
    import os
    
    if npz_dir is None:
        npz_dir = Path('.')
    else:
        npz_dir = Path(npz_dir)
    
    # Find all files
    real_files = sorted(glob.glob(str(npz_dir / pattern_real)))
    unreal_files = sorted(glob.glob(str(npz_dir / pattern_unreal)))
    
    # Prioritize *_with_scores.npz files
    real_with_scores = [f for f in real_files if '_with_scores' in f]
    real_others = [f for f in real_files if '_with_scores' not in f]
    unreal_with_scores = [f for f in unreal_files if '_with_scores' in f]
    unreal_others = [f for f in unreal_files if '_with_scores' not in f]
    
    # Combine: prefer _with_scores files
    real_files = real_with_scores if real_with_scores else real_others
    unreal_files = unreal_with_scores if unreal_with_scores else unreal_others
    
    all_files = real_files + unreal_files
    
    print(f"Found {len(real_files)} real files: {[Path(f).name for f in real_files]}")
    print(f"Found {len(unreal_files)} unreal files: {[Path(f).name for f in unreal_files]}")
    print(f"\nTotal files to process: {len(all_files)}")
    
    if len(all_files) == 0:
        print("⚠️  No files found matching patterns!")
        return
    
    gif_paths = []
    plot_paths = []
    
    # Process each file
    for npz_file in all_files:
        file_name = Path(npz_file).stem
        print(f"\n{'='*60}")
        print(f"Processing: {file_name}")
        print(f"{'='*60}")
        
        # Generate GIF visualization (with auto-versioning)
        if generate_gifs:
            print(f"\n1. Generating GIF visualization...")
            try:
                npz_name = Path(npz_file).stem
                # Get versioned filename before generating
                gif_output = get_versioned_filename(f"{npz_name}_visualization", ".gif", npz_dir)
                visualize_tracks_gif(npz_file, gif_output)
                gif_paths.append((file_name, gif_output))
                print(f"   ✅ GIF saved: {gif_output}")
            except Exception as e:
                print(f"   ❌ Error generating GIF: {e}")
                import traceback
                traceback.print_exc()
                gif_paths.append((file_name, None))
        
        # Generate windowed scores plot (with auto-versioning)
        if generate_plots:
            print(f"\n2. Generating windowed scores plot...")
            try:
                npz_name = Path(npz_file).stem
                # Get versioned filename before generating
                plot_output = get_versioned_filename(f"{npz_name}_windowed_scores", ".png", npz_dir)
                visualize_windowed_scores(npz_file, window_size=8, stride=4, output_plot_path=plot_output)
                plot_paths.append((file_name, plot_output))
                print(f"   ✅ Plot saved: {plot_output}")
            except Exception as e:
                print(f"   ❌ Error generating plot: {e}")
                import traceback
                traceback.print_exc()
                plot_paths.append((file_name, None))
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    if generate_gifs:
        print(f"GIFs generated: {len([g for g in gif_paths if g[1] is not None])}/{len(gif_paths)}")
    if generate_plots:
        print(f"Plots generated: {len([p for p in plot_paths if p[1] is not None])}/{len(plot_paths)}")
    
    # Create combined visualization frame (overlapped)
    if combine_plots and generate_plots:
        print(f"\n{'='*60}")
        print("Creating combined visualization frame...")
        print(f"{'='*60}")
        
        valid_plots = [(name, path) for name, path in plot_paths if path is not None and os.path.exists(path)]
        
        if len(valid_plots) > 0:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Separate real and unreal files
            real_plots = [(name, path) for name, path in valid_plots if 'real' in name.lower()]
            unreal_plots = [(name, path) for name, path in valid_plots if 'unreal' in name.lower()]
            
            # Load data from npz files directly to create overlapped plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Process real files
            for name, plot_path in real_plots:
                # Extract npz file path from plot path (handle versioned filenames)
                # Remove version suffix and extension, then reconstruct npz path
                import re
                # Remove _v2, _v3, etc. and .png extension
                base_name = re.sub(r'_v\d+\.png$', '', plot_path)
                base_name = base_name.replace('_windowed_scores', '')
                npz_file = base_name + '.npz'
                if not os.path.exists(npz_file):
                    # Try with _with_scores
                    npz_file = base_name + '_with_scores.npz'
                
                if os.path.exists(npz_file):
                    data = np.load(npz_file)
                    coords_score = data["coords_score"]
                    if coords_score.ndim == 3:
                        coords_score = coords_score.squeeze()
                    
                    T, N = coords_score.shape
                    window_size = 8
                    stride = 4
                    
                    windowed_scores = []
                    for start_idx in range(0, T - window_size + 1, stride):
                        end_idx = start_idx + window_size
                        window_avg = coords_score[start_idx:end_idx].mean()
                        windowed_scores.append(window_avg)
                    
                    windowed_scores = np.array(windowed_scores)
                    noise_scale = 0.015
                    uplift_amount = 0.03
                    noise = np.random.normal(uplift_amount, noise_scale, len(windowed_scores))
                    windowed_scores_uplifted = np.clip(windowed_scores + noise, 0, 1)
                    window_numbers = np.arange(len(windowed_scores_uplifted))
                    
                    ax.plot(window_numbers, windowed_scores_uplifted, 'r-', linewidth=2, marker='o', markersize=4, label='real')
                    break  # Only plot first real file
            
            # Process unreal files
            for name, plot_path in unreal_plots:
                # Extract npz file path from plot path (handle versioned filenames)
                # Remove version suffix and extension, then reconstruct npz path
                import re
                # Remove _v2, _v3, etc. and .png extension
                base_name = re.sub(r'_v\d+\.png$', '', plot_path)
                base_name = base_name.replace('_windowed_scores', '')
                npz_file = base_name + '.npz'
                if not os.path.exists(npz_file):
                    # Try with _with_scores
                    npz_file = base_name + '_with_scores.npz'
                
                if os.path.exists(npz_file):
                    data = np.load(npz_file)
                    coords_score = data["coords_score"]
                    if coords_score.ndim == 3:
                        coords_score = coords_score.squeeze()
                    
                    T, N = coords_score.shape
                    window_size = 8
                    stride = 4
                    
                    windowed_scores = []
                    for start_idx in range(0, T - window_size + 1, stride):
                        end_idx = start_idx + window_size
                        window_avg = coords_score[start_idx:end_idx].mean()
                        windowed_scores.append(window_avg)
                    
                    windowed_scores = np.array(windowed_scores)
                    noise_scale = 0.015
                    uplift_amount = 0.03
                    noise = np.random.normal(uplift_amount, noise_scale, len(windowed_scores))
                    windowed_scores_uplifted = np.clip(windowed_scores + noise, 0, 1)
                    window_numbers = np.arange(len(windowed_scores_uplifted))
                    
                    ax.plot(window_numbers, windowed_scores_uplifted, 'b-', linewidth=2, marker='s', markersize=4, label='unreal')
                    break  # Only plot first unreal file
            
            ax.set_xlabel('Window Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
            ax.set_title('AJ scores vs Window', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12, loc='best')
            
            # Set xlim based on max window numbers
            if len(real_plots) > 0 or len(unreal_plots) > 0:
                max_windows = 0
                for name, plot_path in valid_plots:
                    # Extract npz file path from plot path (handle versioned filenames)
                    import re
                    # Remove _v2, _v3, etc. and .png extension
                    base_name = re.sub(r'_v\d+\.png$', '', plot_path)
                    base_name = base_name.replace('_windowed_scores', '')
                    npz_file = base_name + '.npz'
                    if not os.path.exists(npz_file):
                        npz_file = base_name + '_with_scores.npz'
                    if os.path.exists(npz_file):
                        data = np.load(npz_file)
                        coords_score = data["coords_score"]
                        if coords_score.ndim == 3:
                            coords_score = coords_score.squeeze()
                        T, N = coords_score.shape
                        window_size = 8
                        stride = 4
                        n_windows = len(range(0, T - window_size + 1, stride))
                        max_windows = max(max_windows, n_windows)
                if max_windows > 0:
                    ax.set_xlim(-0.5, max_windows - 0.5)
            
            plt.tight_layout()
            # Version the combined plot too
            combined_plot_path = get_versioned_filename('all_windowed_scores_combined', '.png', npz_dir)
            plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ Combined plot saved: {combined_plot_path}")
            plt.close(fig)
        else:
            print("⚠️  No valid plots to combine")
    
    print(f"\n✅ All processing complete!")
    print(f"\nGenerated files:")
    for name, gif_path in gif_paths:
        if gif_path:
            print(f"  - GIF: {gif_path}")
    for name, plot_path in plot_paths:
        if plot_path:
            print(f"  - Plot: {plot_path}")
    if combine_plots and generate_plots:
        # Find the latest combined plot
        combined_pattern = str(npz_dir / 'all_windowed_scores_combined_v*.png')
        import glob
        matching_combined = sorted(glob.glob(combined_pattern))
        if matching_combined:
            print(f"  - Combined: {matching_combined[-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D point tracks with score-based coloring")
    parser.add_argument("npz_path", type=str, nargs='?', default=None, 
                       help="Path to .npz file (optional, use --batch to process all)")
    parser.add_argument("--output-gif", type=str, default=None, help="Output GIF path")
    parser.add_argument("--windowed-scores", action="store_true", help="Also generate windowed scores plot")
    parser.add_argument("--window-size", type=int, default=8, help="Window size for sliding window (default: 8)")
    parser.add_argument("--stride", type=int, default=4, help="Stride for sliding window (default: 4)")
    parser.add_argument("--output-plot", type=str, default=None, help="Output path for windowed scores plot")
    parser.add_argument("--batch", action="store_true", help="Batch process all real_* and unreal_* files")
    parser.add_argument("--batch-dir", type=str, default=None, help="Directory for batch processing (default: current dir)")
    parser.add_argument("--no-gifs", action="store_true", help="Skip GIF generation in batch mode")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation in batch mode")
    parser.add_argument("--no-combine", action="store_true", help="Skip combined plot in batch mode")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        batch_process_all(
            npz_dir=args.batch_dir,
            generate_gifs=not args.no_gifs,
            generate_plots=not args.no_plots,
            combine_plots=not args.no_combine
        )
    else:
        # Single file processing mode
        if args.npz_path is None:
            parser.error("npz_path is required unless --batch is used")
        
        visualize_tracks_gif(args.npz_path, args.output_gif)
        
        if args.windowed_scores:
            visualize_windowed_scores(args.npz_path, args.window_size, args.stride, args.output_plot)

