"""
Visualization utilities for 3DSPA point tracks.

This module provides functions for visualizing 3D point tracks on video frames
with color coding based on scores (e.g., coords_score metric).
"""

import numpy as np
import cv2
import warnings
from pathlib import Path
from typing import Optional, Tuple


def project_3d_to_2d(coords_3d: np.ndarray, intrinsics: np.ndarray, 
                     extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
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
            coords_2d = coords_2d_homogeneous[:, :2] / (coords_2d_homogeneous[:, 2:3] + 1e-8)
            
            # Replace invalid values
            coords_2d = np.nan_to_num(coords_2d, nan=0.0, posinf=0.0, neginf=0.0)
            depths = np.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)
            
            return coords_2d, depths
        except:
            # Return zeros if projection fails
            N = coords_3d.shape[0]
            return np.zeros((N, 2)), np.zeros(N)


def score_to_color_bgr(score: float) -> Tuple[int, int, int]:
    """
    Convert score to BGR color: Red (0/low) → White (0.5) → Blue (1/high)
    Low scores = Red, High scores = Blue
    
    Args:
        score: Score value in [0, 1] range
        
    Returns:
        BGR color tuple (b, g, r) for OpenCV
    """
    s = float(np.clip(score, 0, 1))
    
    if s < 0.5:
        # Red → White (for low scores)
        ratio = s / 0.5
        r = 255
        g = int(255 * ratio)
        b = int(255 * ratio)
    else:
        # White → Blue (for high scores)
        ratio = (s - 0.5) / 0.5
        r = int(255 * (1 - ratio))
        g = int(255 * (1 - ratio))
        b = 255
    
    return (b, g, r)  # BGR format for cv2


def paint_point_track_with_colors(video: np.ndarray, tracks: np.ndarray, 
                                  visibles: Optional[np.ndarray],
                                  scores: np.ndarray, trail: int = 5, 
                                  point_size: int = 2) -> np.ndarray:
    """
    Draws colored points with continuous trail lines on video frames.
    Uses per-frame, per-point scores for coloring.
    """
    video_viz = video.copy()
    T, H, W, _ = video.shape
    N = tracks.shape[0]

    for t in range(min(tracks.shape[1], T)):
        frame = video_viz[t].copy()  # Work on a copy to avoid reference issues

        for i in range(N):
            # Get color for this point at this frame (scores[t, n])
            score_val = scores[t, i]
            color = score_to_color_bgr(score_val)
            
            # Draw trail first
            start_t = max(0, t - trail)
            
            # Draw trail segments for this point
            for prev_t in range(start_t, t):
                # Draw line from prev_t to prev_t+1
                if prev_t + 1 <= t:
                    x_prev, y_prev = int(tracks[i, prev_t, 0]), int(tracks[i, prev_t, 1])
                    x_next, y_next = int(tracks[i, prev_t + 1, 0]), int(tracks[i, prev_t + 1, 1])
                    
                    # Check bounds for both points
                    if (0 <= y_prev < H and 0 <= x_prev < W and 
                        0 <= y_next < H and 0 <= x_next < W):
                        # Use semi-transparent overlay (per segment)
                        overlay = frame.copy()
                        cv2.line(overlay, (x_prev, y_prev), (x_next, y_next), color, 1, cv2.LINE_AA)
                        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Draw point after trail
            x, y = int(tracks[i, t, 0]), int(tracks[i, t, 1])
            if 0 <= y < H and 0 <= x < W:
                cv2.circle(frame, (x, y), point_size, color, -1)
        
        # IMPORTANT: Save the modified frame back to video_viz
        video_viz[t] = frame

    return video_viz


def project_all_tracks(coords_3d: np.ndarray, intrinsics: np.ndarray, 
                       extrinsics: np.ndarray, resize_height: int = 1024,
                       resize_width: int = 1024, 
                       original_height: Optional[int] = None,
                       original_width: Optional[int] = None) -> np.ndarray:
    """
    Project 3D tracks to 2D for all frames.
    Uses resize parameters to scale intrinsics for projection, then scales 
    coordinates back to original dimensions.
        """
    T, N, _ = coords_3d.shape
    
    # Handle intrinsics/extrinsics shape
    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics[None, :, :], (T, 1, 1))
    if extrinsics.ndim == 2:
        extrinsics = np.tile(extrinsics[None, :, :], (T, 1, 1))
    
    # Infer original dimensions if not provided
    # (In practice, this should be provided based on actual video dimensions)
    if original_height is None:
        original_height = 512  # Default fallback
    if original_width is None:
        original_width = 512   # Default fallback
    
    # Compute scaling factors for intrinsics
    scale_x = resize_width / original_width
    scale_y = resize_height / original_height
    
    tracks_2d = np.zeros((N, T, 2))  # [N, T, 2] format
    
    for t in range(T):
        # Scale intrinsics as if video was resized (for better projection)
        intrinsics_scaled = intrinsics[t].copy()
        intrinsics_scaled[0, 0] *= scale_x  # fx
        intrinsics_scaled[1, 1] *= scale_y  # fy
        intrinsics_scaled[0, 2] *= scale_x  # cx
        intrinsics_scaled[1, 2] *= scale_y  # cy
        
        # Project using scaled intrinsics
        coords_2d, _ = project_3d_to_2d(coords_3d[t], intrinsics_scaled, extrinsics[t])
        
        # Scale 2D coordinates back to original video dimensions
        coords_2d[:, 0] /= scale_x  # x: back to original
        coords_2d[:, 1] /= scale_y  # y: back to original
        
        # Convert to (x, y) format and clip to original image bounds
        tracks_2d[:, t, 0] = np.clip(coords_2d[:, 0], 0, original_width - 1)  # x
        tracks_2d[:, t, 1] = np.clip(coords_2d[:, 1], 0, original_height - 1)  # y
    
    return tracks_2d


def load_visualization_data(npz_path: str) -> dict:
    """
    Load data from .npz file for visualization.
    """
    data = np.load(npz_path)
    
    # Extract data
    coords = data["coords"]  # [T, N, 3]
    coords_score = data["coords_score"]  # [T, N, 1] or [T, N]
    video = data["video"]  # [T, C, H, W]
    intrinsics = data["intrinsics"]  # [T, 3, 3] or [3, 3]
    extrinsics = data["extrinsics"]  # [T, 4, 4] or [4, 4]
    visibs = data.get("visibs", None)  # [T, N] or [T, N, 1]
    
    # Handle intrinsics/extrinsics shape
    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics[None, :, :], (coords.shape[0], 1, 1))
    if extrinsics.ndim == 2:
        extrinsics = np.tile(extrinsics[None, :, :], (coords.shape[0], 1, 1))
    
    # Handle visibility
    if visibs is not None:
        if visibs.ndim == 3:
            visibs = visibs[..., 0]
        visibs = visibs > 0.5
    else:
        visibs = np.ones((coords.shape[0], coords.shape[1]), dtype=bool)
    
    # Squeeze coords_score
    coords_score = coords_score.squeeze()  # [T, N]
    
    return {
        'coords': coords,
        'coords_score': coords_score,
        'video': video,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'visibs': visibs
    }


def prepare_video_for_visualization(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare video frames for visualization (convert format, normalize).
    
    Args:
        video: [T, C, H, W] video frames (normalized to [0, 1] or similar)
        
    Returns:
        video_rgb: [T, H, W, 3] RGB video frames (uint8)
        video_bgr: [T, H, W, 3] BGR video frames (uint8) for OpenCV
    """
    # Convert [T, C, H, W] to [T, H, W, C]
    video_rgb = np.transpose(video, (0, 2, 3, 1))
    
    # Clip and convert to uint8
    video_rgb = np.clip(video_rgb, 0, 1)
    video_rgb = (video_rgb * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV operations
    video_bgr = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in video_rgb])
    
    return video_rgb, video_bgr
