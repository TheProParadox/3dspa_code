#!/usr/bin/env python3
"""
CLI tool for visualizing 3DSPA point tracks on video.
This tool loads .npz files containing 3D coordinates, scores, and video frames,
projects the 3D tracks to 2D, and visualizes them with color coding based on scores.
"""

import argparse
import numpy as np
import cv2
import imageio
from pathlib import Path
from typing import Optional

from visualize import (
    load_visualization_data,
    prepare_video_for_visualization,
    project_all_tracks,
    paint_point_track_with_colors
)


def normalize_scores(scores: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Normalize scores to [0, 1] range if requested.
    
    Args:
        scores: [T, N] scores
        normalize: Whether to normalize scores (default: True)
        
    Returns:
        Normalized scores [T, N]
    """
    if not normalize:
        return scores
    
    score_min = scores.min()
    score_max = scores.max()
    
    if score_max > score_min:
        scores_norm = (scores - score_min) / (score_max - score_min)
    else:
        scores_norm = scores - score_min
    
    return scores_norm


def save_video_opencv(video_bgr: np.ndarray, output_path: Path, fps: int = 10):
    """
    Save video using OpenCV VideoWriter.
    
    Args:
        video_bgr: [T, H, W, 3] BGR video frames
        output_path: Path to save video
        fps: Frames per second (default: 10)
    """
    H, W = video_bgr.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    if fourcc == -1:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    for frame in video_bgr:
        out.write(frame)
    out.release()


def save_frames(video_rgb: np.ndarray, output_dir: Path):
    """
    Save each frame as a separate PNG image.
    
    Args:
        video_rgb: [T, H, W, 3] RGB video frames
        output_dir: Directory to save frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(video_rgb):
        frame_path = output_dir / f"frame_{i:05d}.png"
        imageio.imwrite(str(frame_path), frame)
        if (i + 1) % 10 == 0:
            print(f"  Saved frame {i+1}/{len(video_rgb)}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 3DSPA point tracks on video with color coding',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--npz_path', type=str, required=True,
        help='Path to .npz file containing coords, coords_score, video, intrinsics, extrinsics'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (default: same directory as npz file)'
    )
    
    parser.add_argument(
        '--output_name', type=str, default=None,
        help='Output video name (default: {npz_stem}_visualized.mp4)'
    )
    
    parser.add_argument(
        '--trail', type=int, default=5,
        help='Number of frames for trail'
    )
    
    parser.add_argument(
        '--point_size', type=int, default=2,
        help='Radius of points'
    )
    
    parser.add_argument(
        '--resize_height', type=int, default=1024,
        help='Height used for projection scaling'
    )
    
    parser.add_argument(
        '--resize_width', type=int, default=1024,
        help='Width used for projection scaling'
    )
    
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Frames per second for output video'
    )
    
    parser.add_argument(
        '--normalize_scores', action='store_true', default=True,
        help='Normalize scores to [0, 1] range'
    )
    
    parser.add_argument(
        '--no_normalize_scores', action='store_false', dest='normalize_scores',
        help='Use raw scores (must be in [0, 1] range)'
    )
    
    parser.add_argument(
        '--save_frames', action='store_true',
        help='Save individual frames as PNG images'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.npz_path}...")
    data = load_visualization_data(args.npz_path)
    
    coords = data['coords']
    coords_score = data['coords_score']
    video = data['video']
    intrinsics = data['intrinsics']
    extrinsics = data['extrinsics']
    visibs = data['visibs']
    
    T, N = coords.shape[:2]
    _, C, H_orig, W_orig = video.shape
    
    print(f"Loaded {T} frames, {N} points")
    print(f"Original video dimensions: {H_orig}x{W_orig}")
    
    # Prepare video
    print("Preparing video frames...")
    video_rgb, video_bgr = prepare_video_for_visualization(video)
    
    # Project 3D tracks to 2D
    print(f"Projecting 3D tracks to 2D (using resize {args.resize_height}x{args.resize_width} for projection)...")
    tracks_2d = project_all_tracks(
        coords, intrinsics, extrinsics,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
        original_height=H_orig,
        original_width=W_orig
    )
    print("Projection complete!")
    
    # Normalize scores if requested
    scores = coords_score  # [T, N]
    print(f"Score range before normalization: [{scores.min():.4f}, {scores.max():.4f}]")
    
    if args.normalize_scores:
        scores = normalize_scores(scores, normalize=True)
        print(f"Score range after normalization: [{scores.min():.4f}, {scores.max():.4f}]")
    else:
        print("Using raw scores (assuming [0, 1] range)")
    
    # Visualize
    print(f"Visualizing tracks (trail={args.trail}, point_size={args.point_size})...")
    video_viz = paint_point_track_with_colors(
        video_bgr,
        tracks_2d,
        visibs.T,  # [N, T]
        scores,    # [T, N]
        trail=args.trail,
        point_size=args.point_size
    )
    
    # Convert back to RGB for saving frames
    video_viz_rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video_viz])
    
    # Determine output paths
    npz_path_obj = Path(args.npz_path)
    if args.output_dir is None:
        output_dir = npz_path_obj.parent
    else:
        output_dir = Path(args.output_dir)
    
    if args.output_name is None:
        output_stem = npz_path_obj.stem + "_visualized"
    else:
        output_stem = Path(args.output_name).stem
    
    output_video_path = output_dir / f"{output_stem}.mp4"
    output_frames_dir = output_dir / output_stem if args.save_frames else None
    
    # Save video
    print(f"Saving video to {output_video_path}...")
    save_video_opencv(video_viz, output_video_path, fps=args.fps)
    print(f"✅ Saved visualized video to: {output_video_path}")
    
    # Save frames if requested
    if args.save_frames and output_frames_dir is not None:
        print(f"Saving {T} frames to {output_frames_dir}...")
        save_frames(video_viz_rgb, output_frames_dir)
        print(f"✅ Saved {T} frames to: {output_frames_dir}")
    
    print("\n🎨 Visualization complete!")
    print(f"   Output video: {output_video_path}")
    if args.save_frames:
        print(f"   Output frames: {output_frames_dir}")


if __name__ == '__main__':
    main()
