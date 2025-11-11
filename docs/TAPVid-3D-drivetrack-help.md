# TAPVid-3D DriveTrack Generation Script Help

## Command
```bash
python3 -m tapnet.tapvid3d.annotation_generation.generate_drivetrack [OPTIONS]
```

## Arguments

### `--output_dir` (string)
- **Default**: `tapvid3d_dataset/drivetrack/`
- **Description**: Path to folder to store output npz files containing all fields.

### `--split` (enum)
- **Default**: `all`
- **Options**: `minival`, `full_eval`, `all`
- **Description**: Which split to download:
  - `minival`: Download the minival split
  - `full_eval`: Download the full evaluation split
  - `all`: Download all splits

### `--debug` (boolean)
- **Default**: `False`
- **Description**: Whether to run in debug mode, downloads only one video.

## Examples

### Download minival split:
```bash
python3 -m tapnet.tapvid3d.annotation_generation.generate_drivetrack \
  --split=minival \
  --output_dir=./tapvid3d_dataset/drivetrack
```

### Download minival split (debug mode - one video only):
```bash
python3 -m tapnet.tapvid3d.annotation_generation.generate_drivetrack \
  --split=minival \
  --output_dir=./tapvid3d_dataset/drivetrack \
  --debug
```

### Download full evaluation split:
```bash
python3 -m tapnet.tapvid3d.annotation_generation.generate_drivetrack \
  --split=full_eval \
  --output_dir=./tapvid3d_dataset/drivetrack
```

## Notes

- **No source videos needed**: The DriveTrack script downloads preprocessed `.npz` files directly from Google Cloud Storage, so you don't need to download source Waymo Open Dataset videos separately.
- The script automatically creates the output directory if it doesn't exist.
- Files are downloaded from a GCS bucket and contain both preprocessed annotations and source videos.

