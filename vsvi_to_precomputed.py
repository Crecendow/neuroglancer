#!/usr/bin/env python3
"""
Convert Vaa3D VSVI/VISI tiled dataset to neuroglancer precomputed format.

Usage:
    python vsvi_to_precomputed.py <input_dir> <output_dir>

Input structure expected:
    input_dir/
      EM_img/
        IARPA_JWR15.vsvi
        mip0/  <section>_*/  <section>_*_tr{r}-tc{c}.png
        mip1/  slice<section>/  <section>_tr{r}-tc{c}.png
        mip2/  ...
      seg/
        IARPA_JWR15_SEG.vsvi
        mip0/  <section:04d>/  seg_r{r}_c{c}.png
        mip1/  slice_<section:04d>/  <section:04d>_tr{r}-tc{c}.png
        ...
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# --- Constants ---
CHUNK_SIZE = (128, 128, 128)


# ============================================================
# VSVI metadata parsing
# ============================================================

def parse_vsvi(path):
    """Parse VSVI metadata, handling Windows-style backslash paths."""
    with open(path) as f:
        raw = f.read()
    # VSVI files use Windows-style backslash paths like ".\mip0\%04d",
    # which are not valid JSON. Replace ALL backslashes with forward slashes.
    fixed = raw.replace('\\', '/')
    return json.loads(fixed)


# ============================================================
# Section → directory mapping
# ============================================================

def discover_em_mip0_sections(mip0_path):
    """
    EM mip0 directories are named like '0306_W02_Sec001_montaged'.
    Returns dict: {section_number: dir_name}
    """
    mapping = {}
    for d in sorted(Path(mip0_path).iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r'^(\d{4})_', d.name)
        if m:
            sec = int(m.group(1))
            mapping[sec] = d.name
    return mapping


def discover_seg_mip0_sections(mip0_path):
    """
    Seg mip0 directories are named like '0000', '0001', ...
    Returns dict: {section_number: dir_name}
    """
    mapping = {}
    for d in sorted(Path(mip0_path).iterdir()):
        if not d.is_dir():
            continue
        if re.match(r'^\d{4}$', d.name):
            sec = int(d.name)
            mapping[sec] = d.name
    return mapping


def discover_em_mipN_sections(mip_path):
    """
    EM mipN directories are named like 'slice0306'.
    Returns dict: {section_number: dir_name}
    """
    mapping = {}
    for d in sorted(Path(mip_path).iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r'^slice(\d+)$', d.name)
        if m:
            sec = int(m.group(1))
            mapping[sec] = d.name
    return mapping


def discover_seg_mipN_sections(mip_path):
    """
    Seg mipN directories are named like 'slice_0000'.
    Returns dict: {section_number: dir_name}
    """
    mapping = {}
    for d in sorted(Path(mip_path).iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r'^slice_(\d+)$', d.name)
        if m:
            sec = int(m.group(1))
            mapping[sec] = d.name
    return mapping


# ============================================================
# Tile path construction
# ============================================================

def em_mip0_tile_path(mip_dir, dir_name, section, row, col):
    """EM mip0: {dir_name}/{dir_name}_tr{row}-tc{col}.png  (row,col are 1-based)"""
    return os.path.join(mip_dir, dir_name,
                        f"{dir_name}_tr{row}-tc{col}.png")


def seg_mip0_tile_path(mip_dir, dir_name, section, row, col):
    """Seg mip0: {dir_name}/seg_r{row}_c{col}.png  (row,col are 0-based)"""
    return os.path.join(mip_dir, dir_name,
                        f"seg_r{row}_c{col}.png")


def em_mipN_tile_path(mip_dir, dir_name, section, row, col):
    """EM mipN: slice{section}/{section:04d}_tr{row}-tc{col}.png  (row,col are 1-based)"""
    return os.path.join(mip_dir, dir_name,
                        f"{section:04d}_tr{row}-tc{col}.png")


def seg_mipN_tile_path(mip_dir, dir_name, section, row, col):
    """Seg mipN: slice_{section:04d}/{section:04d}_tr{row}-tc{col}.png  (0-based assumed)"""
    return os.path.join(mip_dir, dir_name,
                        f"{section:04d}_tr{row}-tc{col}.png")


# ============================================================
# Tile range discovery
# ============================================================

def discover_tile_range(mip_dir, section_map, pattern):
    """
    Determine the max row/col from the first valid section directory.
    All sections share the same tile grid, so scanning one is enough.
    Returns: {section: (dir_name, max_row, max_col)}
    """
    # First, find max row/col from a single section (all sections have same tile grid)
    global_max_r, global_max_c = 0, 0
    sample_dir = None
    for section, dir_name in section_map.items():
        section_dir = os.path.join(mip_dir, dir_name)
        if not os.path.isdir(section_dir):
            continue
        sample_dir = section_dir
        # Scan this directory to find max row/col
        for f in os.listdir(section_dir):
            if not f.endswith('.png'):
                continue
            if pattern == 'seg_mip0':
                m = re.search(r'seg_r(\d+)_c(\d+)', f)
            else:
                m = re.search(r'tr(\d+)-tc(\d+)', f)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                global_max_r = max(global_max_r, r)
                global_max_c = max(global_max_c, c)
        break  # one section is enough

    print(f"    Sampling tiles from '{os.path.basename(sample_dir)}' → "
          f"max_row={global_max_r}, max_col={global_max_c}")

    if global_max_c == 0 and global_max_r == 0:
        return {}

    # Build section_info with the shared max_r/max_c
    print(f"    Verifying {len(section_map)} section directories...")
    result = {}
    checked = 0
    for section, dir_name in section_map.items():
        section_dir = os.path.join(mip_dir, dir_name)
        if os.path.isdir(section_dir):
            result[section] = (dir_name, global_max_r, global_max_c)
        checked += 1
        if checked % 500 == 0:
            print(f"      {checked}/{len(section_map)}...")

    return result


# ============================================================
# Chunk processing
# ============================================================

# Global variables set before ProcessPoolExecutor (shared via fork on Linux)
_PROCESS_SHARED = {}


def process_chunk(args):
    (cx, cy, cz) = args

    shared = _PROCESS_SHARED
    mip_dir = shared['mip_dir']
    section_info = shared['section_info']
    section_list = shared['section_list']
    chunk_sx = shared['chunk_sx']
    chunk_sy = shared['chunk_sy']
    chunk_sz = shared['chunk_sz']
    tile_size = shared['tile_size']
    dim_x = shared['dim_x']
    dim_y = shared['dim_y']
    dim_z = shared['dim_z']
    pattern = shared['pattern']
    out_dir = shared['out_dir']
    scale_key = shared['scale_key']
    is_em = shared['is_em']
    tile_base = shared['tile_base']
    force = shared.get('force', False)

    ts = tile_size

    # Voxel range for this chunk
    x0 = cx * chunk_sx
    y0 = cy * chunk_sy
    z0 = cz * chunk_sz
    x1 = min(x0 + chunk_sx, dim_x)
    y1 = min(y0 + chunk_sy, dim_y)
    z1 = min(z0 + chunk_sz, dim_z)

    if x0 >= dim_x or y0 >= dim_y or z0 >= dim_z:
        return 'skip'

    if z0 >= len(section_list):
        return 'skip'

    # Check if this chunk already exists (resume support)
    chunk_filename = f"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"
    chunk_dir = os.path.join(out_dir, scale_key)
    chunk_path = os.path.join(chunk_dir, chunk_filename)

    if not force and os.path.exists(chunk_path):
        return 'exists'

    # Determine which tiles overlap this chunk in XY (in file-naming convention)
    tile_x0 = x0 // ts + tile_base
    tile_x1 = max(x1 - 1, 0) // ts + tile_base
    tile_y0 = y0 // ts + tile_base
    tile_y1 = max(y1 - 1, 0) // ts + tile_base

    if is_em:
        chunk_data = np.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=np.uint8)
    else:
        chunk_data = np.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=np.uint32)

    # Choose tile path function
    if pattern == 'em_mip0':
        tile_path_fn = em_mip0_tile_path
    elif pattern == 'seg_mip0':
        tile_path_fn = seg_mip0_tile_path
    elif pattern == 'em_mipN':
        tile_path_fn = em_mipN_tile_path
    elif pattern == 'seg_mipN':
        tile_path_fn = seg_mipN_tile_path
    else:
        return None

    for local_z in range(z0, z1):
        if local_z >= len(section_list):
            continue

        # Map Z index → VSVI section number
        section_num = section_list[local_z]
        if section_num not in section_info:
            continue

        dir_name, max_r, max_c = section_info[section_num]
        local_z_offset = local_z - z0

        for tx in range(tile_x0, tile_x1 + 1):
            if tx > max_c:
                continue
            for ty in range(tile_y0, tile_y1 + 1):
                if ty > max_r:
                    continue

                tile_path = tile_path_fn(mip_dir, dir_name, section_num, ty, tx)

                if not os.path.exists(tile_path):
                    continue

                try:
                    tile = Image.open(tile_path)
                    tile_arr = np.array(tile)
                except Exception:
                    continue

                # Global voxel range of this tile
                t_px0 = (tx - tile_base) * ts
                t_py0 = (ty - tile_base) * ts
                t_px1 = t_px0 + ts
                t_py1 = t_py0 + ts

                # Overlap region in chunk-local coordinates
                ox0 = max(x0, t_px0) - x0
                oy0 = max(y0, t_py0) - y0
                ox1 = min(x1, t_px1) - x0
                oy1 = min(y1, t_py1) - y0

                # Overlap region in tile-local pixel coordinates
                tx_p0 = max(x0, t_px0) - t_px0
                ty_p0 = max(y0, t_py0) - t_py0
                tx_p1 = min(x1, t_px1) - t_px0
                ty_p1 = min(y1, t_py1) - t_py0

                if is_em:
                    chunk_data[ox0:ox1, oy0:oy1, local_z_offset] = \
                        tile_arr[ty_p0:ty_p1, tx_p0:tx_p1]
                else:
                    # RGB → uint32: R + G*256 + B*65536 (common Vaa3D convention)
                    rgb = tile_arr[ty_p0:ty_p1, tx_p0:tx_p1, :]
                    vals = (rgb[:, :, 0].astype(np.uint32) +
                            rgb[:, :, 1].astype(np.uint32) * 256 +
                            rgb[:, :, 2].astype(np.uint32) * 65536)
                    chunk_data[ox0:ox1, oy0:oy1, local_z_offset] = vals

    # Write chunk file in unsharded precomputed format
    os.makedirs(chunk_dir, exist_ok=True)

    # Precomputed format requires fortran (column-major) order
    # Data layout: x fastest, z slowest → transpose from [x, y, z] to [z, y, x]
    data_f = np.transpose(chunk_data, (2, 1, 0)).tobytes()
    with open(chunk_path, 'wb') as f:
        f.write(data_f)

    return 'done'


# ============================================================
# Main conversion logic
# ============================================================

def convert_layer(layer_type, layer_dir_name, vsvi_meta, pattern_base,
                  input_dir, output_dir, chunk_size, num_workers,
                  force_overwrite=False):
    """
    Convert one layer (EM_img or seg) from VSVI to precomputed format.

    Parameters
    ----------
    layer_type : str
        'image' or 'segmentation'
    layer_dir_name : str
        'EM_img' or 'seg'
    vsvi_meta : dict
        Parsed VSVI metadata
    pattern_base : str
        'em' or 'seg'
    """
    tile_size = vsvi_meta['SourceTileSizeX']
    dim_x = vsvi_meta['TargetDataSizeX']
    dim_y = vsvi_meta['TargetDataSizeY']
    dim_z = vsvi_meta['TargetDataSizeZ']
    voxel_x = vsvi_meta.get('TargetVoxelSizeXnm', 1)
    voxel_y = vsvi_meta.get('TargetVoxelSizeYnm', 1)
    voxel_z = vsvi_meta.get('TargetVoxelSizeZnm', 1)
    bpp = vsvi_meta.get('SourceBytesPerPixel', 1)

    print(f"\n{'=' * 60}")
    print(f"Layer: {layer_dir_name} ({layer_type})")
    print(f"{'=' * 60}")
    print(f"  Dimensions:         {dim_x} x {dim_y} x {dim_z} voxels")
    print(f"  Voxel resolution:   {voxel_x}nm x {voxel_y}nm x {voxel_z}nm")
    print(f"  Tile size:          {tile_size} px")
    print(f"  Bytes per pixel:    {bpp}")
    print(f"  Chunk size:         {chunk_size[0]} x {chunk_size[1]} x {chunk_size[2]}")
    print(f"  Data type:          {'uint8' if pattern_base == 'em' else 'uint32 (from RGB)'}")
    print(f"  Mode:               {'force overwrite' if force_overwrite else 'resume (skip existing)'}")

    base_path = os.path.join(input_dir, layer_dir_name)
    out_path = os.path.join(output_dir, layer_dir_name)

    # Discover available mip levels
    mip_levels = []
    for m in range(10):
        if os.path.isdir(os.path.join(base_path, f'mip{m}')):
            mip_levels.append(m)

    print(f"  Available mip levels: {mip_levels}")

    scales = []

    for mip_level in mip_levels:
        print(f"\n  --- mip{mip_level} ---")
        mip_dir = os.path.join(base_path, f'mip{mip_level}')

        # Compute dimensions at this scale
        factor = 2 ** mip_level
        scale_x = max(1, dim_x // factor)
        scale_y = max(1, dim_y // factor)
        scale_z = max(1, dim_z // factor)

        # Determine pattern for this mip level
        if mip_level == 0:
            if pattern_base == 'em':
                pattern = 'em_mip0'
                sec_map_raw = discover_em_mip0_sections(mip_dir)
            else:
                pattern = 'seg_mip0'
                sec_map_raw = discover_seg_mip0_sections(mip_dir)
        else:
            if pattern_base == 'em':
                pattern = 'em_mipN'
                sec_map_raw = discover_em_mipN_sections(mip_dir)
            else:
                pattern = 'seg_mipN'
                sec_map_raw = discover_seg_mipN_sections(mip_dir)

        print(f"    Pattern: {pattern}")
        print(f"    Found {len(sec_map_raw)} section directories")

        # Discover tile ranges (samples first section, all sections share same grid)
        section_info = discover_tile_range(mip_dir, sec_map_raw, pattern)
        print(f"    Valid sections (with tiles): {len(section_info)}")

        if not section_info:
            print(f"    No valid sections, skipping")
            continue

        # Build ordered section list for Z-index → section-number mapping
        section_list = sorted(section_info.keys())

        # Use actual section count for Z dimension (more reliable than dividing metadata)
        scale_z = len(section_list)
        if scale_z != (dim_z // factor):
            print(f"    Note: metadata-based Z={dim_z // factor}, "
                  f"actual section count Z={scale_z}")

        # Verify dimensions from actual tiles
        if section_info:
            _, max_r, max_c = next(iter(section_info.values()))
            # Compute actual dimensions from tile grid
            # max_c/max_r are in file-naming convention (1-based for EM, 0-based for seg)
            tile_base = 1 if pattern.startswith('em') else 0
            actual_tiles_x = max_c - tile_base + 1  # number of tile columns
            actual_tiles_y = max_r - tile_base + 1  # number of tile rows
            actual_dim_x = actual_tiles_x * tile_size
            actual_dim_y = actual_tiles_y * tile_size
            if actual_dim_x != scale_x or actual_dim_y != scale_y:
                print(f"    Note: computed dims {scale_x}x{scale_y}, "
                      f"actual tile-based dims ~{actual_dim_x}x{actual_dim_y}")
                scale_x = min(scale_x, actual_dim_x)
                scale_y = min(scale_y, actual_dim_y)

        # Compute chunk grid
        csx, csy, csz = chunk_size
        grid_x = (scale_x + csx - 1) // csx
        grid_y = (scale_y + csy - 1) // csy
        grid_z = (scale_z + csz - 1) // csz
        num_chunks = grid_x * grid_y * grid_z

        print(f"    Scale dimensions: {scale_x} x {scale_y} x {scale_z}")
        print(f"    Chunk grid:       {grid_x} x {grid_y} x {grid_z} = {num_chunks} chunks")

        if num_chunks == 0:
            print(f"    No chunks to process, skipping")
            continue

        # Build task list (just chunk coordinates — shared data via fork)
        tasks = []
        for cx in range(grid_x):
            for cy in range(grid_y):
                for cz in range(grid_z):
                    tasks.append((cx, cy, cz))

        # Set up shared data (inherited by worker processes via fork on Linux)
        _PROCESS_SHARED.clear()
        _PROCESS_SHARED.update({
            'mip_dir': mip_dir,
            'section_info': section_info,
            'section_list': section_list,
            'chunk_sx': csx, 'chunk_sy': csy, 'chunk_sz': csz,
            'tile_size': tile_size,
            'dim_x': scale_x, 'dim_y': scale_y, 'dim_z': scale_z,
            'pattern': pattern,
            'out_dir': out_path,
            'scale_key': str(mip_level),
            'is_em': pattern.startswith('em'),
            'tile_base': 1 if pattern.startswith('em') else 0,
            'force': force_overwrite,
        })

        # Process in parallel
        done = 0
        skipped = 0
        exists = 0
        t_start = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_chunk, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result == 'done':
                        done += 1
                    elif result == 'exists':
                        exists += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"      Error: {e}")
                    skipped += 1
                total = done + skipped + exists
                if total % 500 == 0 or total == num_chunks:
                    elapsed = time.time() - t_start
                    rate = total / elapsed if elapsed > 0 else 0
                    eta = (num_chunks - total) / rate if rate > 0 else 0
                    print(f"      {total}/{num_chunks} "
                          f"({done} new, {exists} existing, {skipped} skipped) "
                          f"| {rate:.1f} chunk/s | "
                          f"elapsed: {elapsed / 60:.1f}min "
                          f"ETA: {eta / 60:.1f}min")

        elapsed = time.time() - t_start
        total_done = done + exists + skipped
        rate = total_done / elapsed if elapsed > 0 else 0
        print(f"    Finished in {elapsed / 60:.1f} minutes ({rate:.1f} chunk/s) "
              f"— new: {done}, existing: {exists}, skipped: {skipped}")

        # Add scale to info
        scales.append({
            'key': str(mip_level),
            'size': [scale_x, scale_y, scale_z],
            'resolution': [voxel_x * factor, voxel_y * factor, voxel_z * factor],
            'voxel_offset': [0, 0, 0],
            'chunk_sizes': [[csx, csy, csz]],
            'encoding': 'raw'
        })

    # Sort scales by resolution ascending (finest first, as required by neuroglancer spec)
    scales.sort(key=lambda s: (s['resolution'][0], s['resolution'][1], s['resolution'][2]))

    # Write info file
    data_type = 'uint8' if pattern_base == 'em' else 'uint32'
    info = {
        '@type': 'neuroglancer_multiscale_volume',
        'type': layer_type,
        'data_type': data_type,
        'num_channels': 1,
        'scales': scales
    }

    os.makedirs(out_path, exist_ok=True)
    info_path = os.path.join(out_path, 'info')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n  Info file written: {info_path}")
    print(f"  URL: precomputed://file://{os.path.abspath(out_path)}")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert Vaa3D VSVI tiled dataset to neuroglancer precomputed format'
    )
    parser.add_argument('input_dir',
                        help='Path to dataset directory (containing EM_img/ and seg/)')
    parser.add_argument('output_dir',
                        help='Path to output directory for precomputed data')
    parser.add_argument('--chunk-size', type=int, default=128,
                        help='Chunk edge size in voxels (default: 128)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing chunks (disable resume)')
    parser.add_argument('--only-em', action='store_true',
                        help='Only convert EM_img layer')
    parser.add_argument('--only-seg', action='store_true',
                        help='Only convert segmentation layer')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    chunk_size = (args.chunk_size, args.chunk_size, args.chunk_size)
    num_workers = args.workers

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # --- EM_img (image layer) ---
    if not args.only_seg:
        em_vsvi = os.path.join(input_dir, 'EM_img', 'IARPA_JWR15.vsvi')
        if os.path.exists(em_vsvi):
            em_meta = parse_vsvi(em_vsvi)
            convert_layer('image', 'EM_img', em_meta, 'em',
                          input_dir, output_dir, chunk_size, num_workers,
                          force_overwrite=args.force)
        else:
            print(f"Warning: EM_img VSVI not found at {em_vsvi}")

    # --- seg (segmentation layer) ---
    if not args.only_em:
        seg_vsvi = os.path.join(input_dir, 'seg', 'IARPA_JWR15_SEG.vsvi')
        if os.path.exists(seg_vsvi):
            seg_meta = parse_vsvi(seg_vsvi)
            convert_layer('segmentation', 'seg', seg_meta, 'seg',
                          input_dir, output_dir, chunk_size, num_workers,
                          force_overwrite=args.force)
        else:
            print(f"Warning: seg VSVI not found at {seg_vsvi}")

    # --- Write top-level neuroglancer state ---
    state = {
        'dimensions': {
            'x': [4e-9, 'm'],
            'y': [4e-9, 'm'],
            'z': [3e-8, 'm'],
        },
        'position': [13312e-9, 13312e-9, 1546e-8],
        'crossSectionScale': 1,
        'projectionOrientation': [0, 1, 0, 0, 0, 0, 1, 0, 0],
        'projectionScale': 2048,
        'layers': []
    }

    if not args.only_seg:
        state['layers'].append({
            'type': 'image',
            'name': 'EM_img',
            'source': f"precomputed://file://{os.path.abspath(os.path.join(output_dir, 'EM_img'))}",
            'opacity': 1.0,
            'blend': 'default',
            'shader': 'void main() { emitRGB(vec3(toNormalized(getDataValue()))); }',
            'visible': True
        })

    if not args.only_em:
        state['layers'].append({
            'type': 'segmentation',
            'name': 'seg',
            'source': f"precomputed://file://{os.path.abspath(os.path.join(output_dir, 'seg'))}",
            'opacity': 0.5,
            'visible': True
        })

    state_path = os.path.join(output_dir, 'neuroglancer_state.json')
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"State file: {state_path}")
    print(f"\nTo view in neuroglancer, open:")
    print(f"  http://localhost:8080/#!{json.dumps(state)}")
    print(f"\nOr serve with:")
    print(f"  cd {os.path.abspath(output_dir)} && python -m http.server 8080")
    print(f"Then open: http://localhost:8080/neuroglancer_state.json")


if __name__ == '__main__':
    main()
