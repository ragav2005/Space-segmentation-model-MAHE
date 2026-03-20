import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import RANSACRegressor

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config

# Global variables for workers (prevents massive NuScenes re-initialization)
nusc = None
nusc_maps = {}

def init_worker(version, dataroot):
    """Initialize NuScenes and map objects globally for worker processes."""
    global nusc, nusc_maps
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    map_names = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
    for map_name in map_names:
        nusc_maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)

def densify_lidar_depth(sparse_map, max_val, min_val):
    """Densify sparse LiDAR depth using inverted dilation."""
    valid_mask = (sparse_map > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inverted = np.where(valid_mask > 0, max_val - sparse_map, 0.0)
    dilated_inverted = cv2.dilate(inverted, kernel, iterations=1)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    blurred_inverted = cv2.GaussianBlur(dilated_inverted, (3, 3), 0)
    recovered_dense_map = np.where(dilated_mask > 0, max_val - blurred_inverted, 0.0)
    
    return np.clip(recovered_dense_map, min_val, max_val)

def densify_lidar_height(sparse_map, max_val, min_val):
    """Densify sparse LiDAR height using bilateral filtering and dilation."""
    valid_mask = (sparse_map != -100.0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    active_points = np.where(valid_mask > 0, sparse_map, -100.0)
    dilated_map = cv2.dilate(active_points, kernel, iterations=1)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    blurred_map = cv2.bilateralFilter(dilated_map.astype(np.float32), 5, 25, 25)
    clipped_valid = np.clip(blurred_map, min_val, max_val)
    dense_map = np.where(dilated_mask > 0, clipped_valid, -100.0)
    
    return dense_map

def process_single_sample(sample_token, camera_name, output_dir):
    """Process a single sample: project map, extract depth/height from LiDAR, generate mask."""
    global nusc, nusc_maps
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][camera_name])
    lidar_data = nusc.get('sample_data', sample['data'][config.LIDAR_SENSOR])
    img_path = str(Path(nusc.dataroot) / cam_data['filename'])
    img = cv2.imread(img_path)
    if img is None:
        return False, f"Failed to load image: {img_path}"
    
    original_h, original_w = img.shape[:2]
    img_resized = cv2.resize(img, (config.IMG_SIZE[1], config.IMG_SIZE[0]))
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    nusc_map = nusc_maps[log['location']]
    cam_ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    box_coords = (cam_ego_pose['translation'][0], cam_ego_pose['translation'][1], 60, 60)
    records = nusc_map.get_records_in_patch(box_coords, ['drivable_area'], mode='intersect')
    
    mask = np.zeros((original_h, original_w), dtype=np.uint8)
    
    for token in records['drivable_area']:
        record = nusc_map.get('drivable_area', token)
        for polygon_token in record['polygon_tokens']:
            polygon = nusc_map.get('polygon', polygon_token)
            
            nodes = np.array([[nusc_map.get('node', n)['x'], nusc_map.get('node', n)['y']] for n in polygon['exterior_node_tokens']])
            if len(nodes) < 3:
                continue
            ground_z = cam_ego_pose['translation'][2] - 1.84 
            nodes_3d = np.hstack((nodes, np.ones((nodes.shape[0], 1)) * ground_z))
            nodes_3d -= np.array(cam_ego_pose['translation'])
            nodes_3d = nodes_3d @ Quaternion(cam_ego_pose['rotation']).rotation_matrix
            nodes_3d -= np.array(cam_calib['translation'])
            nodes_3d = nodes_3d @ Quaternion(cam_calib['rotation']).rotation_matrix
            nodes_3d = nodes_3d.T
            if np.any(nodes_3d[2, :] < 0.1):
                nodes_3d[2, nodes_3d[2, :] < 0.1] = 0.1
            if nodes_3d.shape[1] < 3:
                continue
            points_2d = view_points(nodes_3d, np.array(cam_calib['camera_intrinsic']), normalize=True)
            pts = points_2d[:2, :].T.astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], -16384, 16384)
            pts[:, 1] = np.clip(pts[:, 1], -16384, 16384)
            
            cv2.fillPoly(mask, [pts], 1)
            
    mask_resized = cv2.resize(mask, (config.IMG_SIZE[1], config.IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    pcl, _ = LidarPointCloud.from_file_multisweep(nusc, sample, config.LIDAR_SENSOR, config.LIDAR_SENSOR, nsweeps=3)
    
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pcl.rotate(Quaternion(lidar_calib['rotation']).rotation_matrix)
    pcl.translate(np.array(lidar_calib['translation']))
    ego_pts = pcl.points.copy()
    pcl.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pcl.translate(np.array(lidar_ego_pose['translation']))
    pcl.translate(-np.array(cam_ego_pose['translation']))
    pcl.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)
    pcl.translate(-np.array(cam_calib['translation']))
    pcl.rotate(Quaternion(cam_calib['rotation']).rotation_matrix.T)
    
    depths = pcl.points[2, :]
    points = view_points(pcl.points[:3, :], np.array(cam_calib['camera_intrinsic']), normalize=True)
    
    mask_bounds = np.ones(depths.shape[0], dtype=bool)
    mask_bounds = np.logical_and(mask_bounds, depths > 0)
    mask_bounds = np.logical_and(mask_bounds, points[0, :] > 1)
    mask_bounds = np.logical_and(mask_bounds, points[0, :] < original_w - 1)
    mask_bounds = np.logical_and(mask_bounds, points[1, :] > 1)
    mask_bounds = np.logical_and(mask_bounds, points[1, :] < original_h - 1)
    
    points = points[:, mask_bounds]
    depths = depths[mask_bounds]
    
    valid_ego = ego_pts[:, mask_bounds]
    z_mask = (valid_ego[2, :] > -2.5) & (valid_ego[2, :] < -1.0)
    
    if np.sum(z_mask) > 10:
        ground_pts = valid_ego[:, z_mask]
        X = ground_pts[:2, :].T
        y = ground_pts[2, :]
        try:
            ransac = RANSACRegressor(residual_threshold=0.2)
            ransac.fit(X, y)
            predicted_z = ransac.predict(valid_ego[:2, :].T)
            heights = valid_ego[2, :] - predicted_z
        except Exception:
            heights = valid_ego[2, :] + 1.84
    else:
        heights = valid_ego[2, :] + 1.84
    
    sparse_depth = np.zeros((original_h, original_w))
    sparse_height = np.ones((original_h, original_w)) * -100.0
    points_x, points_y = np.rint(points[0, :]).astype(np.int32), np.rint(points[1, :]).astype(np.int32)
    sparse_depth[points_y, points_x] = depths
    sparse_height[points_y, points_x] = heights
    if config.LIDAR_DENSIFICATION:
        opt_depth = densify_lidar_depth(sparse_depth, config.MAX_DEPTH, config.MIN_DEPTH)
        opt_height = densify_lidar_height(sparse_height, config.MAX_HEIGHT, config.MIN_HEIGHT)
    else:
        opt_depth = sparse_depth
        opt_height = np.where(sparse_height == -100.0, 0.0, sparse_height)
    depth_resized = cv2.resize(opt_depth, (config.IMG_SIZE[1], config.IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    height_resized = cv2.resize(opt_height, (config.IMG_SIZE[1], config.IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    
    # Use raw map polygon mask directly - model learns obstacle boundaries
    # from the height input channel (5-ch: RGB+D+H)
    mask_clean = mask_resized

    depth_16bit = np.clip((depth_resized / config.MAX_DEPTH) * 65535, 0, 65535).astype(np.uint16)
    height_scaled = (np.clip(height_resized, config.MIN_HEIGHT, config.MAX_HEIGHT) - config.MIN_HEIGHT) / (config.MAX_HEIGHT - config.MIN_HEIGHT)
    height_16bit = (height_scaled * 65535).astype(np.uint16)
    out_dir = Path(output_dir) / camera_name
    img_out = out_dir / 'images' / f"{sample_token}.png"
    mask_out = out_dir / 'masks' / f"{sample_token}.png"
    depth_out = out_dir / 'depth' / f"{sample_token}.png"
    height_out = out_dir / 'height' / f"{sample_token}.png"
    cv2.imwrite(str(img_out), img_resized)
    cv2.imwrite(str(mask_out), mask_clean * 255)
    cv2.imwrite(str(depth_out), depth_16bit)
    cv2.imwrite(str(height_out), height_16bit)

    return True, sample_token

def run_multimodal_preprocessing():
    print("Processing multi-modal dataset...")
    local_nusc = NuScenes(version='v1.0-mini', dataroot=str(config.DATA_ROOT), verbose=False)
    tasks = []
    for sample in local_nusc.sample:
        for cam in config.CAMERAS:
            tasks.append((sample['token'], cam, str(config.DATASET_DIR)))
    total_tasks = len(tasks)
    local_nusc = None
    print(f"Total samples: {total_tasks}")
    print(f"Workers: {config.NUM_WORKERS}")
    successful, failed = 0, 0
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS, initializer=init_worker, initargs=('v1.0-mini', str(config.DATA_ROOT))) as executor:
        futures = [executor.submit(process_single_sample, *t) for t in tasks]
        with tqdm(total=total_tasks, desc="Processing", unit="img") as pbar:
            for future in as_completed(futures):
                try:
                    success, msg = future.result()
                    if success: successful += 1
                    else: 
                        failed += 1
                        print(msg)
                except Exception as e:
                    failed += 1
                    print(f"Error: {e}")
                pbar.set_postfix({"Success": successful, "Failed": failed})
                pbar.update(1)
    print(f"\nComplete: {successful} successful, {failed} failed.")

if __name__ == "__main__":
    run_multimodal_preprocessing()
