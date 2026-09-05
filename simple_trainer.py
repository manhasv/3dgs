import math
import cv2
import torch
import numpy as np
import pycolmap
from pathlib import Path
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from gsplat import rasterization

"""
This is a simple 3DGS optimizer for prototype purposes. This is very unoptimized for heavy work.
"""

PROJECT = Path(__file__).parent.resolve()
IMAGE_DIR = PROJECT / "images"
SPARSE_DIR = PROJECT / "sparse" / "0"  # Colmap outputs to subfolder '0' by default
OUTPUT_DIR = PROJECT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("WARNING: CUDA not found. gsplat requires a GPU to run efficiently.")

print(f"=== Loading Colmap Reconstruction from {SPARSE_DIR} ===")
reconstruction = pycolmap.Reconstruction(SPARSE_DIR)

# Extract 3D Points (Means) and Colors
pts3d = reconstruction.points3D
xyz = np.array([p.xyz for p in pts3d.values()])
rgb = np.array([p.color for p in pts3d.values()]) / 255.0

print(f"Loaded {len(xyz)} sparse points.")

print("=== Initializing Gaussian Tensors ===")

# Position (Means)
means = torch.tensor(xyz, dtype=torch.float32, device=device).requires_grad_(True)

# Size (Scales) - initialized very small, optimized in log-space
scales = torch.full((len(xyz), 3), math.log(0.01), device=device).requires_grad_(True)

# Rotation (Quaternions) - initialized as identity [1, 0, 0, 0]
quats = torch.zeros((len(xyz), 4), device=device)
quats[:, 0] = 1.0
quats.requires_grad_(True)

# Opacity - initialized to 0.1, optimized in inverse-sigmoid space
opacities = torch.full((len(xyz),), 0.1, device=device)
opacities = torch.logit(opacities).requires_grad_(True)

# Color - optimizing raw RGB in inverse-sigmoid space for this basic version
colors = torch.tensor(rgb, dtype=torch.float32, device=device)
colors = torch.logit(torch.clamp(colors, 1e-4, 1.0 - 1e-4)).requires_grad_(True)

print("=== Loading Cameras and Images ===")
cameras = []

for image_id, image in reconstruction.images.items():
    cam = reconstruction.cameras[image.camera_id]
    
    # Read Ground Truth image
    img_path = IMAGE_DIR / image.name
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_tensor = torch.from_numpy(img).float()
    
    # View Matrix (World-to-Camera)
    # Handle pycolmap version differences for extrinsics
    pose = getattr(image, "cam_from_world", None)
    
    if pose is not None:
        if callable(pose):
            pose = pose()  # Call it if it's a method
        R = pose.rotation.matrix()
        t = pose.translation
    else:
        # Fallback for older versions
        R = image.rotmat()
        t = image.tvec
        
    viewmat = np.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device)
    
    # Intrinsic Matrix (K)
    K = np.eye(3)
    # Safely extract focal length and principal points based on model
    params = cam.params
    if cam.model_name in ["PINHOLE", "OPENCV"]:
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = params[0], params[1], params[2], params[3]
    else: # SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = params[0], params[0], params[1], params[2]
        
    K = torch.tensor(K, dtype=torch.float32, device=device)
    
    cameras.append({
        "name": image.name,
        "img": img_tensor,
        "viewmat": viewmat,
        "K": K,
        "width": cam.width,
        "height": cam.height
    })

print(f"Loaded {len(cameras)} valid cameras.")

print("=== Starting Optimization ===")

# Adam optimizer
optimizer = torch.optim.Adam([
    {'params': [means], 'lr': 1.6e-4},
    {'params': [scales], 'lr': 5.0e-3},
    {'params': [quats], 'lr': 1.0e-3},
    {'params': [opacities], 'lr': 5.0e-2},
    {'params': [colors], 'lr': 2.5e-3},
])

ITERATIONS = 4000

for step in range(1, ITERATIONS + 1):
    # Pick a random camera
    idx = np.random.randint(len(cameras))
    cam = cameras[idx]
    
    # gsplat expects batched camera matrices [Batch, ...]
    viewmats = cam["viewmat"].unsqueeze(0)
    Ks = cam["K"].unsqueeze(0)
    
    # Normalize inputs for the rasterizer
    norm_quats = F.normalize(quats, dim=-1)
    act_scales = torch.exp(scales)
    act_opacities = torch.sigmoid(opacities)
    act_colors = torch.sigmoid(colors)
    
    # Forward Pass, Render the image
    renders, _, _ = rasterization(
        means=means,
        quats=norm_quats,
        scales=act_scales,
        opacities=act_opacities,
        colors=act_colors,
        viewmats=viewmats,
        Ks=Ks,
        width=cam["width"],
        height=cam["height"],
    )
    
    # The output is [Batch, Height, Width, Channels]. Extract the first (and only) image.
    pred_img = renders[0]
    gt_img = cam["img"].to(device)
    
    # Compute L1 Loss
    loss = F.l1_loss(pred_img, gt_img)
    
    # Backward Pass & Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Logging & Output
    if step % 100 == 0 or step == 1:
        print(f"Step {step:04d} | Loss: {loss.item():.4f}")
        
    if step % 500 == 0:
        # Save a debug image to see progress
        out_path = OUTPUT_DIR / f"render_step_{step:04d}.png"
        img_np = (pred_img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), img_np)

print(f"\nTraining complete. Check the '{OUTPUT_DIR.name}' folder for rendered images.")

print("\n=== Exporting 3D Gaussian Splat (.ply) ===")

# 1. Detach tensors from GPU and convert to numpy
final_means = means.detach().cpu().numpy()
final_scales = scales.detach().cpu().numpy()
final_opacities = opacities.detach().cpu().numpy()

# Normalize quaternions
final_quats = F.normalize(quats, dim=-1).detach().cpu().numpy()

# 2. Convert Sigmoid RGB back to Spherical Harmonics (Degree 0)
# Standard 3DGS viewers expect colors as SH coefficients, not raw RGB.
# The formula for Base SH (DC) is: RGB = SH_DC * 0.28209 + 0.5
SH_C0 = 0.28209479177387814
final_rgb = torch.sigmoid(colors).detach().cpu().numpy()
final_sh_dc = (final_rgb - 0.5) / SH_C0

# 3. Construct the structured numpy array expected by standard viewers
num_pts = final_means.shape[0]

# Define the exact data types and property names required by 3DGS viewers
dtype_full = [
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),       # Position
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),    # Normals (unused, but required)
    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), # Colors (SH)
    ('opacity', 'f4'),                           # Opacity (pre-activation)
    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'), # Scale (pre-activation)
    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4') # Rotation (w, x, y, z)
]

elements = np.empty(num_pts, dtype=dtype_full)

# 4. Populate the array
elements['x'] = final_means[:, 0]
elements['y'] = final_means[:, 1]
elements['z'] = final_means[:, 2]

# Normals are typically zeroed out in 3DGS
elements['nx'] = np.zeros(num_pts)
elements['ny'] = np.zeros(num_pts)
elements['nz'] = np.zeros(num_pts)

elements['f_dc_0'] = final_sh_dc[:, 0]
elements['f_dc_1'] = final_sh_dc[:, 1]
elements['f_dc_2'] = final_sh_dc[:, 2]

elements['opacity'] = final_opacities
elements['scale_0'] = final_scales[:, 0]
elements['scale_1'] = final_scales[:, 1]
elements['scale_2'] = final_scales[:, 2]

elements['rot_0'] = final_quats[:, 0]
elements['rot_1'] = final_quats[:, 1]
elements['rot_2'] = final_quats[:, 2]
elements['rot_3'] = final_quats[:, 3]

# 5. Save
ply_path = OUTPUT_DIR / "splat_model.ply"
el = PlyElement.describe(elements, 'vertex')
PlyData([el]).write(str(ply_path))

print(f"Success! 3D model saved to: {ply_path}")
print("You can now drag and drop this .ply file into WebGL viewers like SuperSplat.")