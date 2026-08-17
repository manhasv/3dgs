from pathlib import Path
import shutil
import pycolmap

PROJECT = Path(__file__).parent.resolve()

IMAGE_DIR = PROJECT / "images"
DATABASE = PROJECT / "database.db"
SPARSE_DIR = PROJECT / "sparse"

# Clean up
if DATABASE.exists():
    DATABASE.unlink()

if SPARSE_DIR.exists():
    shutil.rmtree(SPARSE_DIR)

SPARSE_DIR.mkdir()

print("=== Extracting features ===")

extraction_options = pycolmap.FeatureExtractionOptions() # Using SIFT as default, providing better accuracy for dense reconstruction
extraction_options.use_gpu = True # Use GPU if available

pycolmap.extract_features(
    database_path=DATABASE,
    image_path=IMAGE_DIR,
    camera_mode=pycolmap.CameraMode.SINGLE, # assume same camera for all images
    extraction_options=extraction_options,
    device=pycolmap.Device.auto,
)

print("=== Matching images ===")
# Since the images are extracted from a video, use match_sequential, else there are other options for this
pycolmap.match_sequential(
    database_path=DATABASE,
    device=pycolmap.Device.auto,
)

print("=== Running incremental mapping ===")
# This build the model incrementally
reconstructions = pycolmap.incremental_mapping(
    database_path=DATABASE,
    image_path=IMAGE_DIR,
    output_path=SPARSE_DIR,
)

"""
output contains:
+ cameras.bin: : Lists intrinsic camera parameters, such as camera ID, model type (e.g., PINHOLE, SIMPLE_RADIAL), width, height, and focal length/distortion coefficients
+ frames.bin   : Also a multi-camera rig param
+ images.bin   : Contains extrinsic parameters for every registered image, including image ID, rotation quaternion (qw, qx, qy, qz), translation vector (tx, ty, tz), camera ID, file name, and 2D-3D point correspondences
+ points3D.bin : Stores the sparse 3D point cloud data, detailing the 3D point ID, XYZ coordinates, RGB color triplet, tracking error, and the list of 2D image observations linked to it
+ rigs.bin     : Optional configuration file tracking multi-camera rig relative parameters if a rig setup was specified during processing
"""

print("\n=== Reconstruction results ===")

if not reconstructions:
    print("No reconstruction was created.")
else:
    for model_id, reconstruction in reconstructions.items():
        print(f"\nModel {model_id}")
        print(reconstruction.summary())