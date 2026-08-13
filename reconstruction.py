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

print("\n=== Reconstruction results ===")

if not reconstructions:
    print("No reconstruction was created.")
else:
    for model_id, reconstruction in reconstructions.items():
        print(f"\nModel {model_id}")
        print(reconstruction.summary())