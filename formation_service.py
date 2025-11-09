# backend/formation_service.py
import os
import numpy as np
import cv2

def process_logo(image_path: str, num_points: int = 5, sim_width: float = 10.0, sim_height: float = 10.0, hover_z: float = 1.0):
    """
    Read an image, extract N points from logo contours, normalize them into simulation coordinates,
    save as a .npy file and return (output_path, coords_array).
    """
    # --- load and basic checks
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Resize (fixed size for consistent extraction)
    img = cv2.resize(img, (200, 200))

    # Invert and binarize so logo is white on black
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for c in contours:
        # flatten contour points
        for pt in c:
            points.append(pt[0])

    if len(points) == 0:
        raise ValueError("No logo contours/points found. Try another image or adjust threshold.")

    # Pick evenly spaced points along concatenated contour points
    if len(points) > num_points:
        idx = np.linspace(0, len(points) - 1, num_points, dtype=int)
        points = [points[i] for i in idx]
    else:
        points = points[:num_points]

    points = np.array(points, dtype=np.float32)  # shape (N, 2)

    # Normalize coordinates to 0..1 by bounding box, then scale to sim_width / sim_height
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = max_xy - min_xy
    # Avoid division by zero
    span[span == 0] = 1.0
    normalized = (points - min_xy) / span

    scaled = np.zeros((normalized.shape[0], 3), dtype=np.float32)
    scaled[:, 0] = normalized[:, 0] * sim_width
    # Flip Y so top of image maps to high Y in sim (PyBullet typical)
    scaled[:, 1] = (1.0 - normalized[:, 1]) * sim_height
    scaled[:, 2] = hover_z

    # Save file into backend/data
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    basename = os.path.basename(image_path)
    out_path = os.path.join(os.path.dirname(__file__), "data", f"drone_targets_{basename}.npy")
    np.save(out_path, scaled)

    return out_path, scaled
