import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- Corrected Data Definition (scaled to 640x360 image) ---

# 1. Bounding Box Data: [ymin, xmin, ymax, xmax] format
bounding_boxes = [
    {"box": [290, 110, 330, 190], "label": "Rubik's Cube"},
    {"box": [280, 170, 330, 480], "label": "Hammer"},
    {"box": [80, 520, 140, 590], "label": "Plates/Bowls"},
    {"box": [0, 0, 360, 400], "label": "Robotic arm"},
    {"box": [180, 430, 250, 510], "label": "Gripper/End effector"}
]

# 2. Sparse Keypoints Trajectory Data: [y, x] format
# Hammer Head Trajectory (Sparse)
head_trajectory_sparse = [
    [290, 470],  # P1: Start Position (Resting near target)
    [100, 300],  # P2: Pre-strike Position (Apex of swing)
    [290, 150],  # P3: Impact Position (Striking Rubik's Cube)
    [310, 100]   # P4: Follow-through Position (Post-impact)
]

# Hammer Handle Trajectory (Sparse)
handle_trajectory_sparse = [
    [310, 250],  # P1: Start Position (Grasp point)
    [120, 80],   # P2: Pre-strike Position (Grasp point)
    [310, -70],  # P3: Impact Position (Grasp point, off-screen)
    [330, -120]  # P4: Follow-through Position (Grasp point, off-screen)
]

# 3. Dense Keypoints Trajectory Data: [y, x] format, interpolated between sparse points
head_trajectory_dense = [
    # Interpolation P1 -> P2 (Lifting and pulling back)
    [290, 470], [245, 420], [200, 370], [150, 335], [100, 300],
    # Interpolation P2 -> P3 (Swing to impact)
    [140, 260], [180, 220], [230, 185], [290, 150],
    # Interpolation P3 -> P4 (Follow-through)
    [300, 125], [310, 100]
]

handle_trajectory_dense = [
    # Interpolation P1 -> P2 (Lifting and pulling back)
    [310, 250], [285, 200], [250, 150], [190, 115], [120, 80],
    # Interpolation P2 -> P3 (Swing to impact)
    [160, 40], [200, 0], [250, -35], [310, -70],
    # Interpolation P3 -> P4 (Follow-through)
    [320, -95], [330, -120]
]

# --- Visualization Functions ---

def load_image(image_path="hammer.png"):
    """Loads the original image."""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'. Please replace '{image_path}' with the correct path to your image.")
        return None
    return img

def visualize_bounding_boxes(img, boxes):
    """Draws bounding boxes on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    for box_data in boxes:
        ymin, xmin, ymax, xmax = box_data['box']
        label = box_data['label']

        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')

        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, label, color='g', fontsize=12, backgroundcolor='k')

    ax.set_title("1. Bounding Boxes on Objects (Corrected)")
    ax.axis('off')
    plt.show()

def visualize_sparse_trajectory(img, head_traj, handle_traj):
    """Draws sparse keypoints and connections on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    head_x = [p[1] for p in head_traj]
    head_y = [p[0] for p in head_traj]
    handle_x = [p[1] for p in handle_traj]
    handle_y = [p[0] for p in handle_traj]

    # Plot hammer head path (blue line, blue dots)
    ax.plot(head_x, head_y, color='blue', linestyle='--', marker='o', markersize=8, label='Hammer Head Path')
    # Plot hammer handle path (red line, red dots)
    ax.plot(handle_x, handle_y, color='red', linestyle='--', marker='o', markersize=8, label='Hammer Handle Path')

    # Add labels for key points
    ax.text(head_x[0], head_y[0] + 10, "Start", color='blue', fontsize=12, backgroundcolor='w')
    ax.text(head_x[1], head_y[1] - 10, "Apex", color='blue', fontsize=12, backgroundcolor='w')
    ax.text(head_x[2] + 10, head_y[2] + 10, "Impact", color='blue', fontsize=12, backgroundcolor='w')
    ax.text(head_x[3] + 10, head_y[3] + 10, "Follow-through", color='blue', fontsize=12, backgroundcolor='w')

    ax.set_title("2. Sparse Keypoints Trajectory (Corrected)")
    ax.axis('off')
    plt.show()

def visualize_dense_trajectory(img, head_traj, handle_traj):
    """Draws dense trajectory path on the image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    head_x = [p[1] for p in head_traj]
    head_y = [p[0] for p in head_traj]
    handle_x = [p[1] for p in handle_traj]
    handle_y = [p[0] for p in handle_traj]

    ax.plot(head_x, head_y, color='blue', linewidth=3, label='Hammer Head Path')
    ax.plot(handle_x, handle_y, color='red', linewidth=3, label='Hammer Handle Path')

    ax.set_title("3. Dense Trajectory Path (Corrected)")
    ax.axis('off')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    img = load_image()
    if img:
        visualize_bounding_boxes(img, bounding_boxes)
        visualize_sparse_trajectory(img, head_trajectory_sparse, handle_trajectory_sparse)
        visualize_dense_trajectory(img, head_trajectory_dense, handle_trajectory_dense)