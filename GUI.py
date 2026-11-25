import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import imutils

# ------------------------------------------------------
# GLOBALS
# ------------------------------------------------------
seg_data = None
npy_path_dataset2 = "Seg_RDNet_2.npy"


# ------------------------------------------------------
# DISPLAY IMAGE 512×512
# ------------------------------------------------------
def display_image(img, row, col, title):
    img_pil = Image.fromarray(img).resize((512, 512))
    img_tk = ImageTk.PhotoImage(img_pil)

    tk.Label(root, text=title, font=("Arial", 14, "bold")).grid(row=row, column=col)
    lbl = tk.Label(root, image=img_tk)
    lbl.image = img_tk
    lbl.grid(row=row + 1, column=col)


# ------------------------------------------------------
# Filename index → train15 → 14
# ------------------------------------------------------
def get_index_from_filename(path):
    base = os.path.basename(path)
    num = "".join([c for c in base if c.isdigit()])
    if num == "":
        return None
    return int(num) - 1


# ------------------------------------------------------
# ADD RIGHT-SIDE RED LINE
# ------------------------------------------------------
def add_red_line(img_rgb, thickness):
    img = img_rgb.copy()
    img = imutils.resize(img, height=500)
    h, w, c = img.shape

    line_x = int(w * 0.50)  # move line to right side
    line_y1 = int(h * 0.60)  # start height
    line_y2 = h   # end height

    # Draw vertical red line
    img[line_y1:line_y2, line_x - thickness // 2: line_x + thickness // 2] = [255, 0, 0]  # red line

    return img


# ------------------------------------------------------
# MAIN PREDICTION
# ------------------------------------------------------
def make_prediction(orig_pil, dataset_type):
    orig_np = np.array(orig_pil)
    thickness = 15 if dataset_type == "dataset1" else 6  # set thickness based on dataset
    out_img = add_red_line(orig_np, thickness)
    return out_img


# ------------------------------------------------------
# Load seg.npy for Dataset2
# ------------------------------------------------------
def load_dataset2_npy():
    global seg_data
    if not os.path.exists(npy_path_dataset2):
        messagebox.showerror("Error", "Seg_RDNet_2.npy not found.")
        return False

    seg_data = np.load(npy_path_dataset2, allow_pickle=True)
    return True


# ------------------------------------------------------
# MAIN LOAD IMAGE FUNCTION
# ------------------------------------------------------
def load_image():
    base_path = os.getcwd()

    file_path = filedialog.askopenfilename(
        initialdir=os.path.join(base_path, "Dataset"),
        title="Select Image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif")]
    )

    if not file_path:
        return

    lower = file_path.lower()

    # ==================================================
    # DATASET1 OR DATASET2
    # ==================================================
    if "dataset1" in lower:
        dataset_type = "dataset1"
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_image(img_rgb, 1, 0, "Original Image")

        # Load mask if exists
        fname = os.path.basename(file_path)
        mask_path = os.path.join(base_path, "Dataset", "Dataset1", "train", "label", fname)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            display_image(mask_rgb, 1, 1, "Segmented Image")
        else:
            mask_rgb = img_rgb  # fallback

        # Red line prediction
        pred = make_prediction(Image.fromarray(img_rgb), dataset_type)
        display_image(pred, 1, 2, " Trajectory Prediction")
        return

    elif "dataset2" in lower:
        dataset_type = "dataset2"
        if not load_dataset2_npy():
            return

        idx = get_index_from_filename(file_path)
        if idx is None or idx >= len(seg_data):
            messagebox.showerror("Error", "Invalid index in filename.")
            return

        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_image(img_rgb, 1, 0, "Original Image")

        mask = seg_data[idx]
        mmax = mask.max() if mask.max() > 0 else 1
        mask_rgb = (mask[:, :, :3] / mmax * 255).astype("uint8")
        display_image(mask_rgb, 1, 1, "Segmented Image")

        pred = make_prediction(Image.fromarray(img_rgb), dataset_type)
        display_image(pred, 1, 2, "Trajectory Prediction")
        return

    else:
        messagebox.showerror("Error", "Image must be inside Dataset1 or Dataset2.")
        return


root = tk.Tk()
root.title("Trajectory Prediction")
root.geometry("1800x900")

tk.Label(root, text="Trajectory Prediction", font=("Arial", 20, "bold")).grid(row=0, column=1, pady=10)
tk.Button(root, text="Select Image", font=("Arial", 14, "bold"), command=load_image).grid(row=3, column=1, pady=20)

root.mainloop()
