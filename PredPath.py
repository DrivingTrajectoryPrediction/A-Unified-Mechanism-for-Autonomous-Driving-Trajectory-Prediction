import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("./Dataset/Dataset1/train/img/train10.png")
out = image.copy()

# -----------------------------
# 1. Preprocess image
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

h, w = edges.shape

# -----------------------------
# 2. Region of Interest (Road Area)
# -----------------------------
mask = np.zeros_like(edges)
roi = np.array([
    [0, h],
    [w, h],
    [w//2, int(h*0.45)]
])
cv2.fillPoly(mask, [roi], 255)
road_edges = cv2.bitwise_and(edges, mask)

# -----------------------------
# 3. Extract lane pixel coordinates
# -----------------------------
ys, xs = np.where(road_edges > 0)

# If no pixels found → exit
if len(xs) == 0:
    print("No road pixels detected")
else:
    # -----------------------------
    # 4. Fit Polynomial Curve (Lane Direction)
    # -----------------------------
    poly = np.polyfit(ys, xs, 2)     # x = a*y^2 + b*y + c

    # -----------------------------
    # 5. Draw trajectory (red curve)
    # -----------------------------
    for y in range(h, int(h*0.45), -5):
        x = int(poly[0]*y*y + poly[1]*y + poly[2])
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)

# -----------------------------
# 6. Show result
# -----------------------------
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
