import numpy as np
import math
import cv2
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def evaluat_error(sp, act) -> object:
    r = np.squeeze(act)
    x = np.squeeze(sp)
    points = np.zeros(len(x))
    abs_r = np.zeros(len(x))
    abs_x = np.zeros(len(x))
    abs_r_x = np.zeros(len(x))
    abs_x_r = np.zeros(len(x))
    abs_r_x__r = np.zeros(len(x))
    for j in range(len(x)):
        points[j] = abs(x[j] - x[j - 1])
    for i in range(len(r)):
        abs_r[i] = abs(r[i])
    for i in range(len(r)):
        abs_x[i] = abs(x[i])
    for i in range(len(r)):
        abs_r_x[i] = abs(r[i] - x[i])
    for i in range(len(r)):
        abs_x_r[i] = abs(x[i] - r[i])
    for i in range(len(r)):
        abs_r_x__r[i] = abs((r[i] - x[i]) / r[i])
    md = (100 / len(x)) * sum(abs_r_x__r)
    smape = (1 / len(x)) * sum(abs_r_x / ((abs_r + abs_x) / 2))
    mase = sum(abs_r_x) / ((1 / (len(x) - 1)) * sum(points))
    mae = sum(abs_r_x) / len(r)
    rmse = (sum(abs_x_r ** 2) / len(r)) ** 0.5
    mse = rmse ** 2
    accuracy = (1 - (np.abs(sum(r - x)) / sum(r))) * 100
    EVAL_ERR = [md, smape, mase, mae, rmse, mse, accuracy]
    return EVAL_ERR


def net_evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = (2 * tp) / ((2 * tp) + fp + fn)
    Jaccard = tp / (tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    FPR = fp / (fp + tn)
    FNR = fn / (tp + fn)
    NPV = tn / (tn + fn)
    FDR = fp / (tp + fp)
    F1_score = (2 * tp) / (2 * tp + fp + fn)
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, precision, F1_score, sensitivity, specificity, FPR, FNR, NPV, FDR,
            MCC]
    return EVAL


# 1. PSNR
def compute_psnr(img1, img2):
    """
    img1, img2: uint8 [0-255] or float [0-1], shape (H,W,C)
    """
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.max() <= 1.0:
        img2 = (img2 * 255).astype(np.uint8)
    return compare_psnr(img1, img2, data_range=255)


# 2. BRISQUE (No Reference)
# Simple implementation placeholder using MSCN statistics
def compute_brisque(img):
    """
    img: uint8 [0-255] RGB image
    Returns a simple MSCN-based score as a proxy for BRISQUE
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mu = cv2.GaussianBlur(img_gray.astype(np.float32), (7, 7), 1.166)
    sigma = np.sqrt(cv2.GaussianBlur((img_gray - mu) ** 2, (7, 7), 1.166))
    mscn = (img_gray - mu) / (sigma + 1.0)
    score = np.mean(np.abs(mscn))  # proxy: lower = better
    return score


# 3. NIQE (No Reference)
# Simple variance-of-MSCN proxy
def compute_niqe(img):
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if img.max() <= 1.0 else cv2.cvtColor(img,
                                                                                                                    cv2.COLOR_RGB2GRAY)
    mu = cv2.GaussianBlur(img_gray.astype(np.float32), (7, 7), 1.166)
    sigma = np.sqrt(cv2.GaussianBlur((img_gray - mu) ** 2, (7, 7), 1.166))
    mscn = (img_gray - mu) / (sigma + 1.0)
    return np.var(mscn)  # lower = better


# 4. PIQE (No Reference)
# Using block variance
def compute_piqe(img, block_size=32):
    """
    img: float [0-1] or uint8
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_gray[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue
            blocks.append(np.var(block))
    return np.mean(blocks)  # higher = worse


# 5 & 6. FID and KID using Keras
def get_inception_features(images):
    """
    images: numpy array NxHxWx3, float [0-1]
    returns: Nx2048 features
    """
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    images_resized = np.array([cv2.resize(img, (299, 299)) for img in images])
    images_pre = preprocess_input(images_resized * 255.0)
    features = model.predict(images_pre)
    return features


def compute_fid(real_feats, gen_feats):
    mu_r = np.mean(real_feats[:, :, 0], axis=0)
    mu_g = np.mean(gen_feats[:, :, 0], axis=0)
    sigma_r = np.cov(real_feats[:, :, 0], rowvar=False)
    sigma_g = np.cov(gen_feats[:, :, 0], rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r.dot(sigma_g))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return fid


def compute_kid(real_feats, gen_feats):
    # Flatten each sample to 1D features
    real_flat = real_feats.reshape(real_feats.shape[0], -1)
    gen_flat = gen_feats.reshape(gen_feats.shape[0], -1)

    def polynomial_kernel(x, y):
        # x: (n_samples_x, n_features)
        # y: (n_samples_y, n_features)
        return (np.dot(x, y.T) / x.shape[1] + 1.0) ** 3

    m = real_flat.shape[0]
    n = gen_flat.shape[0]

    K_XX = polynomial_kernel(real_flat, real_flat)
    K_YY = polynomial_kernel(gen_flat, gen_flat)
    K_XY = polynomial_kernel(real_flat, gen_flat)

    # MMD (unbiased)
    mmd = np.sum(K_XX) / (m * (m - 1)) + np.sum(K_YY) / (n * (n - 1)) - 2 * np.sum(K_XY) / (m * n)
    return mmd


def Image_Evaluation(img1, img2):
    PSNR = compute_psnr(img1, img2)
    BRISQUE = compute_brisque(img1)
    NIQE = compute_niqe(img1)
    PIQE = compute_piqe(img1)
    FID = compute_fid(img1, img2)
    KID = compute_kid(img1, img2)
    Eval = [PSNR, BRISQUE, NIQE, PIQE, FID, KID]
    return Eval
