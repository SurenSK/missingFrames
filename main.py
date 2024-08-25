from PIL import Image, ImageChops
from sampling import loadDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
from PIL import Image
import ot
import cv2
from scipy.spatial.distance import cdist
from scipy import ndimage

def prop(l): return mean([x>1 for x in l])
def mean(l): return sum(l)/len(l)
def calculate_emd(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate the 2D Earth Mover's Distance (EMD) between two images.
    
    :param img1: First PIL Image object
    :param img2: Second PIL Image object
    :return: A float representing the EMD
    """
    # Convert images to grayscale numpy arrays
    arr1 = np.array(img1.convert('L')).astype(float)
    arr2 = np.array(img2.convert('L')).astype(float)
    
    # Normalize pixel intensities
    arr1 = arr1 / np.max(arr1)
    arr2 = arr2 / np.max(arr2)
    
    # Get coordinates of non-zero pixels
    y1, x1 = np.nonzero(arr1)
    y2, x2 = np.nonzero(arr2)
    
    # Create points arrays
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    
    # Get intensities as weights
    weights1 = arr1[y1, x1]
    weights2 = arr2[y2, x2]
    
    # Normalize weights
    weights1 = weights1 / np.sum(weights1)
    weights2 = weights2 / np.sum(weights2)
    
    # Calculate distance matrix
    dist_matrix = cdist(points1, points2)
    
    # Calculate EMD
    emd_value = ot.emd2(weights1, weights2, dist_matrix)
    
    return emd_value
def calculate_mse(img1, img2):
    # Convert PIL images to grayscale numpy arrays
    arr1 = np.array(img1.convert('L'), dtype=float)
    arr2 = np.array(img2.convert('L'), dtype=float)
    
    # Calculate MSE
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Convert to an integer score
    score = int(mse)
    
    return score
def calculate_edge_change_ratio(img1, img2):
    # Convert PIL images to grayscale numpy arrays
    arr1 = np.array(img1.convert('L'))
    arr2 = np.array(img2.convert('L'))
    
    # Detect edges using Sobel filter
    edges1 = np.hypot(ndimage.sobel(arr1, axis=0), ndimage.sobel(arr1, axis=1))
    edges2 = np.hypot(ndimage.sobel(arr2, axis=0), ndimage.sobel(arr2, axis=1))
    
    # Threshold edges
    threshold = 50
    edges1 = (edges1 > threshold).astype(int)
    edges2 = (edges2 > threshold).astype(int)
    
    # Calculate edge change ratio
    xor_edges = np.logical_xor(edges1, edges2)
    ecr = np.sum(xor_edges) / max(np.sum(edges1), np.sum(edges2))
    
    # Convert to an integer score
    score = int(ecr * 1000)
    
    return score
def calculate_sad(img1, img2): return np.sum(np.abs(np.array(ImageChops.difference(img1, img2))))
def calculate_block_hist(img1, img2, blocks=4):
    # Convert PIL images to grayscale numpy arrays
    arr1 = np.array(img1.convert('L'))
    arr2 = np.array(img2.convert('L'))
    
    height, width = arr1.shape
    block_h, block_w = height // blocks, width // blocks
    
    total_diff = 0
    for i in range(blocks):
        for j in range(blocks):
            block1 = arr1[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            block2 = arr2[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            
            hist1, _ = np.histogram(block1, bins=256, range=(0, 256))
            hist2, _ = np.histogram(block2, bins=256, range=(0, 256))
            
            diff = np.sum(np.abs(hist1 - hist2))
            total_diff += diff
    
    # Convert to an integer score
    score = int(total_diff / (blocks * blocks))
    
    return score
def calculate_hist(img1, img2):
    # Convert images to grayscale if they're not already
    if img1.mode != 'L':
        img1 = img1.convert('L')
    if img2.mode != 'L':
        img2 = img2.convert('L')
    
    # Convert PIL images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate histograms
    hist1, _ = np.histogram(arr1, bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(arr2, bins=256, range=(0, 256), density=True)
    
    # Calculate entropy for each histogram
    entropy1 = -np.sum(hist1 * np.log2(hist1 + 1e-7))
    entropy2 = -np.sum(hist2 * np.log2(hist2 + 1e-7))
    
    # Calculate the difference in entropy
    entropy_diff = abs(entropy1 - entropy2)
    
    # Convert to an integer score (you can adjust the scaling factor)
    score = int(entropy_diff * 1000)
    
    return score

def getBaseScores(images, metric, sampleidx=None):
    scores = []
    if not sampleidx:
        for i in range(len(images) - 1):
            sad = metric(images[i], images[i + 1])
            scores.append(sad)
        scores.append(0)
    else:
        for i in sampleidx:
            sad = metric(images[i], images[i + 1])
            scores.append(sad)
    return scores

def getSkipScores(images, metric, sampleidx=None):
    scores = []
    if not sampleidx:
        scores.append(0)
        for i in range(1, len(images) - 1):
            sad = metric(images[i - 1], images[i + 1])
            scores.append(sad)
        scores.append(0)
    else:
        for i in sampleidx:
            sad = metric(images[i - 1], images[i + 1])
            scores.append(sad)
    return scores

def getSkipSensitivities(images, metric):
    scores = [0]
    for i in range(1, len(images) - 1):
        score_base = metric(images[i], images[i - 1])
        score_skip = metric(images[i - 1], images[i + 1])
        diff = float(score_skip) - float(score_base)
        scores.append(diff)
    scores.append(0)
    return scores

def plot(base, skip=None, bg=True):
    plt.figure(figsize=(10, 6))
    
    if skip is None:
        plt.plot(base, label='Sensitivity', marker='', color='grey')
        title = 'ImgDist Scores'
        if bg:
            for i, value in enumerate(base):
                if value > 0:
                    plt.axvspan(i, i+1, color='red', alpha=0.3)
                else:
                    plt.axvspan(i, i+1, color='grey', alpha=0.3)
    else:
        plt.plot(base, label='Base Scores', marker='', color='grey')
        plt.plot(skip, label='Skip Scores', marker='', color='red')
        title = 'ImgDist Scores'
        if bg:
            for i in range(min(len(base), len(skip))):
                if base[i] > skip[i]:
                    plt.axvspan(i, i+1, color='grey', alpha=0.3)
                else:
                    plt.axvspan(i, i+1, color='red', alpha=0.3)

    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value' if skip is None else 'SAD Score')
    plt.legend()
    plt.grid(True)
    plt.show()

baseFrames = loadDataset(r"pre_release\pre_release\sample_data\healed")

import numpy as np

# Assuming getSkipSensitivities function and the calculation methods are defined elsewhere
hist_sensitivities = getSkipSensitivities(baseFrames, calculate_hist)
sad_sensitivities = getSkipSensitivities(baseFrames, calculate_sad)
mse_sensitivities = getSkipSensitivities(baseFrames, calculate_mse)
block_hist_sensitivities = getSkipSensitivities(baseFrames, calculate_block_hist)
edge_change_ratio_sensitivities = getSkipSensitivities(baseFrames, calculate_edge_change_ratio)

# Print proportions using the prop() function for each type of sensitivity
print("Proportion of Histogram Sensitivities:", prop(hist_sensitivities))
print("Proportion of SAD Sensitivities:", prop(sad_sensitivities))
print("Proportion of MSE Sensitivities:", prop(mse_sensitivities))
print("Proportion of Block Histogram Sensitivities:", prop(block_hist_sensitivities))
print("Proportion of Edge Change Ratio Sensitivities:", prop(edge_change_ratio_sensitivities))

# Convert the lists to numpy arrays if they are not already
hist_sensitivities = np.array(hist_sensitivities)
sad_sensitivities = np.array(sad_sensitivities)
mse_sensitivities = np.array(mse_sensitivities)
block_hist_sensitivities = np.array(block_hist_sensitivities)
edge_change_ratio_sensitivities = np.array(edge_change_ratio_sensitivities)

# Stack the arrays to form a 2D array where each column is one set of sensitivities
data = np.column_stack((hist_sensitivities, sad_sensitivities, mse_sensitivities, block_hist_sensitivities, edge_change_ratio_sensitivities))

# Compute the correlation matrix
correlation_matrix = np.corrcoef(data, rowvar=False)  # Ensure columns are variables

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler

# Assuming 'data' and 'labels' are defined as in the previous code snippets
labels = ["Histogram", "SAD", "MSE", "Block Histogram", "Edge Change Ratio"]
prop_scores = np.array([0.5881, 0.6178, 0.5606, 0.6316, 0.4416])

# Normalize prop() scores
scaler = MinMaxScaler()
normalized_props = scaler.fit_transform(prop_scores.reshape(-1, 1)).flatten()

# Compute mutual information matrix
mi_matrix = np.zeros((data.shape[1], data.shape[1]))
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        mi_matrix[i, j] = mutual_info_score(data[:, i], data[:, j])

# Calculate weighted mutual information
weighted_mi_matrix = np.zeros_like(mi_matrix)
for i in range(data.shape[1]):
    for j in range(i + 1, data.shape[1]):
        weight = (normalized_props[i] + normalized_props[j]) / 2
        weighted_mi_matrix[i, j] = mi_matrix[i, j] * weight
        weighted_mi_matrix[j, i] = weighted_mi_matrix[i, j]  # Symmetric matrix

# Print the weighted mutual information matrix with labels
print("Weighted Mutual Information Matrix:")
for i in range(len(labels)):
    print(f"{labels[i]:>15}", end=" ")
    for j in range(len(labels)):
        print(f"{weighted_mi_matrix[i, j]:.4f}", end=" ")
    print()  # New line for each metric


pass
# plot(sens)
# plot(baseScores, skipScores)
pass