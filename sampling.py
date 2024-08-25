from PIL import Image, ImageChops
from collections import deque
import random
import os

def loadDataset(target_dir):
    path = os.path.join(os.path.dirname(__file__), target_dir)
    return [Image.open(os.path.join(path, f)) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')) and os.path.isfile(os.path.join(path, f))]

def healDataset(base_dataset, missing_dataset, base_frameIDs):
    base_queue = deque(base_dataset)
    missing_queue = deque(missing_dataset)
    transformed_IDs = [id + i for i, id in enumerate(base_frameIDs)]
    buffer = []
    i = 0

    while base_queue or missing_queue:
        if i in transformed_IDs and missing_queue:
            buffer.append(missing_queue.popleft())
        elif base_queue:
            buffer.append(base_queue.popleft())
        else:
            break
        i += 1
    return buffer

def sampleDataset(baseFrames, sampleIDs):
    sampleIDs = [id + i for i, id in enumerate(sampleIDs)]
    bulkFrames = []
    selFrames = []
    for index, frame in enumerate(baseFrames):
        if index in sampleIDs:
            selFrames.append(frame)
        else:
            bulkFrames.append(frame)
    return bulkFrames, selFrames

def sameImage(img1, img2):
    if img1.mode != img2.mode or img1.size != img2.size:
        return False
    if ImageChops.difference(img1, img2).getbbox():
        return False
    else:
        return True
def checkEquiv(frames1, frames2):
    if isinstance(frames1[0], Image.Image) and isinstance(frames2[0], Image.Image):
        return all([sameImage(f1,f2) for f1,f2 in zip(frames1,frames2)])
    return all([f1==f2 for f1,f2 in zip(frames1, frames2)])

def getSampleIDs(n, k):
    if k > n:
        raise ValueError("k cannot be greater than n")
    def is_contiguous(lst):
        return any(lst[i] + 1 == lst[i + 1] for i in range(len(lst) - 1))
    while True:
        sample = sorted(random.sample(range(1, n + 1), k))
        if not is_contiguous(sample):
            return sample
        
# baseFrames = loadDataset(r"pre_release\pre_release\sample_data\healed")
# bulkFrames = loadDataset(r"pre_release\pre_release\sample_data\original")
# selFrames = loadDataset(r"pre_release\pre_release\sample_data\missing")
# insIDs = [61, 63, 260, 404] # fullIDs

# baseFrames_ = healDataset(bulkFrames, selFrames, insIDs)
# print(checkEquiv(baseFrames_, baseFrames))
# pass
# bulkFrames_, selFrames_ = sampleDataset(baseFrames, insIDs)
# print(checkEquiv(selFrames_, selFrames))
# print(checkEquiv(bulkFrames_, bulkFrames))
# pass


baseFrames = loadDataset(r"pre_release\pre_release\sample_data\healed")
insIDs = getSampleIDs(len(baseFrames)-1,10)
bulkFrames_, selFrames_ = sampleDataset(baseFrames, insIDs)
baseFrames_ = healDataset(bulkFrames_, selFrames_, insIDs)
print(checkEquiv(baseFrames, baseFrames_))
print("")

# Heal test dataset by inserting given missing frames at given locations and renaming files in ascending order
# Function to resample dataset from healed dir with some test IDs for missing frames (non-contiguous)
# Function to score given list of missing frames with test IDs