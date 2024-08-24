import os
from PIL import Image
from collections import deque

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

from PIL import ImageChops
def are_images_same(img1, img2):
    if img1.mode != img2.mode or img1.size != img2.size:
        return False
    if ImageChops.difference(img1, img2).getbbox():
        return False
    else:
        return True
def checkEquiv(frames1, frames2):
    if isinstance(frames1[0], Image.Image) and isinstance(frames2[0], Image.Image):
        return all([are_images_same(f1,f2) for f1,f2 in zip(frames1,frames2)])
    return all([f1==f2 for f1,f2 in zip(frames1, frames2)])

baseFrames = loadDataset(r"pre_release\pre_release\sample_data\healed")
bulkFrames = loadDataset(r"pre_release\pre_release\sample_data\original")
selFrames = loadDataset(r"pre_release\pre_release\sample_data\missing")
insIDs = [61, 63, 260, 404] # fullIDs

baseFrames_ = healDataset(bulkFrames, selFrames, insIDs)
print(checkEquiv(baseFrames_, baseFrames))
pass
bulkFrames_, selFrames_ = sampleDataset(baseFrames, insIDs)
print(checkEquiv(selFrames_, selFrames))
print(checkEquiv(bulkFrames_, bulkFrames))
pass


print("")

# Heal test dataset by inserting given missing frames at given locations and renaming files in ascending order
# Function to resample dataset from healed dir with some test IDs for missing frames (non-contiguous)
# Function to score given list of missing frames with test IDs