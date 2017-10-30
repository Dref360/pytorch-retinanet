anchor_areas = [32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.]
import os
import pickle
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
pjoin = os.path.join

with open('/media/braf3002/hdd2/Downloads/MIO-TCD-Localization/gt_train.csv', "r") as f:
    lines = [(s.split(',')[0], [float(s_) for s_ in s.split(',')[2:]]) for s in f.readlines()]

areas = [(x2 - x1) * (y2 - y1) for _, (x1, y1, x2, y2) in lines]
if os.path.exists('shapes.pkl'):
    shapes = pickle.load(open('shapes.pkl','rb'))
else:
    print("Shapes")
    shapes = [(Image.open(pjoin('/media/braf3002/hdd2/Downloads/MIO-TCD-Localization/train', k[0] + '.jpg')).size[0],
              Image.open(pjoin('/media/braf3002/hdd2/Downloads/MIO-TCD-Localization/train', k[0] + '.jpg')).size[1]) for k in tqdm(lines)]

    pickle.dump(shapes,open('shapes.pkl','wb'))

print("AS")
aspect_ratio = [(max(1,x2 - x1)/w) / (max(1,y2 - y1)/h) for (_, (x1, y1, x2, y2)),(w,h) in zip(lines,shapes)]

print("Scales")
scales = [ar / (sh[0] * sh[1]) for ar, sh in zip(areas, shapes)]

print("Kmeans")
km = KMeans(3)
print("Scales", km.fit(np.array(scales).reshape([-1,1])).cluster_centers_.reshape([-1]))
km = KMeans(3)
print("Aspect ratio", km.fit(np.array(aspect_ratio).reshape([-1,1])).cluster_centers_.reshape([-1]))
