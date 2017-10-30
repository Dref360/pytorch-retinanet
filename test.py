import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
from datagen import ListDataset
import cv2
import numpy as np


print('Loading model..')
net = RetinaNet()
mod = torch.load('ckpt.pth')
net.load_state_dict(mod['net'])
net.eval()

print('Loss',mod['loss'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
dataset = ListDataset(root='/media/braf3002/hdd2/Downloads/MIO-TCD-Localization/test',
                          list_file='/media/braf3002/hdd2/Downloads/MIO-TCD-Localization/gt_test.csv', train=False, transform=transform, input_size=600, max_size=1000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
for idx,(inputs, loc_targets, cls_targets) in enumerate(dataloader):
    img = cv2.resize(cv2.imread('/media/braf3002/hdd2/Downloads/MIO-TCD-Localization/test/' + dataset.fnames[idx]),(600,600))
    loc_preds, cls_preds = net(Variable(inputs))

    print('Decoding..')
    encoder = DataEncoder()
    boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (600,600))

    for x0,y0,x1,y1 in boxes:
        cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),(255,0,0))
    cv2.imshow('lol',img)
    cv2.waitKey(1000)
