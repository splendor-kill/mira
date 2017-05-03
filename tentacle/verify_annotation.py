import os
import numpy as np
import cv2
from annotate import load_json


data_dir = '../dataset'

cats = load_json(os.path.join(data_dir, 'cats.json'))
cats = dict({k: v['name'] for k, v in cats.items()})
print('cat num:', len(cats))

annos = load_json(os.path.join(data_dir, 'annotations.json'))
print('annotation num:', len(annos))

images = load_json(os.path.join(data_dir, 'images.json'))
print('image num:', len(images))


ids = list(annos.keys())
test_ids = ids[:5]
print('following images will be tested')
print(test_ids)

print(annos[1])

for i in test_ids:
    img_info = images[i]
    full_file_name = os.path.join(data_dir, img_info['folder'], img_info['file_name'])
    image = cv2.imread(full_file_name)
    print(full_file_name)
    if image is None:
        print("file '%s' read error" % (full_file_name,))
        continue

    # anno = annos[i]
    # print(anno)
    # for ci in anno.keys():
    #     print('cat id:', ci, cats[ci])
        
    # bbs = reduce_bbs(bbs, 2)
    # for x, y, w, h in bbs:
    #     print('bbox:', x, y, w, h)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     print()
    #     cv2.imshow('pic browser', image)
    #     k = cv2.waitKey(0)
    #     if k == ord('q'):
    #         return
    
    # print(full_file_name)
    # break
