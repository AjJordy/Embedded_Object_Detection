# Author: Jordy A. Faria de Araújo
# Date: 14/09/2018
# Email: jordyfaria0@gmail.com
# Github: AjJordy

import json
import os

"""
This script delete the images from COCO dataset that does't hava bouding box
Parameters:
gt_dir - the path from the json file with the labels
img_file - a txt file with all the paths from the images of the dataset
base - a full path to compare the name of the image with the path 
"""

gt_dir = 'dataset\\annotations\\instances_train2017.json'
img_file = 'dataset\\backup_train.txt' 
base = 'D:\\Humanoid\\squeezeDet\\Embedded_Object_Detection\\dataset\\train2017\\'

with open(img_file) as imgs:
    img_names = imgs.read().splitlines()
imgs.close()

with open(gt_dir,'r') as f:
    data = json.load(f)
f.close()
print("File read")

ann = {}

# Run for each image
for img in img_names:
    ann[img] = []

    # Find the id_image from the name of the image
    for j in range(len(data["images"])):
        dir = base + data["images"][j]['file_name']
        if(dir == img):
            index = data["images"][j]['id']            
            print(img)
            break

    # Search in all json file looking for the image's annotations
    # And make a json with a list of bouding box and class
    for j in range(len(data["annotations"])):
        if data['annotations'][j]['image_id'] == index :
            bb = data["annotations"][j]
            ann[img].append([bb["bbox"][0], bb["bbox"][1], bb["bbox"][2], bb["bbox"][3], bb["category_id"]])

    # Delete the image without label
    if len(ann[img]) == 0:
        del ann[img]
        print("del ",img)
        os.remove(img)


print("cabou")
with open('ann_val_clean.json', 'w') as outfile:
    json.dump(ann, outfile)
outfile.close()
