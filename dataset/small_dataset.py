# Author: Jordy A. Faria de Araújo
# Date: 14/09/2018
# Email: jordyfaria0@gmail.com
# Github: AjJordy

import os
import json

"""
This script delete the images from COCO dataset that does't match with a list of prior classes
Parameters:
gt_dir - the path from the json file with the labels
img_file - a txt file with all the paths from the images of the dataset
base - a full path to compare the name of the image with the path 
class_need - Classes that I want
"""


# img_file = 'images.txt'
img_file = 'img_val.txt'
# gt_dir = 'annotations\\instances_train2017.json'
gt_dir = 'annotations\\instances_val2017.json'
# base = "D:\\Humanoid\\squeezeDet\\Embedded_Object_Detection\\dataset\\train2017\\"
base = "D:\\Humanoid\\squeezeDet\\Embedded_Object_Detection\\dataset\\val2017\\"

class_need = [1,2,3,4,37,73,74,75,77,78,80,82,89]

with open(img_file) as imgs:
    img_names = imgs.read().splitlines()
    print("Read images")
imgs.close()

with open(gt_dir,'r') as f:
    data = json.load(f)
    print("Read annotations")
f.close() 

print("Começando ...")
for img in img_names:
	for j in range(len(data["images"])):
		dir = base + data["images"][j]['file_name']
		if(dir == img):
			index = data["images"][j]['id']
	for j in range(len(data["annotations"])):
		try:
			if data['annotations'][j]['image_id'] == index :
				ann = data["annotations"][j]
				class_id = ann["category_id"]
				if(class_id not in class_need):
					print("Deleting ",img)
					os.remove(img)
		except:
			pass


# 'person' = 1
# 'bicycle'= 2
# 'car'= 3
# 'motorcycle'= 4
# 'airplane'= 5
# 'bus'= 6
# 'train'= 7
# 'truck'= 8
# 'boat'= 9
# 'traffic_light',= 10
# 'fire_hydrant'= 11
# 'stop_sign'= 13
# 'parking_meter'= 14
# 'bench'=15
# 'bird'=16
# 'cat'=17
# 'dog'=18
# 'horse'=19
# 'sheep'=20
# 'cow'=21
# 'elephant'=22
# 'bear'=23
# 'zebra'=24
# 'giraffe'=25
# 'backpack'=27
# 'umbrella'=28
# 'handbag'=31
# 'tie'=32
# 'suitcase'=33
# 'frisbee'=34
# 'skis'=35
# 'snowboard'=36
# 'sports_ball'=37
# 'kite'=38
# 'baseball_bat'=39
# 'baseball_glove'=40
# 'skateboard'=41
# 'surfboard'=42
# 'tennis_racket'=43
# 'bottle'=44
# 'wine_glass'=46
# 'cup'=47
# 'fork'=48
# 'knife'=49
# 'spoon'=50
# 'bowl'=51
# 'banana'=52
# 'apple'=53
# 'sandwich'=54
# 'orange'=55
# 'broccoli'=56
# 'carrot'=57
# 'hot_dog'=58
# 'pizza'=59
# 'donut'=60
# 'cake'=61
# 'chair'=62
# 'couch'=63
# 'potted_plant'=64
# 'bed'=65
# 'dining_table'=67
# 'toilet'=70
# 'tv'=72
# 'laptop'=73
# 'mouse'=74
# 'remote'=75
# 'keyboard'=76
# 'cell_phone'=77
# 'microwave'=78
# 'oven'=79
# 'toaster'=80
# 'sink'=81
# 'refrigerator'=82
# 'book'=84
# 'clock'=85
# 'vase'=86
# 'scissors'=87
# 'teddy_bear'=88
# 'hair_drier'=89
# 'toothbrush'=90

     