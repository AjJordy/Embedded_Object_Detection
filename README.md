# Embedded Object Detection

## Deep Learning for Object Detection

This git is a Neural Network (NN) for object detection that can be embedded. It's calls squeezeDet. In this case the NN was trained on COCO 2017 dataset. 

## How to use

At first, you need to download the dataset from the official site. http://cocodataset.org/#download

I downloaded the files "2017 Train images [118K/18GB]", "2017 Val images [5K/1GB]", "2017 Test images [41K/6GB]" and "2017 Train/Val annotations [241MB]". Put them in the dataset folder.

If you're using linux, you need to change the path from "\\" to "/", don't forget this.

Now you need to create a .txt with the names of the imagens. 
* windows: $ Dir /s/b | sort > images.txt
* linux: $ find /dataset/train2017/ -name "\*jpg" | sort > images.txt

After that, you can change some configs. Just edit the file at libs/config/create_config.py . If you want to change the dataset or parameterize, you need to edit this file. After that you need to run this file to create a new "squeeze.config" that is gonna used by the NN.

On log folder there is a tensorboard folder that contains the information used to tensorboard. And checkpoints folder that contains the file with weights trained. There's an exemple of weights trained on COCO there.

The main files are train.py (just run that to train the NN) and eval.py (just run that to eval the NN trained). 

# References 

https://github.com/BichenWuUCB/squeezeDet

https://github.com/omni-us/squeezedet-keras

https://medium.com/searchink-eng/fast-object-detection-with-squeezedet-on-keras-5cdd124b46ce

https://medium.com/searchink-eng/a-deeper-look-into-squeezedet-on-keras-13f107d0dd32

http://cocodataset.org/#download