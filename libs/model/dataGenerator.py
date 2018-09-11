## Original project source
# Project: squeezeDetOnKeras
# Filename: visualization
# Author: Christopher Ehmann
# Date: 12.12.17
# Organisation: searchInk
# Email: christopher@searchink.com

## Edited project
# Author: Jordy A. Faria de Araújo
# Date: 25/07/2018
# Email: jordyfaria0@gmail.com

import sys
import threading
import cv2
import numpy as np
import random
import json
from libs.utils.utils import bbox_transform_inv, batch_iou, sparse_to_dense


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def load_annotation(data,img_file,base):        
        annotations = []        
        try:
            # search in all json file looking for the 'file_name'        
            for j in range(len(data["images"])):
                dir = base + data["images"][j]['file_name']
                if(dir == img_file): 
                    index = data["images"][j]['id']
                    break

            # search in all json file looking for the image's annotations             
            for j in range(len(data["annotations"])):                                
                if data['annotations'][j]['image_id'] == index :
                    bb = data["annotations"][j]
                    class_id = bb["category_id"]                                        
                    x = bb["bbox"][0]
                    y = bb["bbox"][1]
                    w = bb["bbox"][2]
                    h = bb["bbox"][3]
                    annotations.append([x, y, w, h, class_id])
        except:
            print("bb ",bb)                       
        return annotations


#we could maybe use the standard data generator from keras?
def read_image_and_gt(img_names, data, config, base):
    '''
    Transform images and send transformed image and label
    :param img_files: list of image files including the path of a batch
    :param data: list of gt files including the path of a batch
    :param config: config dict containing various hyperparameters

    :return images and annotations
    '''

    labels = []
    bboxes = []
    deltas = []
    aidxs  = []

    '''
    Loads annotations from file
    : return: list with x and y of central point, weight and height
    '''
    
    #init tensor of images
    imgs = np.zeros((config.BATCH_SIZE,
                     config.IMAGE_HEIGHT,
                     config.IMAGE_WIDTH,
                     config.N_CHANNELS))

    img_idx = 0

    # annotations = load_annotation(data,img_names,base,config)

    #iterate files
    for img_name in img_names:
        #open img
        img = cv2.imread(img_name).astype(np.float32, copy=False)

        # scale image
        img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        #subtract means
        img = (img - np.mean(img))/ np.std(img)

        #store original height and width?
        orig_h, orig_w, _ = [float(v) for v in img.shape]

        # load annotations
        annotations = load_annotation(data,img_name,base)

        #split in classes and boxes
        labels_per_file = [a[4] for a in annotations]

        bboxes_per_file = np.array([a[0:4]for a in annotations])

        #and store
        imgs[img_idx] = np.asarray(img)
        img_idx += 1

        # scale annotation
        x_scale = config.IMAGE_WIDTH / orig_w
        y_scale = config.IMAGE_HEIGHT / orig_h

        #scale boxes
        try:
            bboxes_per_file[:, 0::2] = bboxes_per_file[:, 0::2] * x_scale
            bboxes_per_file[:, 1::2] = bboxes_per_file[:, 1::2] * y_scale
        except:
            continue


        bboxes.append(bboxes_per_file)

        aidx_per_image, delta_per_image = [], []
        aidx_set = set()


        #iterate all bounding boxes for a file
        for i in range(len(bboxes_per_file)):
            #compute overlaps of bounding boxes and anchor boxes
            overlaps = batch_iou(config.ANCHOR_BOX, bboxes_per_file[i])

            #achor box index
            aidx = len(config.ANCHOR_BOX)

            #sort for biggest overlaps
            for ov_idx in np.argsort(overlaps)[::-1]:
                #when overlap is zero break
                if overlaps[ov_idx] <= 0:
                    break
                #if one is found add and break
                if ov_idx not in aidx_set:
                    aidx_set.add(ov_idx)
                    aidx = ov_idx
                    break

            # if the largest available overlap is 0, choose the anchor box with the one that has the
            # smallest square distance
            if aidx == len(config.ANCHOR_BOX):
                dist = np.sum(np.square(bboxes_per_file[i] - config.ANCHOR_BOX), axis=1) # Somatório das distancias euclidiana dos centros e do formato
                for dist_idx in np.argsort(dist):
                    if dist_idx not in aidx_set: # Se a distancia ainda não tiver sido contabilizada, é adicionado a lista
                        aidx_set.add(dist_idx)
                        aidx = dist_idx
                        break


            #compute deltas for regression
            box_cx, box_cy, box_w, box_h = bboxes_per_file[i]
            delta = [0] * 4
            delta[0] = (box_cx - config.ANCHOR_BOX[aidx][0]) / config.ANCHOR_BOX[aidx][2] # box_x - anchor_box_x / anchor_shapes
            delta[1] = (box_cy - config.ANCHOR_BOX[aidx][1]) / config.ANCHOR_BOX[aidx][3] # box_y - anchor_box_y / anchor_shapes
            delta[2] = np.log(box_w / config.ANCHOR_BOX[aidx][2])
            delta[3] = np.log(box_h / config.ANCHOR_BOX[aidx][3])

            aidx_per_image.append(aidx)
            delta_per_image.append(delta)

        deltas.append(delta_per_image)
        aidxs.append(aidx_per_image)
        labels.append(labels_per_file)


    #we need to transform this batch annotations into a form we can feed into the model
    label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []

    #iterate batch
    for i in range(len(labels)):
        #and annotations
        for j in range(len(labels[i])):
            if (i, aidxs[i][j]) not in aidx_set:
                aidx_set.add((i, aidxs[i][j]))
                label_indices.append([i, aidxs[i][j], labels[i][j]])
                mask_indices.append([i, aidxs[i][j]])
                bbox_indices.extend([[i, aidxs[i][j], k] for k in range(4)])
                box_delta_values.extend(deltas[i][j])
                box_values.extend(bboxes[i][j])


    #transform them into matrices
    input_mask = np.reshape(
                        sparse_to_dense(
                            mask_indices,
                            [config.BATCH_SIZE, config.ANCHORS],
                            [1.0] * len(mask_indices)),
                        [config.BATCH_SIZE, config.ANCHORS, 1])


    box_delta_input = sparse_to_dense(
                        bbox_indices, 
                        [config.BATCH_SIZE, config.ANCHORS, 4],
                        box_delta_values)

    box_input = sparse_to_dense(
                        bbox_indices, 
                        [config.BATCH_SIZE, config.ANCHORS, 4],
                        box_values)

    
    labels = sparse_to_dense(
                    label_indices,
                    [config.BATCH_SIZE, config.ANCHORS, config.CLASSES],
                    [1.0] * len(label_indices)) 

    # print(" SIZES: ", len(input_mask)," ",len(box_delta_input)," ",len(box_input)," ",len(labels))

    #concatenate ouputs
    Y = np.concatenate((input_mask, box_input, box_delta_input, labels), axis=-1).astype(np.float32)

    return imgs, Y

def read_image_and_gt_with_original(img_files, data, config,base):
    '''
    Transform images and send transformed image and label, but also return the image only resized
    :param img_files: list of image files including the path of a batch
    :param gt_files: list of gt files including the path of a batch
    :param config: config dict containing various hyperparameters

    :return images and annotations
    '''

    labels = []
    bboxes = []
    deltas = []
    aidxs  = [] 

    imgs = np.zeros((config.BATCH_SIZE, 
                     config.IMAGE_HEIGHT, 
                     config.IMAGE_WIDTH, 
                     config.N_CHANNELS))

    imgs_only_resized = np.zeros((config.BATCH_SIZE, 
                                  config.IMAGE_HEIGHT, 
                                  config.IMAGE_WIDTH, 
                                  config.N_CHANNELS))

    img_idx = 0    

    # iterate files
    for img_name in img_files:
        #open img
        img = cv2.imread(img_name).astype(np.float32, copy=False)

        #store original height and width?
        orig_h, orig_w, _ = [float(v) for v in img.shape]

        # scale image
        img = cv2.resize( img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        imgs_only_resized[img_idx] = img

        #subtract means
        img = (img - np.mean(img))/ np.std(img)
      
        # load annotations
        annotations = load_annotation(data,img_name,base)

        #split in classes and boxes
        labels_per_file = [a[4] for a in annotations]
        bboxes_per_file = np.array([a[0:4] for a in annotations])

        #TODO enable dynamic Data Augmentation
        """
        if config.DATA_AUGMENTATION:
            assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, \
                'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'

            if mc.DRIFT_X > 0 or mc.DRIFT_Y > 0:
                # Ensures that gt boundibg box is not cutted out of the image
                max_drift_x = min(gt_bbox[:, 0] - gt_bbox[:, 2] / 2.0 + 1)
                max_drift_y = min(gt_bbox[:, 1] - gt_bbox[:, 3] / 2.0 + 1)
                assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'

                dy = np.random.randint(-mc.DRIFT_Y, min(mc.DRIFT_Y + 1, max_drift_y))
                dx = np.random.randint(-mc.DRIFT_X, min(mc.DRIFT_X + 1, max_drift_x))

                # shift bbox
                gt_bbox[:, 0] = gt_bbox[:, 0] - dx
                gt_bbox[:, 1] = gt_bbox[:, 1] - dy

                # distort image
                orig_h -= dy
                orig_w -= dx
                orig_x, dist_x = max(dx, 0), max(-dx, 0)
                orig_y, dist_y = max(dy, 0), max(-dy, 0)

                distorted_im = np.zeros(
                    (int(orig_h), int(orig_w), 3)).astype(np.float32)
                distorted_im[dist_y:, dist_x:, :] = im[orig_y:, orig_x:, :]
                im = distorted_im

            # Flip image with 50% probability
            if np.random.randint(2) > 0.5:
                im = im[:, ::-1, :]
                gt_bbox[:, 0] = orig_w - 1 - gt_bbox[:, 0]
        """

        #and store
        imgs[img_idx] = np.asarray(img)
        
        img_idx += 1

        # scale annotation
        x_scale = config.IMAGE_WIDTH / orig_w
        y_scale = config.IMAGE_HEIGHT / orig_h

        try:
            # scale boxes
            bboxes_per_file[:, 0::2] = bboxes_per_file[:, 0::2] * x_scale
            bboxes_per_file[:, 1::2] = bboxes_per_file[:, 1::2] * y_scale
        except:
            print('bboxes per file')
            continue

        bboxes.append(bboxes_per_file)

        aidx_per_image, delta_per_image = [], []
        aidx_set = set()

        #iterate all bounding boxes for a file
        for i in range(len(bboxes_per_file)):

            #compute overlaps of bounding boxes and anchor boxes
            overlaps = batch_iou(config.ANCHOR_BOX, bboxes_per_file[i])

            #achor box index
            aidx = len(config.ANCHOR_BOX)

            #sort for biggest overlaps
            for ov_idx in np.argsort(overlaps)[::-1]:
                #when overlap is zero break
                if overlaps[ov_idx] <= 0:
                    break
                #if one is found add and break
                if ov_idx not in aidx_set:
                    aidx_set.add(ov_idx)
                    aidx = ov_idx
                    break

            # if the largest available overlap is 0, choose the anchor box with the one that has the
            # smallest square distance
            if aidx == len(config.ANCHOR_BOX):
                dist = np.sum(np.square(bboxes_per_file[i] - config.ANCHOR_BOX), axis=1)
                for dist_idx in np.argsort(dist):
                    if dist_idx not in aidx_set:
                        aidx_set.add(dist_idx)
                        aidx = dist_idx
                        break


            #compute deltas for regression
            box_cx, box_cy, box_w, box_h = bboxes_per_file[i]
            delta = [0] * 4
            delta[0] = (box_cx - config.ANCHOR_BOX[aidx][0]) / config.ANCHOR_BOX[aidx][2]
            delta[1] = (box_cy - config.ANCHOR_BOX[aidx][1]) / config.ANCHOR_BOX[aidx][3]
            delta[2] = np.log(box_w / config.ANCHOR_BOX[aidx][2])
            delta[3] = np.log(box_h / config.ANCHOR_BOX[aidx][3])

            aidx_per_image.append(aidx)
            delta_per_image.append(delta)

        deltas.append(delta_per_image)
        aidxs.append(aidx_per_image)
        labels.append(labels_per_file)


    #print(labels)
    #we need to transform this batch annotations into a form we can feed into the model
    label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []

    aidx_set = set()


    #iterate batch
    for i in range(len(labels)):
        #and annotations
        for j in range(len(labels[i])):
            if (i, aidxs[i][j]) not in aidx_set:
                aidx_set.add((i, aidxs[i][j]))
                label_indices.append(
                    [i, aidxs[i][j], labels[i][j]])
                mask_indices.append([i, aidxs[i][j]])
                bbox_indices.extend(
                    [[i, aidxs[i][j], k] for k in range(4)])
                box_delta_values.extend(deltas[i][j])
                box_values.extend(bboxes[i][j])


    #transform them into matrices
    input_mask =  np.reshape(
            sparse_to_dense(
                mask_indices,
                [config.BATCH_SIZE, config.ANCHORS],
                [1.0] * len(mask_indices)),

            [config.BATCH_SIZE, config.ANCHORS, 1])

    box_delta_input =  sparse_to_dense(
            bbox_indices, [config.BATCH_SIZE, config.ANCHORS, 4],
            box_delta_values)

    box_input =  sparse_to_dense(
            bbox_indices, [config.BATCH_SIZE, config.ANCHORS, 4],
            box_values)

    labels = sparse_to_dense(
            label_indices,
            [config.BATCH_SIZE, config.ANCHORS, config.CLASSES],
            [1.0] * len(label_indices))

    #concatenate ouputs
    Y = np.concatenate((input_mask, box_input,  box_delta_input, labels), axis=-1).astype(np.float32)

    return imgs, Y, imgs_only_resized




@threadsafe_generator
def generator_from_data_path(img_names, data, base, config, return_filenames=False, shuffle=False):
    """
    Generator that yields (X, Y)
    :param img_names: list of images names with full path
    :param gt_dir: path of the .json file with the anotations
    :param config: config dict containing various hyperparameters
    :return: a generator yielding images and ground truths
   """

    if shuffle:
        #permutate images
        shuffled = list(img_names)
        random.shuffle(shuffled)
        img_names = shuffled

    """
    Each epoch will only process an integral number of batch_size
    but with the shuffling of list at the top of each epoch, we will
    see all training samples eventually, but will skip an amount
    less than batch_size during each epoch
    """

    nbatches, n_skipped_per_epoch = divmod(len(img_names), config.BATCH_SIZE)
    count = 1
    epoch = 0

    # data = input_file(gt_dir)

    while 1:
        epoch += 1
        i, j = 0, config.BATCH_SIZE

        for _ in range(nbatches):
            img_names_batch = img_names[i:j]            
            try:
                #get images and ground truths                
                imgs, gts = read_image_and_gt(img_names_batch, data, config,base)
                yield (imgs, gts)
            except IOError as err:
                print("IOError ", err)
                count -= 1
            i = j
            j += config.BATCH_SIZE
            count += 1



def visualization_generator_from_data_path(img_names, data,base, config, return_filenames=False, shuffle=False ):
    """
    Generator that yields (Images, Labels, unnormalized images)
    :param img_names: list of images names with full path
    :param gt_names: list of gt names with full path
    :param config
    :return:
    """

    """
    Each epoch will only process an integral number of batch_size
    # but with the shuffling of list at the top of each epoch, we will
    # see all training samples eventually, but will skip an amount
    # less than batch_size during each epoch
    """

    nbatches, n_skipped_per_epoch = divmod(len(img_names), config.BATCH_SIZE)

    count = 1
    epoch = 0

    while 1:
        epoch += 1
        i, j = 0, config.BATCH_SIZE

        for _ in range(nbatches):
            img_names_batch = img_names[i:j]
            try:
                #get images, ground truths and original color images
                imgs, gts, imgs_only_resized = read_image_and_gt_with_original(img_names_batch, data, config, base)
                yield (imgs, gts, imgs_only_resized)
            except IOError as err:
                print("IOError ",err)
                count -= 1

            i = j
            j += config.BATCH_SIZE
            count += 1
