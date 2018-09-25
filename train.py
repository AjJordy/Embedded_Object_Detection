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
# Github: AjJordy


from libs.model.squeezeDet import  SqueezeDet
from libs.model.dataGenerator import generator_from_data_path, read_image_and_gt
from libs.model.modelLoading import load_only_possible_weights
from libs.model.multi_gpu_model_checkpoint import  ModelCheckpointMultiGPU
from libs.config.create_config import load_dict

from keras import optimizers
import tensorflow as tf
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model
import pickle
import json
import argparse
import os
import gc
import yaml

""" ----------------  Global variables can be set by optional arguments ------------- """
CONFIG = "libs\\config\\squeeze.config"
log_dir_name = 'log'

# Parameters
EPOCHS = 8
OPTIMIZER = 'adam' # "default"
CUDA_VISIBLE_DEVICES = "0"
GPUS = 1
PRINT_TIME = 0
REDUCELRONPLATEAU = True
VERBOSE= False
STEPS = 200


def train():
    """
    Def trains a Keras model of SqueezeDet and stores the checkpoint after each epoch
    """

    #create config object
    cfg = load_dict(CONFIG)

    #create subdirs for logging of checkpoints and tensorboard stuff
    checkpoint_dir = log_dir_name +"/checkpoints"
    tb_dir = log_dir_name +"/tensorboard" 

    #delete old checkpoints and tensorboard stuff
    if tf.gfile.Exists(checkpoint_dir) and cfg.init_file == 'none':
        tf.gfile.DeleteRecursively(checkpoint_dir)   

    if tf.gfile.Exists(tb_dir):
        tf.gfile.DeleteRecursively(tb_dir)

    tf.gfile.MakeDirs(tb_dir)
    tf.gfile.MakeDirs(checkpoint_dir) 


    #add stuff for documentation to config    
    cfg.EPOCHS = EPOCHS
    cfg.OPTIMIZER = OPTIMIZER
    cfg.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
    cfg.GPUS = GPUS
    cfg.REDUCELRONPLATEAU = REDUCELRONPLATEAU

    if cfg.init_file != 'none':    
        base = 'D:\\Humanoid\\squeezeDet\\Embedded_Object_Detection\\imagetagger\\TRAIN\\'
        gt_dir = 'imagetagger\\ball_ann.json'
        img_file = 'imagetagger\\train_img.txt'
               
    else:
        # ---------------------- COCO ------------------------ 
        # img_file = '.\\dataset\\images.txt'
        # gt_dir = '.\\dataset\\annotations\\instances_train2017.json'
        # gt_dir = '.\\dataset\\annotations\\ann_train_clean.json'

        # ------------------ Small COCO ------------------------ 
        # img_file = '.\\dataset\\train_small.txt'
        # gt_dir = '.\\dataset\\annotations\\train_small.json'
        # base = "D:\\Humanoid\\squeezeDet\\Embedded_Object_Detection\\dataset\\train2017_small\\"

        # ------------------ ImageTagger ------------------------ 
        img_file = 'imagetagger\\jpg\\train_jpg.txt'
        gt_dir = 'imagetagger\\train_jpg.json'
        base = "D:\\Humanoid\\squeezeDet\\Embedded_Object_Detection\\imagetagger\\jpg\\TRAIN\\"
        

    #open files with images and ground truths files with full path names
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()


    #set gpu
    if GPUS < 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    else:
        gpus = ""
        for i in range(GPUS):
            gpus +=  str(i)+","
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    #scale batch size to gpus
    cfg.BATCH_SIZE = cfg.BATCH_SIZE * GPUS
   
    if STEPS is not None:
        nbatches_train = STEPS
    else:
        nbatches_train, mod = divmod(len(img_names), cfg.BATCH_SIZE) 

    #tf config and session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess   = tf.Session(config=config)
    K.set_session(sess)

    #instantiate model
    squeeze = SqueezeDet(cfg)

    #callbacks
    cb = []

    # print some run info
    print("Number of epochs:  {}".format(EPOCHS))
    print("Batch size: {}".format(cfg.BATCH_SIZE))
    print("STEPS ",nbatches_train)

    # set optimizer
    # multiply by number of workers do adjust for increased batch size
    if OPTIMIZER == "adam":
        if cfg.init_file != 'none':
            cfg.LR= 1e-6 * GPUS
            opt = optimizers.Adam(lr=cfg.LR * GPUS, clipnorm=cfg.MAX_GRAD_NORM)            
            print("adam with learning rate ",cfg.LR)
        else:
            cfg.LR= 1e-5 * GPUS
            opt = optimizers.Adam(lr=cfg.LR * GPUS, clipnorm=cfg.MAX_GRAD_NORM)            
            print("adam with learning rate ",cfg.LR)            
    elif OPTIMIZER == "rmsprop":
        opt = optimizers.RMSprop(lr=0.001 * GPUS, clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR= 0.001 * GPUS
    elif OPTIMIZER == "adagrad":
        opt = optimizers.Adagrad(lr=1.0 * GPUS, clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR = 1 * GPUS
    # use default if nothing was given
    else:
        # create sgd with momentum and gradient clipping
        opt= optimizers.SGD(lr=cfg.LEARNING_RATE * GPUS,
                            decay=0,
                            momentum=cfg.MOMENTUM,
                            nesterov=False,
                            clipnorm=cfg.MAX_GRAD_NORM)

        cfg.LR = cfg.LEARNING_RATE  * GPUS
        print("Learning rate: {}".format(cfg.LEARNING_RATE * GPUS))

        #add manuall learning rate decay
        #lrCallback = LearningRateScheduler(schedule)
        #cb.append(lrCallback)

    # save config file to log dir
    with open( log_dir_name  +'/config.pkl', 'wb') as f:
        pickle.dump(cfg, f, pickle.HIGHEST_PROTOCOL)

    # add tensorboard callback
    tbCallBack= TensorBoard(log_dir=tb_dir,
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)

    cb.append(tbCallBack)

    # if flag was given, add reducelronplateu callback
    # Reduce learning rate when a metric has stopped improving
    if REDUCELRONPLATEAU:
        reduce_lr=ReduceLROnPlateau(monitor='loss',
                                    factor=0.1,
                                    verbose=1,
                                    patience=5,
                                    min_lr=0.0)
        cb.append(reduce_lr)

    # print keras model summary
    if VERBOSE:
        print(squeeze.model.summary())

    if cfg.init_file != "none":
        print("Weights initialized by name from {}".format(cfg.init_file))
        load_only_possible_weights(squeeze.model, cfg.init_file, verbose=VERBOSE)
        
        """ # since these layers already existed in the ckpt they got loaded, you can reinitialized them. TODO set flag for that
        for layer in squeeze.model.layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if "fire10" in layer.name or "fire11" in layer.name or "conv12" in layer.name:
                    if hasattr(v_arg, 'initializer'):
                        initializer_method = getattr(v_arg, 'initializer')
                        initializer_method.run(session=sess)
                        #print('reinitializing layer {}.{}'.format(layer.name, v))
        """
    
    #create train generator    
    with open(gt_dir,'r') as f:
        data = json.load(f)
    f.close()
    print("File read")    

    train_generator = generator_from_data_path(img_names, data, base, config=cfg)

    # early = EarlyStopping(monitor='loss',
    #                       min_delta=0,
    #                       patience=0,
    #                       verbose=0, 
    #                       mode='auto')

    #make model parallel if specified
    if GPUS > 1:
        #use multigpu model checkpoint
        ckp_saver = ModelCheckpointMultiGPU(checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5",
                                            monitor='loss',
                                            verbose=0,
                                            save_best_only=False,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=1)

        cb.append(ckp_saver)

        print("Using multi gpu support with {} GPUs".format(GPUS))
        # make the model parallel
        parallel_model = multi_gpu_model(squeeze.model, gpus=GPUS)
        parallel_model.compile(optimizer=opt,
                              loss=[squeeze.loss],
                              metrics = [squeeze.loss_without_regularization,
                                        squeeze.bbox_loss,
                                        squeeze.class_loss,
                                        squeeze.conf_loss])

        #actually do the training
        parallel_model.fit_generator(train_generator,
                                    epochs=EPOCHS,
                                    steps_per_epoch=nbatches_train,
                                    callbacks=[cb,early])
    else:
        # add a checkpoint saver
        ckp_saver = ModelCheckpoint(checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5",
                                    monitor='loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=True,
                                    mode='auto',
                                    period=1)
        cb.append(ckp_saver)

        print("Using single GPU")
        #compile model from squeeze object, loss is not a function of model directly
        squeeze.model.compile(optimizer=opt,
                              loss=[squeeze.loss],
                              metrics = [squeeze.loss_without_regularization,
                                        squeeze.bbox_loss,
                                        squeeze.class_loss,
                                        squeeze.conf_loss])

        #actually do the training
        squeeze.model.fit_generator(train_generator,
                                    epochs=EPOCHS,
                                    steps_per_epoch=nbatches_train,
                                    callbacks=cb,
                                    verbose=1)

    gc.collect()


if __name__ == "__main__":
   
    K.clear_session()

    #parse arguments
    parser = argparse.ArgumentParser(description='Train squeezeDet model.')
    parser.add_argument("--steps",    type=int,  help="steps per epoch. DEFAULT: #imgs/ batch size")
    parser.add_argument("--epochs",   type=int,  help="number of epochs. DEFAULT: 100")
    parser.add_argument("--gpus",     type=int,  help="number of GPUS to use when using multi gpu support. Overwrites gpu flag. DEFAULT: 1")
    parser.add_argument("--resume",   type=bool, help="Resumes training and does not delete old dirs. DEFAULT: False")
    parser.add_argument("--reducelr", type=bool, help="Add ReduceLrOnPlateu callback to training. DEFAULT: True")
    parser.add_argument("--verbose",  type=bool, help="Prints additional information. DEFAULT: False")
    parser.add_argument("--optimizer",help="Which optimizer to use. DEFAULT: SGD with Momentum and lr decay OPTIONS: SGD, ADAM")
    parser.add_argument("--logdir",   help="dir with checkpoints and loggings. DEFAULT: ./log")
    parser.add_argument("--img",      help="file of full path names for the training images. DEFAULT: img_train.txt")
    parser.add_argument("--gt",       help="file of full path names for the corresponding training gts. DEFAULT: gt_train.txt")
    parser.add_argument("--gpu",      help="which gpu to use. DEFAULT: 0")
    parser.add_argument("--init",     help="keras checkpoint to start training from. If argument is none, training starts from the beginnin. DEFAULT: init_weights.h5")
    parser.add_argument("--config",   help="Dictionary of all the hyperparameters. DEFAULT: squeeze.config")

    args = parser.parse_args()

    #set global variables    
    if args.logdir is not None:
        log_dir_name = args.logdir
    if args.gpu is not None:
        CUDA_VISIBLE_DEVICES = args.gpu
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.optimizer is not None:
        OPTIMIZER = args.optimizer.lower()    
    if args.gpus is not None:
        GPUS= args.gpus
    if args.reducelr is not None:
        REDUCELRONPLATEAU = args.reducelr
    if args.verbose is not None:
        VERBOSE=args.verbose
    if args.config is not None:
        CONFIG = args.config

    train()

    K.clear_session()
