import keras.backend as K
from keras import optimizers
import tensorflow as tf
import os
import cv2
import time
import numpy as np
import pprint

from libs.config.create_config import load_dict
from libs.model.squeezeDet import  SqueezeDet
from libs.model.modelLoading import load_only_possible_weights
from libs.model.evaluation import filter_batch, filter_prediction


CONFIG = "libs\\config\\squeeze.config"

init_file = "log\\model.10-8.73.hdf5"

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)

cfg = load_dict(CONFIG)

squeeze = SqueezeDet(cfg)

adam = optimizers.Adam(lr=0.0001, clipnorm=cfg.MAX_GRAD_NORM)
squeeze.model.compile(optimizer=adam,
	                  loss=[squeeze.loss],
	                  metrics=[squeeze.bbox_loss,
	                           squeeze.class_loss,
	                           squeeze.conf_loss,
	                           squeeze.loss_without_regularization])

model = squeeze.model
load_only_possible_weights(model, init_file, verbose=False)


cap = cv2.VideoCapture(0)

while True:
	# Capture frame-by-frame
	ret, frame = cap.read()
	# Display the resulting frame
	img = cv2.resize(frame,(624,768))
	# cv2.imshow('frame',img)

	img = np.reshape(img,[1,624,768,3])
	start_time = time.time() # start time of the loop
	predictions = model.predict(img)#,batch_size=None, verbose=0, steps=None)
	boxes , classes, scores = filter_batch(predictions, cfg)	
	# print("boxes ",boxes)
	# print("classes ",classes)
	# print("scores ",scores)
	
	print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
	cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()