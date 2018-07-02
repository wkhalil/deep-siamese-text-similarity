#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-6-27 下午3:49
# @Author  : lihui
# @File    : test.py
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
import math
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "corpus_test.txt", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1530450443/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1530450443/checkpoints/model-71000", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test,x2_test,y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 39)
out_put_file=open('submission1.csv','w')
out_put_file.write('y_pre'+'\n')
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/out").outputs[0]

        # accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        #emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test,x2_test)), 2*FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]
        for db in batches:
            x1_dev_b,x2_dev_b = zip(*db)
            batch_predictions= sess.run([predictions], {input_x1: x1_dev_b, input_x2: x2_dev_b,dropout_keep_prob: 1.0})
            out_put_file.write('\n'.join([str(i) for i in batch_predictions[0]]))
            out_put_file.write('\n')
            # all_predictions = np.concatenate([all_predictions, batch_predictions])
            print(batch_predictions[0])
            # all_d = np.concatenate([all_d, batch_sim])
            # print("DEV acc {}".format(batch_acc))
        # for ex in all_predictions:
        #     print ex
        # correct_predictions = float(np.mean(all_d == y_test))
        # print("Accuracy: {:g}".format(correct_predictions))
