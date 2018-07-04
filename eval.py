# coding=utf-8
# ! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper

# Parameters
# ==================================================
word_char_chose = 'word'
if word_char_chose =='word':
    max_document_length = 39
    tf.flags.DEFINE_string("eval_filepath", "/home/zhangyu9/下载/魔镜杯/trans_test.txt",
                           "Evaluate on this data (Default: None)")
elif word_char_chose =='char':
    max_document_length = 58
    tf.flags.DEFINE_string("eval_filepath", "/home/zhangyu9/下载/魔镜杯/trans_train.train",
                           "Evaluate on this data (Default: None)")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocab_filepath",
                       "/home/zhangyu9/桌面/local_to_server/deep-siamese-text-similarity/runs/1530598197/checkpoints/vocab",
                       "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model",
                       "/home/zhangyu9/桌面/local_to_server/deep-siamese-text-similarity/runs/1530598197/checkpoints/model-66000",
                       "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath == None or FLAGS.vocab_filepath == None or FLAGS.model == None:
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test, x2_test, y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, max_document_length)

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
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        # predictions = graph.get_operation_by_name("output/distance").outputs[0]
        predictions = graph.get_operation_by_name("output/out").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        # emb = graph.get_operation_by_name("embedding/W").outputs[0]
        # embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test, x2_test, y_test)), 2 * FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        # all_predictions = []
        # all_d=[]
        # for db in batches:
        #     x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
        #     batch_predictions, batch_acc, batch_sim = sess.run([predictions,accuracy,sim], {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
        #     all_predictions = np.concatenate([all_predictions, batch_predictions])
        #     print(batch_predictions)
        #     all_d = np.concatenate([all_d, batch_sim])
        #     print("DEV acc {}".format(batch_acc))
        # for ex in all_predictions:
        #     print ex
        # correct_predictions = float(np.mean(all_d == y_test))
        # print("Accuracy: {:g}".format(correct_predictions))

        all_y_pre = []
        all_d = []
        count = 0
        for db in batches:
            count += 1
            x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
            batch_predictions, batch_acc, batch_sim = sess.run([predictions, accuracy, sim],
                                                               {input_x1: x1_dev_b, input_x2: x2_dev_b,
                                                                input_y: y_dev_b, dropout_keep_prob: 1.0})
            all_y_pre = np.concatenate([all_y_pre, batch_predictions])
            all_d = np.concatenate([all_d, batch_sim])

        print 'count ', count
        correct_predictions = float(np.mean(all_d == y_test))
        print("Accuracy: {:g}".format(correct_predictions), '\n')

        import time

        now_t = time.strftime('%m.%d_%H:%M')
        file_path = '/home/zhangyu9/下载/魔镜杯/submission_predict/submission_%s.csv' % now_t
        print file_path, '\n'
        with open(file_path, 'w+') as f:
            f.write('y_pre\n')
            for i in all_y_pre:
                f.write(str(i) + '\n')
