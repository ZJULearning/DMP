# -*- coding:utf-8 -*-

# INFO model hparams
from utils.params import model_args
from utils.data_utils import load_all_data, next_batch
import tensorflow as tf
from model.DMP import DMP_Model
import numpy as np
import os
import logging
from utils.export_params import export_params

logger = logging.getLogger("model_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if model_args.log_path and len(model_args.log_path) > 0:
    file_handler = logging.FileHandler(model_args.log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
else:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def train(train_data_set, dev_data_set, model, config, session, saver_param):
    batch_size = config.batch_size
    max_sen_len = config.max_sentence_len
    dropout_keep_rate = config.dropout_keep_rate
    ckpt_path = os.path.join(config.base_dir, config.model_name, "ckpt", "model")
    best_ckpt_path = os.path.join(config.base_dir, config.model_name, "best", "model")
    summary_path = os.path.join(config.base_dir, config.model_name, "summary")

    stop_flag = False
    step = 0
    num_train = len(train_data_set)
    train_op = model.get_train_op()
    global_step = model.global_step
    model_loss = model.get_loss()
    model_acc = model.get_acc()
    model_summary = model.get_summary()

    tb_writer = tf.summary.FileWriter(summary_path)

    show_msg_interval = config.show_msg_interval
    eval_interval = config.eval_interval
    last_dev_acc = 0.0
    last_best_dev_step = 0
    show_acc = 0.0
    show_loss = 0.0
    while not stop_flag:
        batch_data = next_batch(step, batch_size, train_data_set, num_train, max_sen_len)
        batch_data["dropout_keep_rate"] = dropout_keep_rate
        feed_dict = model.get_feed_dict(batch_data)
        _, step, loss_result, acc, summary = session.run([train_op, global_step, model_loss, model_acc, model_summary], feed_dict)
        # INFO log summary
        tb_writer.add_summary(summary, step)

        # logger.info("step:{}, loss:{}, acc:{}".format(step, loss_result, acc))
        show_acc += acc
        show_loss += loss_result
        if step % show_msg_interval == 0:
            logger.info("----- step:{}, loss:{}, acc: {}".format(step, show_loss / show_msg_interval,
                                                                 show_acc / show_msg_interval))
            show_loss = 0.0
            show_acc = 0.0
        if step % eval_interval == 0:
            saver_param.save(session, ckpt_path, global_step=step)
            logger.info("+++++ starting eval model ... ")
            logger.info("+++++ last_best_dev_step: {}, last_best_dev_acc: {}".format(last_best_dev_step,
                                                                                     last_dev_acc))
            dev_acc = dev(dev_data_set, config, model, session)
            logger.info("+++++ dev acc: {}".format(dev_acc))
            if dev_acc > last_dev_acc:
                last_dev_acc = dev_acc
                last_best_dev_step = step
                saver_param.save(session, best_ckpt_path, global_step=step)
                logger.info("+++++ saving checkpoint ...")


def dev(dev_data_set, config, model, session):
    batch_size = config.batch_size
    max_sen_len = config.max_sentence_len
    num_batch = int(len(dev_data_set) / batch_size)
    if num_batch == 0:
        num_batch = 1
    total_num = num_batch * batch_size
    total_data_set = len(dev_data_set)
    total_correct = 0
    logits = model.get_logits()
    for i in range(num_batch):
        batch_data = next_batch(i, batch_size, dev_data_set, total_data_set, max_sen_len)
        batch_data["dropout_keep_rate"] = 1.0
        feed_dict = model.get_feed_dict(batch_data)
        predicts = session.run(logits, feed_dict)
        predicts_idx = np.argmax(predicts, axis=1)
        gold_labels = batch_data["marker_ids"]
        for k in range(batch_size):
            if int(predicts_idx[k]) == int(gold_labels[k]):
                total_correct += 1

    return (total_correct + 0.0) / total_num


def test(test_data_set, config, model, session):
    batch_size = config.batch_size
    max_sen_len = config.max_sentence_len
    num_batch = int(len(test_data_set) / batch_size)
    total_num = num_batch * batch_size
    total_correct = 0
    logits = model.get_logits()
    for i in range(num_batch):
        batch_data = next_batch(i, batch_size, test_data_set, total_num, max_sen_len)
        batch_data["dropout_keep_rate"] = 1.0
        feed_dict = model.get_feed_dict(batch_data)
        predicts = session.run(logits, feed_dict)
        predicts_idx = np.argmax(predicts, axis=1)
        gold_labels = batch_data["marker_ids"]
        for k in range(batch_size):
            if int(predicts_idx[k]) == int(gold_labels[k]):
                total_correct += 1

    return (total_correct + 0.0) / total_num


def init_or_restore(ckp_base_path, sess_param, saver_param):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess_param.run(init_op)
    try:
        ckpt_state = tf.train.get_checkpoint_state(ckp_base_path)
    except:
        pass
    if ckpt_state and ckpt_state.model_checkpoint_path:
        logger.info("restore from {}".format(ckpt_state.model_checkpoint_path))
        saver_param.restore(sess_param, ckpt_state.model_checkpoint_path)
    else:
        logger.info("no pre saved models")


def export(config, sess_param):
    export_params_path = config.export_params_path
    export_params(sess_param, export_params_path, logger)


if __name__ == '__main__':
    logger.info("params:\n{}".format(model_args))

    logger.info("loading data set and embedding ... ")
    embedding, train_set, dev_set, test_set = load_all_data(model_args.emb_path, model_args.data_set_path)
    logger.info("embedding size:{}, dim:{}".format(len(embedding), len(embedding[0])))
    logger.info("train set size:{}".format(len(train_set)))
    logger.info("valid set size:{}".format(len(dev_set)))
    logger.info("test set size:{}".format(len(test_set)))
    # logger.info("0-5 data:{}".format(train_set[0:5]))
    logger.info("building model ... ")
    model_obj = DMP_Model(model_args, embedding)

    ckpt_base_path = os.path.join(model_args.base_dir, model_args.model_name, "ckpt")
    best_ckpt_base_path = os.path.join(model_args.base_dir, model_args.model_name, "best")
    if not os.path.exists(ckpt_base_path):
        os.makedirs(ckpt_base_path)
    if not os.path.exists(best_ckpt_base_path):
        os.makedirs(best_ckpt_base_path)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    saver = tf.train.Saver(max_to_keep=5)

    logger.info("+++++ init or restore from checkpoint ... ")
    init_or_restore(ckpt_base_path, sess, saver)
    logger.info("mode: {}".format(model_args.mode))
    if model_args.mode == "train":
        logger.info("starting to train ... ")
        train(train_set, dev_set, model_obj, model_args, sess, saver)
    elif model_args.mode == "test":
        test(test_set, model_args, model_obj, sess)
    elif model_args.mode == "export":
        export(model_args, sess)
    else:
        NotImplementedError
    pass
