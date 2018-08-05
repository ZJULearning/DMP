# -*- coding:utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

pa = parser.add_argument

pa("--model_name", type=str, default="sentence_encoder", help="Give model name, this will name logs and checkpoints made. ")
pa("--base_dir", type=str, default="./outputs", help="base work dir for running model. ")
pa("--max_sentence_len", type=int, default=50, help="max len of sentence.")
pa("--hidden_size", type=int, default=256, help="hidden size of bi-LSTM cell.")
pa("--predict_size", type=int, default=8, help="classes num 8|14 [8].")
pa("--project_layer_num", type=int, default=5, help="num of FC layers for projecting inner representation to output.")
pa("--data_set_path", type=str, help="path of data set . ")
pa("--emb_path", type=str, help="path of embedding . ")
pa("--mode", type=str, default="train", help="train | test | export [train]. ")
pa("--batch_size", type=int, default=64, help="batch size. ")
pa("--learning_rate", type=float, default=0.001, help="learning rate. ")
pa("--dropout_keep_rate", type=float, default=0.95, help="dropout keep rate. ")
pa("--epoch", type=int, default=10, help="epoch for training . ")


pa("--show_msg_interval", type=int, default=100, help="show training message interval steps num . ")
pa("--eval_interval", type=int, default=1000, help="dev model interval steps num . ")
pa("--log_path", type=str, default="./runtime.log", help="runtime log file . ")
pa("--export_params_path", type=str, default="./exported_params.pkl.gz", help="exported params path . ")


model_args = parser.parse_args()