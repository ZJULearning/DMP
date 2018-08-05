# -*- coding:utf-8 -*-
import tensorflow as tf
import pickle
import gzip


def export_params(sess, saved_path, logger):
    """
    format：
        key: tensor.name
        value: dict obj
            value: numpy value
            original_name: original name of tensor
            shape: shape of tensor
    """
    t_vars = tf.trainable_variables()
    desire_t_vars = []
    for v in t_vars:
        if "encoder" in v.name:
            desire_t_vars.append(v)
    logger.info("desired_t_vars:{}".format(desire_t_vars))
    values = sess.run(desire_t_vars)
    result = {}
    for i in range(len(desire_t_vars)):
        result[desire_t_vars[i].name] = {
            "value": values[i],
            "original_name": desire_t_vars[i].name,
            "shape": desire_t_vars[i].get_shape().as_list()
        }
    saved_path_1 = saved_path + ".py2"
    saved_path_2 = saved_path + ".py3"
    temp_f_1 = gzip.open(saved_path_1, 'wb')
    temp_f_2 = gzip.open(saved_path_2, 'wb')
    pickle.dump(result, temp_f_1, 2)
    pickle.dump(result, temp_f_2)
    temp_f_1.close()
    temp_f_2.close()

    logger.info("saved.")

