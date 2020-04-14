"""
Script to define the architecture and loss function of yolo version 1

Paper link: https://arxiv.org/abs/1506.02640

Sheng FANG
2020-04-14
"""
from fslib import io_util, tf_util
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, Activation, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model


def build_yolov1(model_cfg, input_size, grid=(7, 7), nb_box=2, nb_class=20):
    output_shape = (*grid, 5 * nb_box + nb_class)
    output_flatten = grid[0] * grid[1] * (5 * nb_box + nb_class)
    nb_layer = len(model_cfg)
    activation_map = tf_util.activation_map()

    inputs = Input(input_size)
    x = inputs
    for layer_idx in range(nb_layer):
        print("Creating layer {}/{}".format(layer_idx, nb_layer))
        layer_cfg = model_cfg[str(layer_idx)]
        if layer_cfg["name"] == "conv":
            curr_act = activation_map[layer_cfg["activation"]["name"]]
            x = Conv2D(filters=layer_cfg["filters"], kernel_size=layer_cfg["kernel_size"], strides=layer_cfg["strides"],
                   padding=layer_cfg["padding"])(x)
            if layer_cfg["batch_norm"]:
                x = BatchNormalization()(x)
            x = curr_act(x, **layer_cfg["activation"]["kwargs"])
        elif layer_cfg["name"] == "maxpooling":
            x = MaxPooling2D(pool_size=(layer_cfg["pool_size"], layer_cfg["pool_size"]),
                             strides=layer_cfg["strides"])(x)
        elif layer_cfg["name"] == "dense":
            curr_act = activation_map[layer_cfg["activation"]["name"]]
            if layer_cfg["units"]:
                curr_units = layer_cfg["units"]
            else:
                curr_units = output_flatten
            x = Dense(units=curr_units)(x)
            x = curr_act(x, **layer_cfg["activation"]["kwargs"])
        elif layer_cfg["name"] == "flatten":
            x = Flatten()(x)
        elif layer_cfg["name"] == "reshape":
            x = Reshape(output_shape)(x)
        else:
            raise Exception("Unsupported layer name: {}".format(layer_cfg["name"]))

    outputs = x

    curr_model = Model(inputs=inputs, outputs=outputs)

    return curr_model


def yolo_loss_v1_2box(nb_box=2, nb_class=20):
    """
    Return the loss function with configuration of number of classes
    Args:
        nb_box:
        nb_class:

    Returns:

    """
    def loss_fn(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        xy1_true, wh1_true, p1_true, xy2_true, wh2_true, p2_true, c_true = tf.split(y_true,
                                                                                    (2, 2, 1, 2, 2, 1, nb_class),
                                                                                    axis=-1)
        xy1_pred, wh1_pred, p1_pred, xy2_pred, wh2_pred, p2_pred, c_pred = tf.split(y_pred,
                                                                                    (2, 2, 1, 2, 2, 1, nb_class),
                                                                                    axis=-1)

        loss_xy_1 = tf.reduce_sum(tf.multiply(tf.square(tf.add(xy1_true, -xy1_pred)), p1_true))
        loss_wh_1 = tf.reduce_sum(tf.multiply(tf.square(tf.add(tf.sqrt(wh1_true), -tf.sqrt(wh1_pred))), p1_true),
                                  axis=0)
        loss_pro_1 = tf.reduce_sum(tf.multiply(tf.square(tf.add(p1_true, -p1_pred)), p1_true))

        loss_xy_2 = tf.reduce_sum(tf.multiply(tf.square(tf.add(xy2_true, -xy2_pred)), p2_true))
        loss_wh_2 = tf.reduce_sum(tf.multiply(tf.square(tf.add(tf.sqrt(wh2_true), -tf.sqrt(wh2_pred))), p2_true),
                                  axis=0)
        loss_pro_2 = tf.reduce_sum(tf.multiply(tf.square(tf.add(p2_true, -p2_pred)), p2_true))

        loss_cls_obj = tf.reduce_sum(tf.multiply(tf.square(tf.add(c_true, - c_pred)), c_true))
        loss_cls_noobj = tf.reduce_sum(tf.multiply(tf.square(tf.add(c_true, - c_pred)), tf.add(1, -c_true)))

        loss = 5 * (loss_xy_1 + loss_xy_2 + loss_wh_1 + loss_wh_2) + loss_cls_obj + 0.5 * loss_cls_noobj + \
               loss_pro_1 + loss_pro_2

        return loss / batch_size

    return loss_fn


if __name__ == '__main__':
    cfg = io_util.load_json("net_config/yolo_v1.json")
    model = build_yolov1(cfg, (448, 448, 3))
    loss_fn = yolo_loss_v1_2box()
    opt = tf.keras.optimizers.SGD(momentum=0.9, decay=0.0005)
    model.compile(loss=loss_fn, optimizer=opt)
    model.summary()

    print("debug line!")
