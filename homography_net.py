"""
Architecture and loss function defined in this script for HomographyNet

Paper: https://arxiv.org/abs/1606.03798

Sheng FANG
2020-04-15
"""
import tqdm
from fslib import io_util, tf_util
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from keras import backend as K


def build_homography_net(model_cfg, input_size):
    nb_layer = len(model_cfg)
    activation_map = tf_util.activation_map()

    inputs = Input(input_size)
    x = inputs
    for layer_idx in tqdm.tqdm(range(1, nb_layer+1), desc="Creating layer"):
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
            x = Dense(units=layer_cfg["units"])(x)
            x = curr_act(x, **layer_cfg["activation"]["kwargs"])
        elif layer_cfg["name"] == "flatten":
            x = Flatten()(x)
        elif layer_cfg["name"] == "dropout":
            x = Dropout(**layer_cfg["kwargs"])(x)
        else:
            raise Exception("Unsupported layer name: {}".format(layer_cfg["name"]))

    outputs = x

    curr_model = Model(inputs=inputs, outputs=outputs)

    return curr_model


def homography_net_reg_loss():
    """
    return the loss function for
    Returns:

    """
    def loss_fn(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

    return loss_fn


def homography_net_reg_metric():
    """
    Return Mean Average Corner Error metric
    Returns:

    """
    def mace(y_true, y_pred):
        return K.mean(32 * K.sqrt(K.sum(K.square(
            K.reshape(y_pred, (-1, 4, 2)) - K.reshape(y_true, (-1, 4, 2))), axis=-1, keepdims=True)), axis=1)

    return mace


if __name__ == '__main__':
    cfg = io_util.load_json("net_config/homography_net.json")
    model = build_homography_net(cfg, (128, 128, 2))
    model.summary()
