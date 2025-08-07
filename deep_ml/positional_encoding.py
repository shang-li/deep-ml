import numpy as np


def pos_encoding(position: int, d_model: int):
    ret_list = []
    for k in range(position):
        pos_encoding = np.where(
            np.arange(d_model) % 2 == 0,
            np.sin(k / 10000**(2 * np.arange(d_model)//2 / d_model)),
            np.cos(k / 10000**(2 * np.arange(d_model)//2 / d_model))
        )
        pos_encoding = np.float16(pos_encoding)
        ret_list.append(pos_encoding)
    return np.stack(ret_list, axis=0)
