import os
import cv2
import numpy as np
from scipy.io import loadmat

from impl.toolbox import conv_tri, grad2
from ctypes import *


# NOTE:
#    In NMS, `if edge < interp: out = 0`, I found that sometimes edge is very close to interp.
#    `edge = 10e-8` and `interp = 11e-8` in C, while `edge = 10e-8` and `interp = 9e-8` in python.
#    ** Such slight differences (11e-8 - 9e-8 = 2e-8) in precision **
#    ** would lead to very different results (`out = 0` in C and `out = edge` in python). **
#    Sadly, C implementation is not expected but needed :(
solver = cdll.LoadLibrary("cxx/lib/solve_csa.so")
c_float_pointer = POINTER(c_float)
solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer, c_int, c_int, c_float, c_int, c_int]


def nms_process_one_image(image, save_path=None, save=True,ed_si=1.01):
    """"
    :param image: numpy array, edge, model output
    :param save_path: str, save path
    :param save: bool, if True, save .png
    :return: edge
    NOTE: in MATLAB, uint8(x) means round(x).astype(uint8) in numpy
    """
    # print("*************")
    # print("DEBUG: tipo de imagen antes de conv_tri")
    # print(type(image), image.shape, image.dtype)
    # print("*************")

    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    edge = np.float32(edge)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    #ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    ori = np.mod(np.arctan2(oyy * np.sign(-oxy), oxx + 1e-5), np.pi)

    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(ed_si), int(out.shape[1]), int(out.shape[0])
    print("NMS params> ",r,s,m)
    solver.nms(out.ctypes.data_as(c_float_pointer),
               edge.ctypes.data_as(c_float_pointer),
               ori.ctypes.data_as(c_float_pointer),
               r, s, m, w, h)
    edge = np.round(out * 255).astype(np.uint8)
    if save:
        cv2.imwrite(save_path, edge)
    return edge


def nms_process(result_dir, nms_dir, key=None, file_format=".mat",
                edge_size=1.01):
    valid_formats = {".mat", ".npy", ".png"}
    assert file_format in valid_formats
    assert os.path.isdir(result_dir)
    os.makedirs(nms_dir, exist_ok=True)

    for file in os.listdir(result_dir):
        ext = os.path.splitext(file)[-1].lower()
        if ext not in valid_formats:
            continue

        save_name = os.path.join(nms_dir, f"{os.path.splitext(file)[0]}.png")
        if os.path.isfile(save_name):
            continue

        abs_path = os.path.join(result_dir, file)
        print(abs_path)
        if ext == ".mat":
            assert key is not None
            image = loadmat(abs_path)
            image = image[key]

            # Desempaquetar si es array de objetos
            if isinstance(image, np.ndarray) and image.shape == (1, 1):
                image_struct = image[0, 0]
                if isinstance(image_struct, np.ndarray):
                    image_struct = image_struct[0]
                boundaries = image_struct['Boundaries']
                if isinstance(boundaries, np.ndarray):
                    if boundaries.ndim == 2:
                        image = boundaries[0, 0]
                    elif boundaries.ndim == 1:
                        image = boundaries[0]
                    else:
                        raise ValueError(f"Estructura inesperada en 'Boundaries': ndim = {boundaries.ndim}")
                else:
                    image = boundaries
            image = np.squeeze(image).astype(np.float32)

        elif ext == ".npy":
            image = np.load(abs_path).astype(np.float32)

        elif ext == ".png":
            image = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.bitwise_not(image)  # if tmp_edges background
            image = image.astype(np.float32) / 255.0  # normalizar a [0, 1]

            # image[image>0]=1.

        else:
            raise NotImplementedError(f"Formato no soportado: {ext}")

        nms_process_one_image(image, save_name, True, edge_size)


if __name__ == '__main__':
    nms_process("hed_result", "NMS_RESULT_FOLDER", key="groundTruth")