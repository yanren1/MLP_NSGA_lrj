import onnxruntime
import numpy as np


def run_onnx(onnx_pth,input_tensor):
    # 加载 ONNX
    sess = onnxruntime.InferenceSession(onnx_pth)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred = sess.run([label_name], {input_name: input_tensor.astype(np.float32)})[0]

    return pred


if __name__ == '__main__':

    # test输入数据
    input_tensor0 = np.array([[ 7.4558,  8.1608,  7.0000,  0.0000,  5.0000,  3.9227,  0.0985,  0.1483,
             11.0000,  0.6356,  5.4302,  2.8596,  0.7522,  0.6525,  2.9087,
              2.3264,  0.7419,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000,  0.0000,  0.0000]])

    input_tensor1 = np.array([[ 7.4266,  8.8618, 12.0000,  2.0000,  5.0000,  3.8759,  0.1092,  0.1255,
              8.0000,  0.6034,  5.8314,  3.0112,  0.9130,  0.5969,  1.6423,
              4.0911,  0.9711,  0.7819,  4.1457,  3.2216,  0.3408,  0.5767,  2.9217,
              2.4695,  0.9989,  0.6465,  1.0000]])

    # out0 = np.array([[ 27.0916,   0.7835, 289.0353]])
    # out1 = np.array([[ 29.8036,   0.8065, 268.0174]])

    print(run_onnx('../final.onnx',input_tensor0))
    print(run_onnx('../final.onnx',input_tensor1))