import numpy as np
import cv2
from rknn.api import RKNN


ONNX_MODEL = '/home/yuyang/python_project/DORN/model/depth.onnx'
RKNN_MODEL = '/home/yuyang/python_project/DORN/model/depth.rknn'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255], target_platform='rk3588s')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target="rk3588s", eval_mem=True, perf_debug=False)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    rknn.eval_perf()
    rknn.eval_memory()


    # Set inputs
    img = cv2.imread('./dog_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)




    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img], data_format=['nhwc'])
    np.save('./onnx_resnet50v2_0.npy', outputs[0])
    x = outputs[0]
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    print('done')

    rknn.release()