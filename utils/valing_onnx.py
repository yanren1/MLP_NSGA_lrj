from utils.onnx_inf import run_onnx
import os
import pandas as pd
import numpy as np


def read_excel(data_dir):
    f_pth = os.path.join(data_dir, 'data.xlsx')
    # f_pth = os.path.join(root_dir, 'data.xlsx')
    df = pd.read_excel(f_pth, )

    # window_type = ['5+12A+5+12A+5LOWE', '5+12A+5+12A+5LOWE*2', '5+12Ar+5+12Ar+5LOWE', '5+12Ar+5+12Ar+5LOWE*2',
    #  '5+12A+5LOWE+12A+5LOWE', '5+12A+5LOWE*2+12A+5LOWE*2', '5+12Ar+5LOWE+12Ar+5LOWE', '5LOWE+12A+5LOWE+12A+5LOWE',
    #  '5+12Ar+5LOWE*2+12Ar+5LOWE*2', '5LOWE*2+12A+5LOWE*2+12A+5LOWE*2', '5LOWE+12Ar+5LOWE+12Ar+5LOWE',
    #  '5LOWE*2+12Ar+5LOWE*2+12Ar+5LOWE*2']
    # window_type = [i.strip() for i in pd.read_excel(os.path.join('data', '1323.xlsx'))['window_type']]
    p_type = ['内廊式', '中庭式']

    # df['外窗类型'] = [window_type.index(i.strip()) for i in df['外窗类型']]
    df['平面形式'] = [p_type.index(i.strip()) for i in df['平面形式']]

    np_data = df.to_numpy()

    sample,target = np_data[:,:-3],np_data[:,-3:]

    return sample,target

def onnx_val(onnx_pth, data_dir, output_name):
    sample, val_target = read_excel(data_dir)

    output = run_onnx(onnx_pth,sample)
    result_dict = {'Pred_能耗': [], 'Pred_舒适时间': [], 'Pred_增量成本': [], 'Val_能耗': [], 'Val_舒适时间': [], 'Val_增量成本': []}
    # result_dict = {f'{tag[i]}_{target_names[j]}':[] for i in range(len(tag)) for j in range(len(target_names))}

    for bs in range(len(output)):
        result_dict['Pred_能耗'].append(output[bs][0])
        result_dict['Pred_舒适时间'].append(output[bs][1])
        result_dict['Pred_增量成本'].append(output[bs][2])

        result_dict['Val_能耗'].append(val_target[bs][0])
        result_dict['Val_舒适时间'].append(val_target[bs][1])
        result_dict['Val_增量成本'].append(val_target[bs][2])

    df=pd.DataFrame(result_dict)
    val_MAPE = np.mean(np.abs((val_target - output) / val_target)) * 100

    df.to_excel(output_name + f'_MAPE_{val_MAPE}.xlsx')


if __name__ == '__main__':
    onnx_val('../final.onnx', '../data', 'result')