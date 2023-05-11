import onnxruntime
import numpy as np
import os
import pandas as pd


def read_excel(data_dir):
    f_pth = os.path.join(data_dir,)
    # f_pth = os.path.join(root_dir, 'data.xlsx')
    df = pd.read_excel(f_pth, )

    if '内廊式' in df['平面形式'].to_numpy() or '中庭式' in df['平面形式'].to_numpy():
        p_type = ['内廊式', '中庭式']
        df['平面形式'] = [p_type.index(i.strip()) for i in df['平面形式']]

    np_data = df.to_numpy()
    if len(np_data[0]) == 27:
        return np_data
    else:
        sample,_ = np_data[:,:-3],np_data[:,-3:]
        return sample


def run_onnx(onnx_pth,input_tensor):
    # 加载 ONNX
    sess = onnxruntime.InferenceSession(onnx_pth)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred = sess.run([label_name], {input_name: input_tensor.astype(np.float32)})[0]

    return pred

def write_pred(data_pth,onnx_pth,output_name):
    print('reading file...')
    sample = read_excel(data_pth)
    print('pred...')
    pred = run_onnx(onnx_pth, sample)

    key_names = ['房间东西向长度', '房间南北向长度', '建筑东西向房间数', '建筑南北向房间数', '建筑层数', '建筑层高', '屋面传热系数', '外墙传热系数', '外窗类型编号', '南向窗墙比', '南向窗宽',
     '南向窗高', '南向窗台高', '北向窗墙比', '北向窗宽', '北向窗高', '北向窗台高', '东向窗墙比', '东向窗宽', '东向窗高', '东向窗台高', '西向窗墙比', '西向窗宽', '西向窗高',
     '西向窗台高', '中庭天窗比', '平面形式', '单位面积总能耗', '舒适时间占全年时间百分比', '一次性投入成本','最终面积']

    df_dict = {i: [] for i in key_names}
    for bs in range(len(sample)):
        for ele in range(len(sample[bs])):
            df_dict[key_names[ele]].append(sample[bs][ele])

        df_dict['单位面积总能耗'].append(pred[bs][0])
        df_dict['舒适时间占全年时间百分比'].append(pred[bs][1])
        df_dict['一次性投入成本'].append(pred[bs][2])

    for i in range(len(sample)):
        if df_dict['平面形式'][i] == 0:
            area = (df_dict[key_names[0]][i] * df_dict[key_names[1]][i]* df_dict[key_names[2]][i] * 2 + df_dict[key_names[0]][i] * df_dict[key_names[2]][
                i] * 2) * df_dict[key_names[4]][i]
        else:
            area = ((df_dict[key_names[0]][i] * df_dict[key_names[1]][i] * df_dict[key_names[2]][i] * (df_dict[key_names[3]][i] + 2)) * df_dict[key_names[4]][i]) - (
                    (df_dict[key_names[0]][i] * (df_dict[key_names[2]][i] - 2) - 4) * (df_dict[key_names[1]][i] * df_dict[key_names[3]][i] - 4) * (df_dict[key_names[4]][i] - 1))
        df_dict['最终面积'].append(area)

    df=pd.DataFrame(df_dict)

    df.to_excel(f'{output_name}.xlsx')



if __name__ == '__main__':

    # test输入数据
    # input_tensor0 = np.array([[ 7.4558,  8.1608,  7.0000,  0.0000,  5.0000,  3.9227,  0.0985,  0.1483,
    #          11.0000,  0.6356,  5.4302,  2.8596,  0.7522,  0.6525,  2.9087,
    #           2.3264,  0.7419,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #           0.0000,  0.0000,  0.0000,  0.0000]])
    #
    # input_tensor1 = np.array([[ 7.4266,  8.8618, 12.0000,  2.0000,  5.0000,  3.8759,  0.1092,  0.1255,
    #           8.0000,  0.6034,  5.8314,  3.0112,  0.9130,  0.5969,  1.6423,
    #           4.0911,  0.9711,  0.7819,  4.1457,  3.2216,  0.3408,  0.5767,  2.9217,
    #           2.4695,  0.9989,  0.6465,  1.0000]])
    #
    # # out0 = np.array([[ 27.0916,   0.7835, 289.0353]])
    # # out1 = np.array([[ 29.8036,   0.8065, 268.0174]])
    #
    # print(run_onnx('../final.onnx',input_tensor0))
    # print(run_onnx('../final.onnx',input_tensor1))
    write_pred('../data/data.xlsx','../final.onnx','pred')