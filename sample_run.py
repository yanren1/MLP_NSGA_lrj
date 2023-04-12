from deap_main import *
from utils.onnx_inf import run_onnx
from utils.valing_onnx import onnx_val
from utils.valing_pytorch import pytorch_val


def sample_val(onnx = 1):
    if onnx:
        onnx_val(onnx_pth = 'final.onnx',   #模型位置
                 data_dir='data',        #数据位置
                 output_name = 'result')    #输出excel名字 （无需加拓展名）
    else:
        pytorch_val(model_pth='final.pt',  #模型位置
                    data_pth='data',       #数据位置
                    output_name='result')  #输出excel名字 （无需加拓展名）


def sample_onnx_inf():
    input_tensor0 = np.array([
        [7.4558, 8.1608, 7.0000, 0.0000, 5.0000, 3.9227, 0.0985, 0.1483,
         11.0000, 0.6356, 5.4302, 2.8596, 0.7522, 0.6525, 2.9087,
         2.3264, 0.7419, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000],

        [7.4558, 8.1608, 7.0000, 0.0000, 5.0000, 3.9227, 0.0985, 0.1483,
         11.0000, 0.6356, 5.4302, 2.8596, 0.7522, 0.6525, 2.9087,
         2.3264, 0.7419, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000]])

    pred = run_onnx('final.onnx',  #模型位置
                    input_tensor0) #输入张量

    print(pred)

def sample_mlp_nsga():
    pop_size = 200  # 初始种群数
    NGEN = 200  # 迭代次数
    cxProb = 0.8  # 交叉概率
    muteProb = 0.2  # 变异概率
    plat = 1  # 0 -> 内廊 ，1 ->中庭
    thre_area = 20000  # 最大面积约束
    thre_room_num_ew = 12  # 东西房间数约束
    thre_room_num_ns = 4  # 南北房间数约束   （内廊可忽略强制为0）
    thre_build_level_num = 6  # 层数约束
    thre_build_level_height = 4.2  # 层高约束

    final_pop,top1 = run_mlp_nsga(pop_size=pop_size,
                             NGEN=NGEN,
                             onnx_pth='final.onnx',
                             cxProb=cxProb,
                             muteProb=muteProb,
                             plat=plat,
                             thre_area=thre_area,
                             thre_room_num_ew=thre_room_num_ew,
                             thre_room_num_ns=thre_room_num_ns,
                             thre_build_level_num=thre_build_level_num,
                             thre_build_level_height=thre_build_level_height)


    write_result(final_pop,'final_pop')
    write_result(top1, 'top1')



if __name__ == '__main__':
    sample_mlp_nsga()


