import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm
import onnxruntime
import numpy as np
from itertools import repeat
import pandas as pd
import os

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

toolbox = base.Toolbox()


def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= indpb:
            x = individual[i]
            if xl == xu:
                continue

            if type(x) == int:
                x = random.randint(xl, xu)
                individual[i] = x
                return individual,

            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x



    return individual,

def create_individual(lower_limits, upper_limits):
    room_len_ew = random.uniform(lower_limits[0], upper_limits[0])
    room_len_ns = random.uniform(lower_limits[1], upper_limits[1])

    room_num_ew = random.randint(lower_limits[2], upper_limits[2])
    room_num_ns = random.randint(lower_limits[3], upper_limits[3])

    build_level_num = random.randint(lower_limits[4], upper_limits[4])

    if lower_limits[5]!= upper_limits[5]:
        build_level_height = random.uniform(lower_limits[5], upper_limits[5])
    else:
        build_level_height = lower_limits[5]

    wumian_chuanre = random.uniform(lower_limits[6], upper_limits[6])
    waiqiang_chuanre = random.uniform(lower_limits[7], upper_limits[7])

    win_type = random.randint(lower_limits[8], upper_limits[8])

    win_ratio_s = random.uniform(lower_limits[9], upper_limits[9])
    win_width_s = random.uniform(lower_limits[10], upper_limits[10])
    win_height_s = random.uniform(lower_limits[11], upper_limits[11])
    chuangtai_height_s = random.uniform(lower_limits[12], upper_limits[12])

    win_ratio_n = random.uniform(lower_limits[13], upper_limits[13])
    win_width_n = random.uniform(lower_limits[14], upper_limits[14])
    win_height_n = random.uniform(lower_limits[15], upper_limits[15])
    chuangtai_height_n = random.uniform(lower_limits[16], upper_limits[16])

    win_ratio_e = random.uniform(lower_limits[17], upper_limits[17])
    win_width_e = random.uniform(lower_limits[18], upper_limits[18])
    win_height_e = random.uniform(lower_limits[19], upper_limits[19])
    chuangtai_height_e = random.uniform(lower_limits[20], upper_limits[20])

    win_ratio_w = random.uniform(lower_limits[21], upper_limits[21])
    win_width_w = random.uniform(lower_limits[22], upper_limits[22])
    win_height_w = random.uniform(lower_limits[23], upper_limits[23])
    chuangtai_height_w = random.uniform(lower_limits[24], upper_limits[24])

    zhongting_win_ratio = random.uniform(lower_limits[25], upper_limits[25])

    plat_type = random.randint(lower_limits[26], upper_limits[26])


    return [room_len_ew,room_len_ns,
            room_num_ew,room_num_ns,
            build_level_num,build_level_height,
            wumian_chuanre,waiqiang_chuanre,
            win_type,
            win_ratio_s,win_width_s,win_height_s,chuangtai_height_s,
            win_ratio_n, win_width_n, win_height_n, chuangtai_height_n,
            win_ratio_e, win_width_e, win_height_e, chuangtai_height_e,
            win_ratio_w, win_width_w, win_height_w, chuangtai_height_w,
            zhongting_win_ratio,plat_type]

def is_valid(individual,plat,thre_list,thre_area):
    [room_len_ew, room_len_ns,
     room_num_ew, room_num_ns,
     build_level_num, build_level_height,
     wumian_chuanre, waiqiang_chuanre,
     win_type,
     win_ratio_s, win_width_s, win_height_s, chuangtai_height_s,
     win_ratio_n, win_width_n, win_height_n, chuangtai_height_n,
     win_ratio_e, win_width_e, win_height_e, chuangtai_height_e,
     win_ratio_w, win_width_w, win_height_w, chuangtai_height_w,
     zhongting_win_ratio, plat_type] = individual
    # print(thre_area)
    for i in range(len(individual)):
        if individual[i] < thre_list[i][0] or individual[i] > thre_list[i][1]:
            return False

    if plat == 0:
        area = (room_len_ew * room_len_ns * room_num_ew * 2 + room_len_ew * room_num_ew * 2) * build_level_num
        # pos_max_area = (thre_list[0][1] * thre_list[1][1] * thre_list[2][1] * 2 + thre_list[0][1] * thre_list[2][1] * 2) * thre_list[4][1]
    else:
        area = ((room_len_ew * room_len_ns * room_num_ew *(room_num_ns+2)) * build_level_num)-((room_len_ew*(room_num_ew-2)-4) * (room_len_ns*room_num_ns-4)*(build_level_num-1))
        # pos_max_area = ((thre_list[0][1] * thre_list[1][1] * thre_list[2][1] *(thre_list[3][1]+2)) * thre_list[4][1])-((thre_list[0][1]*(thre_list[2][1]-2)-4) * (thre_list[1][1]*thre_list[3][1]-4)*(thre_list[4][1]-1))

    if area > thre_area:
        return False

    return True

# 三个目 单位面积总能耗（min）、舒适时间占全年时间百分比（max）、单位面积增量成本（min）
def evaluate(individual,sess,pred_thresh):
    bad_result =[9999, -9999, 9999]

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    input_tensor = np.array(individual).reshape([1,27])
    pred = sess.run([label_name], {input_name: input_tensor.astype(np.float32)})[0]

    if pred[0][0] < pred_thresh[0]:
        return bad_result
    if pred[0][1] > pred_thresh[1]:
        return bad_result
    if pred[0][2] < pred_thresh[2]:
        return bad_result
    if pred[0][2] > pred_thresh[3]:
        return bad_result

    return pred[0][0],pred[0][1],pred[0][2]

# 约束条件
# 需要满足的约束
# 平面形式、建筑面积、建筑东西向房间数、建筑南北向房间数、建筑层数、建筑层高

def run_mlp_nsga(pop_size, NGEN, onnx_pth = 'final.onnx',cxProb = 0.8,
                                                muteProb = 0.2,
                                                plat = 0,
                                                thre_area = 20000,
                                                thre_room_len_ew=-1,
                                                thre_room_len_ns=-1,
                                                thre_room_num_ew = 12,
                                                thre_room_num_ns = 4,
                                                thre_build_level_num = 6,
                                                thre_build_level_height=4.2,
                                                thre_win_ratios = [-1,-1,-1,-1],
                                                pred_thresh = [23,0.89,140,300]):
    # 加载 ONNX
    sess = onnxruntime.InferenceSession(onnx_pth)
    # print(pred_thresh)
    # random.seed(256)
    # 问题定义
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    toolbox.register("evaluate", evaluate,sess=sess,pred_thresh = pred_thresh)
    if plat == 0:
        thre_list = [[6, 10], [6, 10],
                     [4, 12], [0, 0],
                     [3, 6], [3.3, 4.2],
                     [0.087063, 0.207499], [0.112422, 0.17051],
                     [0, 11],
                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                     [0, 0], [0, 0], [0, 0], [0, 0],
                     [0, 0], [0, 0], [0, 0], [0, 0],
                     [0, 0],
                     [0, 0]]
        # 房间长度约束
        if thre_room_len_ew != -1:
            thre_list[0][0] = thre_room_len_ew
            thre_list[0][1] = thre_room_len_ew
        if thre_room_len_ns != -1:
            thre_list[1][0] = thre_room_len_ns
            thre_list[1][1] = thre_room_len_ns
        # 房间数量约束
        if thre_room_num_ew != -1:
            thre_list[2][0] = thre_room_num_ew
            thre_list[2][1] = thre_room_num_ew
        # 建筑层数约束
        if thre_build_level_num != -1:
            thre_list[4][0] = thre_build_level_num
            thre_list[4][1] = thre_build_level_num
        # 建筑层高约束
        if thre_build_level_height != -1:
            thre_list[5][0] = thre_build_level_height
            thre_list[5][1] = thre_build_level_height
        # 最小窗比约束
        for i in range(2):
            if thre_win_ratios[i] !=-1:
                if thre_win_ratios[i] < thre_list[9+i*4][0]:
                    continue
                if thre_win_ratios[i] > thre_list[9+i*4][1]:
                    thre_list[9 + i * 4][0] = thre_list[9+i*4][1]
                else:
                    thre_list[9 + i * 4][0] = thre_win_ratios[i]



        lower_limits = [thre_list[i][0] for i in range(len(thre_list))]
        upper_limits = [thre_list[i][1] for i in range(len(thre_list))]
        # calculate max thresh are
        pos_max_area = (thre_list[0][1] * thre_list[1][1] * thre_list[2][1] * 2 + thre_list[0][1] * thre_list[2][
            1] * 2) * thre_list[4][1]
        pos_min_area = (thre_list[0][0] * thre_list[1][0] * thre_list[2][0] * 2 + thre_list[0][0] * thre_list[2][
            0] * 2) * thre_list[4][0]

        if thre_area == -1:
            thre_area = pos_max_area
            print('Set as Default pos_max_area!')

        elif thre_area <= pos_min_area:
            thre_area = pos_min_area
            print('Your thre_area is smaller than pos_min_area!!')
            print('Now you will only get min area setup!!')

        elif pos_max_area < thre_area:
            thre_area = pos_max_area
            print('Your thre_area is larger than pos_max_area!!')

        print(f'Final thre_area = {thre_area}')
        print()

        if thre_area == pos_min_area:
            toolbox.register("create_individual", create_individual, lower_limits=lower_limits,
                             upper_limits=lower_limits)
        else:
            toolbox.register("create_individual", create_individual,lower_limits=lower_limits,
                             upper_limits=upper_limits)

        toolbox.register("is_valid", is_valid,plat=0,thre_list = thre_list,thre_area = thre_area)


    else:
        thre_list = [[6, 10], [6, 10],
                     [4, 12], [1, 4],
                     [3, 6], [3.3, 4.2],
                     [0.087063, 0.207499], [0.112422, 0.17051],
                     [0, 11],
                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                     [0.2, 0.9],
                     [1, 1]]
        # 房间长度约束
        if thre_room_len_ew != -1:
            thre_list[0][0] = thre_room_len_ew
            thre_list[0][1] = thre_room_len_ew
        if thre_room_len_ns != -1:
            thre_list[1][0] = thre_room_len_ns
            thre_list[1][1] = thre_room_len_ns
        # 房间数量约束
        if thre_room_num_ew != -1:
            thre_list[2][0] = thre_room_num_ew
            thre_list[2][1] = thre_room_num_ew
        if thre_room_num_ns != -1:
            thre_list[3][0] = thre_room_num_ns
            thre_list[3][1] = thre_room_num_ns
        # 建筑层数约束
        if thre_build_level_num != -1:
            thre_list[4][0] = thre_build_level_num
            thre_list[4][1] = thre_build_level_num
        # 建筑层高约束
        if thre_build_level_height != -1:
            thre_list[5][0] = thre_build_level_height
            thre_list[5][1] = thre_build_level_height
        # 最小窗比约束
        for i in range(4):
            if thre_win_ratios[i] !=-1:
                if thre_win_ratios[i] < thre_list[9+i*4][0]:
                    continue
                if thre_win_ratios[i] > thre_list[9+i*4][1]:
                    thre_list[9 + i * 4][0] = thre_list[9+i*4][1]
                else:
                    thre_list[9 + i * 4][0] = thre_win_ratios[i]


        lower_limits = [thre_list[i][0] for i in range(len(thre_list))]
        upper_limits = [thre_list[i][1] for i in range(len(thre_list))]

        # calculate max thresh are
        pos_max_area = ((thre_list[0][1] * thre_list[1][1] * thre_list[2][1] * (thre_list[3][1] + 2)) * thre_list[4][1]) - (
                    (thre_list[0][1] * (thre_list[2][1] - 2) - 4) * (thre_list[1][1] * thre_list[3][1] - 4) * (thre_list[4][1] - 1))

        pos_min_area = ((thre_list[0][0] * thre_list[1][0] * thre_list[2][0] * (thre_list[3][0] + 2)) * thre_list[4][0]) - (
                    (thre_list[0][0] * (thre_list[2][0] - 2) - 4) * (thre_list[1][0] * thre_list[3][0] - 4) * (thre_list[4][0] - 1))

        if thre_area == -1:
            thre_area = pos_max_area
            print('Set as Default pos_max_area!')

        elif thre_area <= pos_min_area:
            thre_area = pos_min_area
            print('Your thre_area is smaller than pos_min_area!!')
            print('Now you will only get min area setup!!')

        elif pos_max_area < thre_area:
            thre_area = pos_max_area
            print('Your thre_area is larger than pos_max_area!!')

        print(f'Final thre_area = {thre_area}')
        print()

        if thre_area == pos_min_area:
            toolbox.register("create_individual", create_individual, lower_limits=lower_limits,
                             upper_limits=lower_limits)
        else:
            toolbox.register("create_individual", create_individual,lower_limits=lower_limits,
                             upper_limits=upper_limits)

        toolbox.register("is_valid", is_valid,plat=1,thre_list = thre_list, thre_area=thre_area)

    toolbox.decorate("evaluate", tools.DeltaPenalty(toolbox.is_valid, delta=[9999, -9999, 9999]))
    # toolbox.register("evaluate", evaluate)


    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.create_individual)

    # 选择、交叉和变异算子
    # toolbox.register("selectGen1", tools.selTournament,tournsize = 2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutPolynomialBounded,eta=20,low = lower_limits,up=upper_limits,indpb=0.1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=pop_size)
    for _ in tqdm(range(NGEN)):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxProb, mutpb=muteProb)
        comb_pop = pop + offspring
        fits = toolbox.map(toolbox.evaluate, comb_pop)

        for fit, ind in zip(fits, comb_pop):
            ind.fitness.values = fit

        pop = toolbox.select(comb_pop, k=pop_size)

    top1 = tools.selBest(pop,20)
    print("Best individual:", top1[0], "Fitness:", top1[0].fitness.values)

    return pop,top1


def write_result(pop,output_name):
    key_names = ['房间东西向长度', '房间南北向长度', '建筑东西向房间数', '建筑南北向房间数', '建筑层数', '建筑层高', '屋面传热系数', '外墙传热系数', '外窗类型编号', '南向窗墙比', '南向窗宽',
     '南向窗高', '南向窗台高', '北向窗墙比', '北向窗宽', '北向窗高', '北向窗台高', '东向窗墙比', '东向窗宽', '东向窗高', '东向窗台高', '西向窗墙比', '西向窗宽', '西向窗高',
     '西向窗台高', '中庭天窗比', '平面形式', '单位面积总能耗', '舒适时间占全年时间百分比', '一次性投入成本','最终面积']

    df_dict = {i:[] for i in key_names}
    for bs in range(len(pop)):
        for ele in range(len(pop[bs])):
            df_dict[key_names[ele]].append(pop[bs][ele])

        df_dict['单位面积总能耗'].append(pop[bs].fitness.values[0])
        df_dict['舒适时间占全年时间百分比'].append(pop[bs].fitness.values[1])
        df_dict['一次性投入成本'].append(pop[bs].fitness.values[2])

    for i in range(len(pop)):
        if df_dict['平面形式'][i] == 0:
            area = (df_dict[key_names[0]][i] * df_dict[key_names[1]][i]* df_dict[key_names[2]][i] * 2 + df_dict[key_names[0]][i] * df_dict[key_names[2]][
                i] * 2) * df_dict[key_names[4]][i]
        else:
            area = ((df_dict[key_names[0]][i] * df_dict[key_names[1]][i] * df_dict[key_names[2]][i] * (df_dict[key_names[3]][i] + 2)) * df_dict[key_names[4]][i]) - (
                    (df_dict[key_names[0]][i] * (df_dict[key_names[2]][i] - 2) - 4) * (df_dict[key_names[1]][i] * df_dict[key_names[3]][i] - 4) * (df_dict[key_names[4]][i] - 1))
        df_dict['最终面积'].append(area)

    df=pd.DataFrame(df_dict)

    df.to_excel(f'{output_name}.xlsx')




if __name__ == "__main__":
    pop_size = 20   #初始种群数
    NGEN = 20       #迭代次数
    cxProb = 0.8     #交叉概率
    muteProb = 0.2   #变异概率
    plat = 0         # 0 -> 内廊 ，1 ->中庭
    thre_area = -1               # 最大面积约束
    thre_room_len_ew = 3           # 东西房间长度约束
    thre_room_len_ns = 4          # 南北房间长度约束
    thre_room_num_ew = -1           # 东西房间数约束
    thre_room_num_ns = -1            # 南北房间数约束   （内廊可忽略强制为0）
    thre_build_level_num = -1        # 层数约束
    thre_build_level_height = -1   # 层高约束
    thre_win_ratios = [0.5,-1,0.5,0.6]     #最小窗比约束 SNEW
    pred_thresh = [23,0.89,140,300]    #最小能耗、最大百分比、最小成本、最大成本


    final_pop,top1 = run_mlp_nsga(pop_size=pop_size,NGEN=NGEN,onnx_pth = 'final.onnx', cxProb = cxProb,
                                                                muteProb=muteProb,
                                                                plat = plat,
                                                                thre_area = thre_area,
                                                                thre_room_len_ew = thre_room_len_ew,
                                                                thre_room_len_ns = thre_room_len_ns,
                                                                thre_room_num_ew = thre_room_num_ew,
                                                                thre_room_num_ns = thre_room_num_ns,
                                                                thre_build_level_num = thre_build_level_num,
                                                                thre_build_level_height=thre_build_level_height,
                                                                thre_win_ratios = thre_win_ratios,
                                                                pred_thresh = pred_thresh)


    write_result(final_pop,'final_pop')
    # write_result(top1, 'top1')
