import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm
import onnxruntime
import numpy as np
from itertools import repeat

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
    build_level_height = random.uniform(lower_limits[5], upper_limits[5])

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
    else:
        area = ((room_len_ew * room_len_ns * room_num_ew *(room_num_ns+2)) * build_level_num)-((room_len_ew*(room_num_ew-2)-4) * (room_len_ns*room_num_ns-4)*(build_level_num-1))

    if area > thre_area:
        return False

    return True

# 三个目 单位面积总能耗（min）、舒适时间占全年时间百分比（max）、单位面积增量成本（min）
def evaluate(individual,sess):

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    input_tensor = np.array(individual).reshape([1,27])
    pred = sess.run([label_name], {input_name: input_tensor.astype(np.float32)})[0]
    # print(pred[0][0],pred[0][1],pred[0][2])
    return pred[0][0],pred[0][1],pred[0][2]

# 约束条件
# 需要满足的约束
# 平面形式、建筑面积、建筑东西向房间数、建筑南北向房间数、建筑层数、建筑层高

def run_mlp_nsga(pop_size, NGEN, onnx_pth = 'final.onnx',cxProb = 0.8,
                                                muteProb=0.2,
                                                plat = 0,
                                                thre_area = 20000,
                                                thre_room_num_ew = 12,
                                                thre_room_num_ns = 4,
                                                thre_build_level_num = 6,
                                                thre_build_level_height=4.2):
    # 加载 ONNX
    sess = onnxruntime.InferenceSession(onnx_pth)

    # random.seed(256)
    # 问题定义
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox.register("evaluate", evaluate,sess=sess)
    if plat == 0:
        neilang_threshhold_list = [[6, 10], [6, 10],
                                   [4, thre_room_num_ew], [0, 0],
                                   [3, thre_build_level_num], [3.3, thre_build_level_height],
                                   [0.087063, 0.207499], [0.112422, 0.17051],
                                   [0, 11],
                                   [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                                   [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                                   [0, 0], [0, 0], [0, 0], [0, 0],
                                   [0, 0], [0, 0], [0, 0], [0, 0],
                                   [0, 0],
                                   [0, 0]]

        lower_limits = [neilang_threshhold_list[i][0] for i in range(len(neilang_threshhold_list))]
        upper_limits = [neilang_threshhold_list[i][1] for i in range(len(neilang_threshhold_list))]

        toolbox.register("create_individual", create_individual, lower_limits=lower_limits,upper_limits=upper_limits)
        toolbox.register("is_valid", is_valid,plat=0,thre_list = neilang_threshhold_list,thre_area = thre_area)


    else:
        zhongting_threshhold_list = [[6, 10], [6, 10],
                                     [4, thre_room_num_ew], [1, thre_room_num_ns],
                                     [3, thre_build_level_num], [3.3, thre_build_level_height],
                                     [0.087063, 0.207499], [0.112422, 0.17051],
                                     [0, 11],
                                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                                     [0.2, 0.8], [1, 6], [2, 4.2], [0.2, 1.2],
                                     [0.2, 0.9],
                                     [1, 1]]

        lower_limits = [zhongting_threshhold_list[i][0] for i in range(len(zhongting_threshhold_list))]
        upper_limits = [zhongting_threshhold_list[i][1] for i in range(len(zhongting_threshhold_list))]

        toolbox.register("create_individual", create_individual,lower_limits=lower_limits,upper_limits=upper_limits)
        toolbox.register("is_valid", is_valid,plat=1,thre_list = zhongting_threshhold_list, thre_area=thre_area)

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

    top1 = tools.selBest(pop,1)
    print("Best individual:", top1[0], "Fitness:", top1[0].fitness.values)

    return pop


if __name__ == "__main__":

    pop_size = 200   #初始种群数
    NGEN = 200       #迭代次数
    cxProb = 0.8     #交叉概率
    muteProb = 0.2   #变异概率
    plat = 1         # 0 -> 内廊 ，1 ->中庭
    thre_area = 20000               # 最大面积约束
    thre_room_num_ew = 12           # 最大东西房间数约束
    thre_room_num_ns = 4            # 最大南北房间数约束   （内廊可忽略强制为0）
    thre_build_level_num = 6        # 最大层数约束
    thre_build_level_height = 4.2   # 最大层高约束

    run_mlp_nsga(pop_size=pop_size,NGEN=NGEN,onnx_pth = '../final.onnx', cxProb = 0.8,
                                                                muteProb=0.2,
                                                                plat = 0,
                                                                thre_area = 20000,
                                                                thre_room_num_ew = 12,
                                                                thre_room_num_ns = 4,
                                                                thre_build_level_num = 6,
                                                                thre_build_level_height=4.2)

