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


# 加载 ONNX
sess = onnxruntime.InferenceSession('final.onnx')
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


toolbox = base.Toolbox()
neilang_threshhold_list = [[6, 10], [6, 10],
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
zhongting_threshhold_list = [[6, 10], [6, 10],
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

def create_individual_neilang():
    room_len_ew = random.uniform(6, 10)
    room_len_ns = random.uniform(6, 10)

    room_num_ew = random.randint(4, 12)
    room_num_ns = random.randint(0, 0)

    build_level_num = random.randint(3, 6)
    build_level_height = random.uniform(3.3, 4.2)

    wumian_chuanre = random.uniform(0.087063, 0.207499)
    waiqiang_chuanre = random.uniform(0.112422, 0.17051)

    win_type = random.randint(0, 11)

    win_ratio_s = random.uniform(0.2, 0.8)
    win_width_s = random.uniform(1, 6)
    win_height_s = random.uniform(2, 4.2)
    chuangtai_height_s = random.uniform(0.2, 1.2)

    win_ratio_n = random.uniform(0.2, 0.8)
    win_width_n = random.uniform(1, 6)
    win_height_n = random.uniform(2, 4.2)
    chuangtai_height_n = random.uniform(0.2, 1.2)

    win_ratio_e = random.randint(0, 0)
    win_width_e = random.randint(0, 0)
    win_height_e = random.randint(0, 0)
    chuangtai_height_e = random.randint(0, 0)

    win_ratio_w = random.randint(0, 0)
    win_width_w = random.randint(0, 0)
    win_height_w = random.randint(0, 0)
    chuangtai_height_w = random.randint(0, 0)

    zhongting_win_ratio = random.randint(0, 0)

    plat_type = random.randint(0, 0)

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

def is_valid_neilang(individual,thre_list,thre_area):
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
    area = (room_len_ew * room_len_ns * room_num_ew * 2 + room_len_ew * room_num_ew * 2) * build_level_num
    if area > thre_area:
        return False
    # print('True')
    return True

def create_individual_zhongting():
    room_len_ew = random.uniform(6, 10)
    room_len_ns = random.uniform(6, 10)

    room_num_ew = random.randint(4, 12)
    room_num_ns = random.randint(1, 4)

    build_level_num = random.randint(3, 6)
    build_level_height = random.uniform(3.3, 4.2)

    wumian_chuanre = random.uniform(0.087063, 0.207499)
    waiqiang_chuanre = random.uniform(0.112422, 0.17051)

    win_type = random.randint(0, 11)

    win_ratio_s = random.uniform(0.2, 0.8)
    win_width_s = random.uniform(1, 6)
    win_height_s = random.uniform(2, 4.2)
    chuangtai_height_s = random.uniform(0.2, 1.2)

    win_ratio_n = random.uniform(0.2, 0.8)
    win_width_n = random.uniform(1, 6)
    win_height_n = random.uniform(2, 4.2)
    chuangtai_height_n = random.uniform(0.2, 1.2)

    win_ratio_e = random.uniform(0.2, 0.8)
    win_width_e = random.uniform(1, 6)
    win_height_e = random.uniform(2, 4.2)
    chuangtai_height_e = random.uniform(0.2, 1.2)

    win_ratio_w = random.uniform(0.2, 0.8)
    win_width_w = random.uniform(1, 6)
    win_height_w = random.uniform(2, 4.2)
    chuangtai_height_w = random.uniform(0.2, 1.2)

    zhongting_win_ratio = random.uniform(0.2, 0.9)

    plat_type = random.randint(1, 1)

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

def is_valid_zhongting(individual,thre_list,thre_area):
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

    area = ((room_len_ew * room_len_ns * room_num_ew *(room_num_ns+2)) * build_level_num)-((room_len_ew*(room_num_ew-2)-4) * (room_len_ns*room_num_ns-4)*(build_level_num-1))
    # print(area)
    if area > thre_area:
        return False
    return True

# 三个目 单位面积总能耗（min）、舒适时间占全年时间百分比（max）、单位面积增量成本（min）
def evaluate(individual):

    input_tensor = np.array(individual).reshape([1,27])
    pred = sess.run([label_name], {input_name: input_tensor.astype(np.float32)})[0]
    # print(pred[0][0],pred[0][1],pred[0][2])
    return pred[0][0],pred[0][1],pred[0][2]

# 约束条件
# 需要满足的约束
# 平面形式、建筑面积、建筑东西向房间数、建筑南北向房间数、建筑层数、建筑层高

def run(pop_size,NGEN,cxProb = 0.8,muteProb=0.2,plat = 0,thre_area = 20000):


    # random.seed(256)
    # 问题定义
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox.register("evaluate", evaluate)
    if plat == 0:
        lower_limits = [neilang_threshhold_list[i][0] for i in range(len(neilang_threshhold_list))]
        upper_limits = [neilang_threshhold_list[i][1] for i in range(len(neilang_threshhold_list))]

        toolbox.register("create_individual", create_individual_neilang)
        toolbox.register("is_valid", is_valid_neilang,thre_list = neilang_threshhold_list,thre_area = thre_area)


    else:
        lower_limits = [zhongting_threshhold_list[i][0] for i in range(len(zhongting_threshhold_list))]
        upper_limits = [zhongting_threshhold_list[i][1] for i in range(len(zhongting_threshhold_list))]

        toolbox.register("create_individual", create_individual_zhongting)
        toolbox.register("is_valid", is_valid_zhongting,thre_list = zhongting_threshhold_list, thre_area=thre_area)

    toolbox.decorate("evaluate", tools.DeltaPenalty(toolbox.is_valid, delta=[9999, -9999, 9999]))
    # toolbox.register("evaluate", evaluate)


    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.create_individual)

    # 选择、交叉和变异算子
    toolbox.register("selectGen1", tools.selTournament,tournsize = 2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("selectOffs", tools.selTournamentDCD)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutPolynomialBounded,eta=20,low = lower_limits,up=upper_limits,indpb=0.1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=pop_size)
    for _ in tqdm(range(NGEN)):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxProb, mutpb=muteProb)
        # comb_pop = pop + offspring
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))

    top1 = tools.selBest(pop,1)
    print("Best individual:", top1[0], "Fitness:", top1[0].fitness.values)

    return pop


if __name__ == "__main__":

    pop_size = 200
    NGEN = 200
    cxProb = 0.8
    muteProb = 0.2
    plat = 1 # 0 -> 内廊 ，1 ->中庭
    thre_area = 20000

    run(pop_size=pop_size,NGEN=NGEN,cxProb=cxProb,muteProb=muteProb,plat = plat,thre_area=thre_area)

