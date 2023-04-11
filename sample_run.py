from utils.deap_main import *
from utils.onnx_inf import run_onnx

if __name__ == '__main__':
    pop_size = 200
    NGEN = 200
    cxProb = 0.8
    muteProb = 0.2
    plat = 1 # 0 -> 内廊 ，1 ->中庭
    thre_area = 20000

    run(pop_size=pop_size,NGEN=NGEN,cxProb=cxProb,muteProb=muteProb,plat = plat,thre_area=thre_area)



