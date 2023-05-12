import tkinter
from tkinter import ttk,filedialog
# import sv_ttk
from deap_main import run_mlp_nsga,write_result
from utils.onnx_inf import write_pred




class Pred_module(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="预测模块", padding=15)
        self.add_widgets()
        self.selected_data_path = ''
        self.selected_onnx_path = ''



    def add_widgets(self):
        self.button_select_file = ttk.Button(self, text="选择数据文件",command = self.select_data)
        self.button_select_file.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        self.title_filename = ttk.Label(self, text='')
        self.title_filename.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.button_select_model = ttk.Button(self, text="选择onnx模型文件",command = self.select_model)
        self.button_select_model.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        self.title_modelname = ttk.Label(self, text='')
        self.title_modelname.grid(row=1, column=1, padx=5, pady=10, sticky="ew")

        self.button1 = ttk.Button(self, text="输出预测文件!", command= self.run_pred)
        self.button1.grid(row=2, column=0, padx=5, pady=10, sticky="ew")

        self.title_run = ttk.Label(self, text='')
        self.title_run.grid(row=3, column=0, padx=5, pady=10, sticky="ew")

    def select_data(self):
        # 单个文件选择
        self.selected_file_path = filedialog.askopenfilename()
        self.title_filename.config(text=self.selected_file_path)
        self.title_filename.update_idletasks()

    def select_model(self):
        # 单个文件选择
        self.selected_onnx_path = filedialog.askopenfilename()
        self.title_modelname.config(text=self.selected_onnx_path)
        self.title_modelname.update_idletasks()

    def run_pred(self):
        self.title_run.config(text='计算中....')
        self.title_run.update_idletasks()

        write_pred(self.selected_file_path,self.selected_onnx_path,'pred_result')
        self.title_run.config(text='计算完成！ 结果输出至 pred_result.xlsx')
        self.title_run.update_idletasks()




class NSGA_module(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame",text= '寻优模块', padding=15)
        self.plat = tkinter.StringVar()  # 0 -> 内廊 ，1 ->中庭

        self.columnconfigure(0, weight=1)
        self.add_widgets()


    def add_widgets(self):
        self.plat_type = ttk.Radiobutton(self, text="内廊",variable=self.plat, value=0)
        self.plat_type.grid(row=0, column=0, pady=(0, 10), sticky="w")

        self.plat_type = ttk.Radiobutton(self, text="中庭",variable=self.plat, value=1)
        self.plat_type.grid(row=0, column=1, pady=10, sticky="w")
        ###############################################################################
        self.title_pop_size = ttk.Label(self,text='初始种群数')
        self.title_pop_size.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        self.box_pop_size = ttk.Spinbox(self, from_=0, to=2000, increment=1)
        self.box_pop_size.insert(0, "200")
        self.box_pop_size.grid(row=1, column=1, padx=5, pady=10, sticky="ew")

        self.title_NGEN = ttk.Label(self,text='迭代次数')
        self.title_NGEN.grid(row=1, column=2, padx=5, pady=10, sticky="ew")
        self.box_NGEN = ttk.Spinbox(self, from_=0, to=2000, increment=1)
        self.box_NGEN.insert(0, "200")
        self.box_NGEN.grid(row=1, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title10 = ttk.Label(self,text='交叉概率')
        self.title10.grid(row=2, column=0, padx=5, pady=10, sticky="ew")
        self.box_cxProb= ttk.Spinbox(self, from_=0, to=1, increment=0.01)
        self.box_cxProb.insert(0, "0.8")
        self.box_cxProb.grid(row=2, column=1, padx=5, pady=10, sticky="ew")

        self.title_muteProb = ttk.Label(self,text='变异概率')
        self.title_muteProb.grid(row=2, column=2, padx=5, pady=10, sticky="ew")
        self.box_muteProb = ttk.Spinbox(self, from_=0, to=1, increment=0.01)
        self.box_muteProb.insert(0, "0.2")
        self.box_muteProb.grid(row=2, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_thre_area = ttk.Label(self,text='最大面积约束')
        self.title_thre_area.grid(row=3, column=0, padx=5, pady=10, sticky="ew")
        self.box_thre_area = ttk.Spinbox(self, from_=-1, to=30000, increment=1)
        self.box_thre_area.insert(0, "-1")
        self.box_thre_area.grid(row=3, column=1, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_thre_room_len_ew = ttk.Label(self, text='房间东西向长度约束(6-10)')
        self.title_thre_room_len_ew.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
        self.box_thre_room_len_ew = ttk.Spinbox(self, from_=6, to=10, increment=0.5)
        self.box_thre_room_len_ew.insert(0, "-1")
        self.box_thre_room_len_ew.grid(row=4, column=1, padx=5, pady=10, sticky="ew")

        self.title_thre_room_len_ns = ttk.Label(self, text='房间南北向长度约束(6-10)')
        self.title_thre_room_len_ns.grid(row=4, column=2, padx=5, pady=10, sticky="ew")
        self.box_thre_room_len_ns = ttk.Spinbox(self, from_=6, to=10, increment=0.5)
        self.box_thre_room_len_ns.insert(0, "-1")
        self.box_thre_room_len_ns.grid(row=4, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_thre_room_num_ew = ttk.Label(self, text='东西房间数约束(4-10)')
        self.title_thre_room_num_ew.grid(row=5, column=0, padx=5, pady=10, sticky="ew")
        self.box_thre_room_num_ew = ttk.Spinbox(self, from_=4, to=10, increment=1)
        self.box_thre_room_num_ew.insert(0, "-1")
        self.box_thre_room_num_ew.grid(row=5, column=1, padx=5, pady=10, sticky="ew")

        self.title_thre_room_num_ns = ttk.Label(self, text='南北房间数约束(1-4) 内廊可忽略')
        self.title_thre_room_num_ns.grid(row=5, column=2, padx=5, pady=10, sticky="ew")
        self.box_thre_room_num_ns = ttk.Spinbox(self, from_=1, to=4, increment=1)
        self.box_thre_room_num_ns.insert(0, "-1")
        self.box_thre_room_num_ns.grid(row=5, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_thre_build_level_num = ttk.Label(self,text='层数约束(3-6)')
        self.title_thre_build_level_num.grid(row=6, column=0, padx=5, pady=10, sticky="ew")
        self.box_thre_build_level_num = ttk.Spinbox(self, from_=3, to=6, increment=1)
        self.box_thre_build_level_num.insert(0, "-1")
        self.box_thre_build_level_num.grid(row=6, column=1, padx=5, pady=10, sticky="ew")

        self.title_thre_build_level_height = ttk.Label(self,text='层高约束(3.3-4.2)')
        self.title_thre_build_level_height.grid(row=6, column=2, padx=5, pady=10, sticky="ew")
        self.box_thre_build_level_height = ttk.Spinbox(self, from_=3.3, to=4.2, increment=0.01)
        self.box_thre_build_level_height.insert(0, "-1")
        self.box_thre_build_level_height.grid(row=6, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_thre_win_ratios_s = ttk.Label(self,text='最小南向窗墙比')
        self.title_thre_win_ratios_s.grid(row=7, column=0, padx=5, pady=10, sticky="ew")
        self.box_thre_win_ratios_s = ttk.Spinbox(self, from_=0.2, to=0.8, increment=0.1)
        self.box_thre_win_ratios_s.insert(0, "-1")
        self.box_thre_win_ratios_s.grid(row=7, column=1, padx=5, pady=10, sticky="ew")

        self.title_thre_win_ratios_n = ttk.Label(self,text='最小北向窗墙比')
        self.title_thre_win_ratios_n.grid(row=7, column=2, padx=5, pady=10, sticky="ew")
        self.box_thre_win_ratios_n = ttk.Spinbox(self, from_=0.2, to=0.8, increment=0.01)
        self.box_thre_win_ratios_n.insert(0, "-1")
        self.box_thre_win_ratios_n.grid(row=7, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_thre_win_ratios_e = ttk.Label(self,text='最小东向窗墙比')
        self.title_thre_win_ratios_e.grid(row=8, column=0, padx=5, pady=10, sticky="ew")
        self.box_thre_win_ratios_e = ttk.Spinbox(self, from_=0.2, to=0.8, increment=0.1)
        self.box_thre_win_ratios_e.insert(0, "-1")
        self.box_thre_win_ratios_e.grid(row=8, column=1, padx=5, pady=10, sticky="ew")

        self.title_thre_win_ratios_w = ttk.Label(self,text='最小西向窗墙比')
        self.title_thre_win_ratios_w.grid(row=8, column=2, padx=5, pady=10, sticky="ew")
        self.box_thre_win_ratios_w = ttk.Spinbox(self, from_=0.2, to=0.8, increment=0.01)
        self.box_thre_win_ratios_w.insert(0, "-1")
        self.box_thre_win_ratios_w.grid(row=8, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_nenghao = ttk.Label(self,text='最小能耗')
        self.title_nenghao.grid(row=9, column=0, padx=5, pady=10, sticky="ew")
        self.box_nenghao = ttk.Spinbox(self, from_=23, to=50, increment=1)
        self.box_nenghao.insert(0, "23")
        self.box_nenghao.grid(row=9, column=1, padx=5, pady=10, sticky="ew")

        self.title_shushi = ttk.Label(self,text='最大舒适百分比')
        self.title_shushi.grid(row=9, column=2, padx=5, pady=10, sticky="ew")
        self.box_shushi = ttk.Spinbox(self, from_=0, to=1, increment=0.01)
        self.box_shushi.insert(0, "0.89")
        self.box_shushi.grid(row=9, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.title_min_chengben = ttk.Label(self,text='最小成本')
        self.title_min_chengben.grid(row=10, column=0, padx=5, pady=10, sticky="ew")
        self.box_min_chengben = ttk.Spinbox(self, from_=130, to=300, increment=1)
        self.box_min_chengben.insert(0, "140")
        self.box_min_chengben.grid(row=10, column=1, padx=5, pady=10, sticky="ew")

        self.title_max_chengben = ttk.Label(self,text='最大成本')
        self.title_max_chengben.grid(row=10, column=2, padx=5, pady=10, sticky="ew")
        self.box_max_chengben = ttk.Spinbox(self, from_=140, to=300, increment=1)
        self.box_max_chengben.insert(0, "300")
        self.box_max_chengben.grid(row=10, column=3, padx=5, pady=10, sticky="ew")
        ###############################################################################
        self.nsga_button = ttk.Button(self, text="开始寻优!",command=self.run_nsga)
        self.nsga_button.grid(row=11, column=1, padx=5, pady=10, sticky="ew")

        self.title_run = ttk.Label(self, text='')
        self.title_run.grid(row=12, column=0, padx=5, pady=10, sticky="ew")

    def run_nsga(self):

        self.title_run.config(text='正在执行...')
        self.title_run.update_idletasks()

        # try:
        final_pop, top1 = run_mlp_nsga(pop_size=int(self.box_pop_size.get()),
                                       NGEN=int(self.box_NGEN.get()),
                                       onnx_pth='final.onnx',
                                       cxProb=float(self.box_cxProb.get()),
                                       muteProb=float(self.box_muteProb.get()),
                                       plat=int(self.plat.get()),
                                       thre_area=int(self.box_thre_area.get()),
                                       thre_room_len_ew=float(self.box_thre_room_len_ew.get()),
                                       thre_room_len_ns=float(self.box_thre_room_len_ns.get()),
                                       thre_room_num_ew=int(self.box_thre_room_num_ew.get()),
                                       thre_room_num_ns=int(self.box_thre_room_num_ns.get()),
                                       thre_build_level_num=int(self.box_thre_build_level_num.get()),
                                       thre_build_level_height=float(self.box_thre_build_level_height.get()),
                                       thre_win_ratios=[float(self.box_thre_win_ratios_s.get()), float(self.box_thre_win_ratios_n.get()),
                                                        float(self.box_thre_win_ratios_e.get()),float(self.box_thre_win_ratios_w.get()),],
                                       pred_thresh=[int(self.box_nenghao.get()),float(self.box_shushi.get()),
                                                    int(self.box_min_chengben.get()),int(self.box_max_chengben.get()),])

        write_result(final_pop, 'final_pop')
        self.title_run.config(text='执行完毕! 结果已输出至 final_pop.xlsx ')
        self.title_run.update_idletasks()
        # except:
        #     self.title_run.config(text='执行错误，请检查变量')
        #     self.title_run.update_idletasks()

class Main_UI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=15)
        for index in range(2):
            self.columnconfigure(index, weight=1)
            self.rowconfigure(index, weight=1)

        Pred_module(self).grid(row=0, column=0, padx=(0, 10), pady=(0, 20), sticky="nsew")
        NSGA_module(self).grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nsew"
        )


def main():
    root = tkinter.Tk()
    root.title("")
    # sv_ttk.set_theme("light")
    Main_UI(root).pack()

    root.mainloop()


if __name__ == "__main__":
    main()
