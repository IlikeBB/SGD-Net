import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.colors 
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pandastable import Table, TableModel
from MNIs.SGD_Atlas import main2
from SGD_main import main
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

class SGD_Net():
    def __init__(self):
        self.root = Tk()
        self.root.title("SGD-NET - Predict GUI")
        self.root.geometry('1650x920')
        self.root.resizable(width=0, height=0)
        if True:#init frame config
            self.left_frame = LabelFrame(self.root, text= "Tools Area", padx=5, pady=5)
            self.left_frame.grid(padx=2, pady=2, row=0,column=0, rowspan=4,sticky='ewns')

            self.bottom_left_frame = LabelFrame(self.root, text= "Processing Info", padx=5, pady=5)
            self.bottom_left_frame.grid(padx=2, pady=2, row=4, rowspan=2,column=0,sticky='ewns')

            self.top_middle_frame = LabelFrame(self.root, text='Info Area', padx=10, pady=5)
            self.top_middle_frame.grid(padx=2, pady=2, row=0, column=1,sticky='ewns')

            self.middle_middel_frame = LabelFrame(self.root, text='Atlas Pattern List', padx=10, pady=5)
            # self.middle_middel_frame.place(height=250, width=250)
            self.middle_middel_frame.grid(padx=2, pady=2, row=1, rowspan=4,column=1, sticky='ewns')
            self.middle_middel_frame.columnconfigure(1, weight=1) 
            self.middle_middel_frame.rowconfigure(1, weight=1)
            
            self.bottom_middle_frame = LabelFrame(self.root, text="Display Area", padx=10, pady=5)
            # self.bottom_middle_frame.grid(padx=10, pady=10, row=2, column=1, rowspan=4,sticky=W+S)
            self.bottom_middle_frame.grid(padx=2, pady=2, row=5, column=1,sticky='wens')
            self.bottom_middle_frame.columnconfigure(1, weight=1) 
            self.bottom_middle_frame.rowconfigure(1, weight=1)

            self.top_right_frame = LabelFrame(self.root, text="Atlas Display Area", padx=5, pady=5)
            self.top_right_frame.grid(padx=2, pady=2, row=0, column=3, rowspan=6,sticky='ewns')
        
        if True:# init gui left
            self.tool_label = Label(self.left_frame, text=f" ", width=20)
            self.tool_label.grid(row=0,padx=10)

            self.folder_bu = Button(self.left_frame, text="Source Folder", command = self.select_folder, width=12)
            self.folder_bu.grid(row=0, padx=10, pady=15)

            self.predict_bu = Button(self.left_frame, text='Predict', command=None, state=DISABLED, width=12)
            self.predict_bu.grid(row=1,padx=10, pady=15)

            # self.atlas_bu = Button(self.left_frame, text='Atlas', command=self.Atlas_img_reader_1, state='normal', width=12)
            self.atlas_bu = Button(self.left_frame, text='Atlas', command=None, state=DISABLED, width=12)
            self.atlas_bu.grid(row=2,padx=10, pady=15)

            self.clear_bu = Button(self.left_frame, text='Clear', command=self.reset, state=DISABLED, width=12)
            self.clear_bu.grid(row=3,padx=10, pady=15)

            self.quit_bu = Button(self.left_frame, text='Exit', command=self.root.quit, width=12)
            self.quit_bu.grid(row=4, padx=10, pady=10)

        if True:# init gui bottom left
            init_state = 'None...'
            self.process_area(init_state)

        if True:# init gui top middle
            self.info_list = ['~./', 'None','None', 'None' , 'None']
            self.radio_init = IntVar()
            self.radio_init.set(0)
            print(self.radio_init.get())
            self.Radbutn = DISABLED
            self.img_info()

        if True:#init gui midde middle
            self.atlas_list()

        if True:# init gui bottom middle
            self.dwi_len = 1
            self.dwi_init_patch = 1
            self.dwi_loop_state_f, self.dwi_loop_state_b  = 'normal', 'normal'
            self.init = True
            self.pattern_len=256
            self.dwi_img_reader()

        if True:# init gui top right
            self.MNI_Atlas_stack=[]
            self.atlas_stack = ['AAL Pattern', 'BA Pattern', 'JHU Pattern', 'JUELICH Pattern']
            self.atlas_init()
            # self.toolbar.grid(row=1, column=0, columnspan=3)

            self.back_bu = Button(self.top_right_frame, text="<<")
            self.back_bu.grid(row=2, column=0)
            self.atlas_pattern = Label(self.top_right_frame, text=str(self.atlas_stack[self.radio_init.get()]), bd=1)
            self.atlas_pattern.grid(row=2, column=1)
            self.forward_bu = Button(self.top_right_frame, text=">>")
            self.forward_bu.grid(row=2, column=2)


        self.root.mainloop()

    def select_folder(self):
        self._selected = filedialog.askdirectory(initialdir='~./', title='Select Dicom Folder')
        self.info_list = [self._selected, 'None','None', 'None' , 'None']
        self.img_info()
        self.temp_label = Label(self.left_frame, text=' ')
        self.temp_label.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky=W)
        self.predict_bu = Button(self.left_frame, text='Predict', command=self.pred_dwi, state="normal", width=12)
        self.predict_bu.grid(row=1,padx=10, pady=15)
        self.clear_bu = Button(self.left_frame, text='Clear', command=self.reset, state="normal", width=12)
        self.clear_bu.grid(row=3,padx=10, pady=15)

    def img_info(self,):
        L1 = Label(self.top_middle_frame,text="Source Path: ", width=12, padx=20, anchor=E).grid(row=0, column=0, columnspan=2)
        E1 = Entry(self.top_middle_frame, width=53)
        E1.insert(0, f"{self.info_list[0]}")
        E1.grid(row=0, column=2)

        L2 = Label(self.top_middle_frame,text="Patient Name: ", width=12, padx=20, anchor=E).grid(row=1, column=0, columnspan=2)
        E2 = Entry(self.top_middle_frame, width=53)
        E2.insert(0, f"{self.info_list[1]}")
        E2.grid(row=1, column=2, columnspan=2)

        L3 = Label(self.top_middle_frame,text="Slices Number: ", width=12, padx=20, anchor=E).grid(row=2, column=0, columnspan=2)
        E3 = Entry(self.top_middle_frame, width=53)
        E3.insert(0, f"{self.info_list[2]}")
        E3.grid(row=2, column=2, columnspan=2)

        L4 = Label(self.top_middle_frame,text="L/N: ", width=12, padx=20, anchor=E).grid(row=3, column=0, columnspan=2)
        E4 = Entry(self.top_middle_frame, width=53)
        E4.insert(0, f"{self.info_list[3]}")
        E4.grid(row=3, column=2, columnspan=2)

        L5 = Label(self.top_middle_frame,text="A/P: ", width=12, padx=20, anchor=E).grid(row=4, column=0, columnspan=2)
        E5 = Entry(self.top_middle_frame, width=53)
        E5.insert(0, f"{self.info_list[4]}")
        E5.grid(row=4, column=2, columnspan=2)

        F6 = LabelFrame(self.top_middle_frame, text= "Atlas Radio", padx=5, pady=5, labelanchor='nw')
        F6.columnconfigure(0, weight=0)
        F6.rowconfigure(0, weight=0)
        F6.grid(padx=2 ,pady=15 ,row=5, column=0, columnspan=3,sticky='ws')
        
        # self.Atlas_img_reader_2()
        Radiobutton(F6, text='AAL', variable=self.radio_init, command=lambda:self.Atlas_img_reader_1(), value=0, width=10, state=self.Radbutn,anchor=N+W).grid(row=0, column=0)
        Radiobutton(F6, text='BA', variable=self.radio_init, command=lambda:self.Atlas_img_reader_1(), value=1, width=10, state=self.Radbutn, anchor=N+W).grid(row=0, column=1)
        Radiobutton(F6, text='JHU', variable=self.radio_init, command=lambda:self.Atlas_img_reader_1(), value=2, width=10, state=self.Radbutn, anchor=N+W).grid(row=0, column=2)
        Radiobutton(F6, text='JUELICH', variable=self.radio_init, command=lambda: self.Atlas_img_reader_1(), value=3, width=10, state=self.Radbutn, anchor=N+W).grid(row=0, column=3)
        # messagebox.showinfo('#############')

    def atlas_list(self,):
        self.table = Table(self.middle_middel_frame)
        self.table.show()
        # column_header = ['regions', 'Region Names', 'Lesion voxels', 'Lesion Percentage', 'Region Percnetage']
        # self.tv1 = ttk.Treeview(self.middle_middel_frame, columns=column_header, show = "headings" )
        # self.tv1.place(relheight=1, relwidth=1)
        # treescrolly = Scrollbar(self.middle_middel_frame, orient="vertical", command=self.tv1.yview) # command means update the yaxis view of the widget
        # treescrollx = Scrollbar(self.middle_middel_frame, orient="horizontal", command=self.tv1.xview) 
        # self.tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) 
        # treescrollx.pack(side="bottom", fill="x")
        # treescrolly.pack(side="right", fill="y")
        # self.tv1.pack(side = TOP, expand = 1, fill = BOTH)


        pass

    def pred_dwi(self):#predict dwi lesion
        init_state = 'Dim to Nii: Done\nPred DWI: Done'
        self.process_area(init_state)
        self.dwi_img, self.pred_mask, img_name, LN, AP=main(self._selected)
        self.info_list = [self._selected, img_name, str(self.pred_mask.shape[0]), LN , AP]
        self.img_info()
        self.atlas_bu.config(command=self.Atlas_img_reader_1, state='normal')
        self.dwi_loop_state_f, self.dwi_loop_state_b  = 'disable', 'normal'
        self.dwi_len = self.pred_mask.shape[0]
        self.pattern_len=self.pred_mask.shape[1]
        self.init = False
        self.dwi_img_reader()
        self.predict_bu.config(state=DISABLED)

    def Atlas_img_reader_1(self):#大腦俯視圖
        try: self.atlas_show.clear()
        except: pass
        # print(len(self.MNI_Atlas_stack))
        init_state = 'Dim to Nii: Done\nPred DWI: Done\nAtlas MNI: Done'
        self.process_area(init_state)
        if self.Radbutn != 'normal':
            self.Radbutn = 'normal'
            self.img_info()
        self.atlas_bu.config(state=DISABLED)
        if len(self.MNI_Atlas_stack)==0:
            self.MNI_Atlas_stack, self.MNI_Lestion, self.MNI_Atlas_sorted = main2(self.info_list[1])
        atlas, lesion = self.MNI_Atlas_stack[self.radio_init.get()], self.MNI_Lestion
        self.atlas_show = plt.figure(figsize=(8,8))
        for i in range(1, atlas.shape[0]):
            plt.subplot(9,10,i)
            plt.imshow(atlas[i-1], alpha= 0.7, cmap = matplotlib.colors.ListedColormap(['lightblue' ,'gray', 'black']))
            plt.imshow(lesion[i-1]*150, alpha=0.9, cmap = matplotlib.colors.ListedColormap(['None' ,'black', 'red']))
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(wspace =0.1, hspace =0)
        
        self.atlas_display.figure=self.atlas_show
        self.atlas_display.draw()
        self.toolbar.update()
        # self.toolbar.update()
        # self.atlas_display = FigureCanvasTkAgg(self.atlas_show, master=self.top_right_frame,)
        # self.atlas_display.get_tk_widget().grid(row=0, column=0, columnspan=3)
        # self.toolbar.destroy()
        # self.toolbar = NavigationToolbar2Tk(self.atlas_display, self .toolbarFrame).grid(row=0, column=0)
        self.atlas_pattern.config(text="")
        self.atlas_pattern.config(text=str(self.atlas_stack[self.radio_init.get()]))
        df = self.MNI_Atlas_sorted[str(self.atlas_stack[self.radio_init.get()]).replace(' Pattern','')]
        temp_ = pd.DataFrame(df)
        temp_.columns = temp_.iloc[0]
        df = temp_[1:]
        
        # df = TableModel.getSampleData(df)
        self.table = Table(self.middle_middel_frame, dataframe=df,
                                showtoolbar=False, showstatusbar=False)
        # self.table.set_default()                             
        self.table.show()
        # if True: #atlas list
        #     import pandas as pd
        #     self.tv1["columns"]=[]
        #     self.tv1.delete(*self.tv1.get_children())
        #     df = self.MNI_Atlas_sorted[str(self.atlas_list[self.radio_init.get()]).replace(' Pattern','')]
        #     temp_ = pd.DataFrame(df)
        #     temp_.columns = temp_.iloc[0]
        #     df = temp_[1:]
        #     self.tv1["column"] = list(df.columns)
        #     self.tv1["show"] = "headings"
        #     col_width = [200,400,200,200,200]
        #     for idx, column in enumerate(self.tv1["columns"]):
        #         self.tv1.heading(column, text=column) 
            # df_rows = df.to_numpy().tolist()
            # for row in df_rows:
            #     self.tv1.insert("", "end", values=row) 

    def Atlas_img_reader_2(self): #大腦正面圖
        pass

    def dwi_img_reader(self):
        dwi_fig, self.dwi_ax = plt.subplots(1,2,figsize=(5,2.7))
        dwi_ax1 = self.dwi_ax[0]
        dwi_ax2 = self.dwi_ax[1]
        if True:
            dwi_ax1.text(self.pattern_len//2, (self.pattern_len//6)*0.5, "A",fontsize=12, color="white")
            dwi_ax1.text(self.pattern_len//2, (self.pattern_len//6)*6, "P",fontsize=12, color="white")
            dwi_ax1.text((self.pattern_len//6)*0.2, self.pattern_len//2, "L",fontsize=12, color="white")
            dwi_ax1.text((self.pattern_len//6)*5.5, self.pattern_len//2, "R",fontsize=12, color="white")
            dwi_ax2.text(self.pattern_len//2, (self.pattern_len//6)*0.5, "A",fontsize=12, color="white")
            dwi_ax2.text(self.pattern_len//2, (self.pattern_len//6)*6, "P",fontsize=12, color="white")
            dwi_ax2.text((self.pattern_len//6)*0.2, self.pattern_len//2, "L",fontsize=12, color="white")
            dwi_ax2.text((self.pattern_len//6)*5.5, self.pattern_len//2, "R",fontsize=12, color="white")
        if self.pattern_len==256:
            init_img = np.ones((256,256))*120
            dwi_ax1.imshow(init_img)
            dwi_ax2.imshow(init_img)
        else:
            dwi_ax1.imshow(self.dwi_img[self.dwi_init_patch-1], cmap="bone")
            dwi_ax2.imshow(self.pred_mask[self.dwi_init_patch-1], cmap="bone")
        dwi_ax1.axis('off')
        dwi_ax2.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        canvas_dwi = FigureCanvasTkAgg(dwi_fig, master=self.bottom_middle_frame,)
        canvas_dwi.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky='ewns')
        # print(self.dwi_loop_state_b, self.dwi_loop_state_f, self.dwi_init_patch)
            # --------------------------------------------------------------------------------------------------
        if self.dwi_init_patch!=1:
            self.back_bu = Button(self.bottom_middle_frame, text="<<", state = "normal", 
                                                        command=lambda: [self.dwi_img_loop_(int(-1), plt), self.dwi_img_reader()])
        else:
            self.back_bu = Button(self.bottom_middle_frame, text="<<", state = DISABLED, 
                                                        command=lambda: [self.dwi_img_loop_(int(-1), plt), self.dwi_img_reader()])
        self.back_bu.grid(row=1, column=0)
        # --------------------------------------------------------------------------------------------------
        
        self.slice_num = Label(self.bottom_middle_frame, text=f"{self.dwi_init_patch}/{self.dwi_len}", bd=1)
        self.slice_num.grid(row=1, column=1)

        # --------------------------------------------------------------------------------------------------
        if self.dwi_init_patch!=(self.dwi_len):
            self.forward_bu = Button(self.bottom_middle_frame, text=">>", state = 'normal', 
                                    command=lambda : [self.dwi_img_loop_(int(1), plt), self.dwi_img_reader()])
        else:
            self.forward_bu = Button(self.bottom_middle_frame, text=">>", state = DISABLED, 
                                                command=lambda : [self.dwi_img_loop_(int(1), plt), self.dwi_img_reader()])
        self.forward_bu.grid(row=1, column=2)
        # --------------------------------------------------------------------------------------------------

    def dwi_img_loop_(self, count, plt):
        self.dwi_init_patch = self.dwi_init_patch+count
        plt.clf()
        plt.close("all")

    def process_area(self,init_state):
            self.process_label = Label(self.bottom_left_frame, text=f"Process Object\n"+"\n"+                                                    
                                                                                            f"{init_state} \n",justify='left', width=20)
            self.process_label.grid( padx=10, pady=2, sticky=N+W, row=0, column=0)

    def atlas_init(self):
            self.atlas_show = plt.figure(figsize=(8,8))
            for i in range(1, 90):
                plt.subplot(10,9,i)
                blank = np.ones((91,109))
                plt.imshow(blank, cmap='bone')
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
            plt.subplots_adjust(wspace =0.1, hspace =0)
            self.atlas_display =   FigureCanvasTkAgg(self.atlas_show, master=self.top_right_frame,)
            self.atlas_display.get_tk_widget().grid(row=1, column=0, columnspan=3)
            self.toolbarFrame = LabelFrame(master=self.top_right_frame)
            self.toolbarFrame.grid(row=0, column=0, columnspan=3, sticky='ewns')
            self.toolbarFrame.rowconfigure(1, weight=1)
            self.toolbar = NavigationToolbar2Tk(self.atlas_display, self .toolbarFrame)
            # self.toolbar.update()

    def  reset(self):
        self.info_list = ['~./', 'None','None', 'None' , 'None']
        self.Radbutn = DISABLED
        self.img_info()
        init_state = 'None...'
        self.table = Table(self.middle_middel_frame)
        self.atlas_list()
        self.process_label.config(text=init_state)
        self.predict_bu.config(state=DISABLED)
        self.clear_bu.config(state=DISABLED)
        self.atlas_bu.config(state=DISABLED)
        self.dwi_len = 1
        self.init ==True
        self.pattern_len=256
        self.dwi_init_patch = 1
        self.atlas_init()
        # self.atlas_display = FigureCanvasTkAgg(self.atlas_show, master=self.top_right_frame,)
        # self.atlas_display.get_tk_widget().grid(row=0, column=0, columnspan=3)
        self.slice_num.config(text=f"{self.dwi_init_patch}/{self.dwi_len}")
        self.dwi_img_reader()
        # self.tv1.delete(*self.tv1.get_children())
        # self.tv1["columns"]=[]
        self.MNI_Atlas_stack=[]
if __name__ == '__main__':
    gui_reply = SGD_Net()