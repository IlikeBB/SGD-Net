import numpy as np
import matplotlib.pyplot as pyplot
from tkinter import *
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

x = np.load('./dwi_img.npy')
y = np.load('./pred_mask.npy')
root = Tk()
root_panel = Frame(root)
fig, ax = plt.subplots(1,2)
ax1 = ax[0]
ax1.imshow(x[10])
ax1.axis('off')

ax2 = ax[1]
ax2.imshow(y[10])
ax2.axis('off')

bg = LabelFrame(root, text='draw area')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
canvas._tkcanvas.pack(side="top", fill="both", expand=1)
img_label = Label (bg, image=None)


root.mainloop()