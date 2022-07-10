from tkinter import *
from math import floor
from numpy import argmax

from neuralNetwork import nNetwork_class
from globals import input_nodes

true_size = 28
canvas_size = 28*12
resize = canvas_size / true_size


class app(Frame):
    def __init__(self, root, NN):
        Frame.__init__(self, root)
        self.root = root

        self.__brush_size = 12
        self.__NN  = NN
        self.answer = [0.0]*10

        self.setUI()
        pass

    def draw(self, event):
        self.canvas.create_oval(event.x - self.__brush_size,
                                event.y - self.__brush_size,
                                event.x + self.__brush_size,
                                event.y + self.__brush_size,
                                fill="black", outline="black")
        
        pass
    
    def sendData(self):
        self.answer = self.__NN.query(ImageDecoder.get_input(self.canvas))
        for i in range(10):
            self.__labelDigit[i].configure(text = str(i) + ': ' + str(int(round(self.answer[i] * 100))) + '%')
        self.__label_answer.configure(text = str(argmax(self.answer)))
        pass

    def reset(self):
        self.canvas.delete("all")
        pass

    def setUI(self):
        self.root.title("Digits recognizer v0.1")
        self.pack(expand = True, fill = BOTH) 
        
        self.canvas = Canvas(self, bg="white", width = canvas_size, height = canvas_size)
        self.canvas.grid(row = 0, column = 0, rowspan = 15, padx = 50, pady = 10, sticky=NW+W+N)
        self.canvas.bind("<B1-Motion>", self.draw)

        resetBtn = Button(self, text="Reset", width = 10, padx = 10, command = self.reset)
        resetBtn.grid(row = 0, column = 1, padx = 20, pady=10)

        confirmBtn = Button(self, text="Confirm", width = 10, padx = 10, command = self.sendData)
        confirmBtn.grid(row = 1, column = 1, padx = 20, pady = 10)

        self.__labelDigit = []

        for i in range(10):
            self.__labelDigit.append(Label(self, text = str(i) + ': ' + str(round(self.answer[i], 2) * 100) + '%'))
            self.__labelDigit[i].grid(row = i+2, column = 2)

        self.__label_answer = Label(self, text = "None", font = "Arial 28")
        self.__label_answer.grid(row = 3, column = 1, rowspan = 10)
        pass
        
class ImageDecoder:
    @staticmethod
    def get_pixel(canvas, x, y):
        ids = canvas.find_overlapping(x, y, x, y)

        if len(ids) > 0:
            index = ids[-1]
            color = canvas.itemcget(index, "fill")
            color = color.upper()
            if color != '':
                return True

        return False

    @staticmethod
    def get_input(canvas):
        height = canvas_size
        width = canvas_size
        user_data = [0.0]*true_size*true_size

        for x in range(width):
            for y in range(height):
                if ImageDecoder.get_pixel(canvas, x, y):
                    user_data[floor(x/resize)+floor(y/resize)*true_size] = 1.0
        return user_data



class window:
    def __init__(self, NN):
        self = Tk()
        self.geometry("640x420+300+40")
        self.resizable(False, False)
        self.iconbitmap("./local/assests/icon.ico")

        content = app(self, NN)

        self.mainloop()
        pass