import pyglet
from pyglet import image
from pyglet.window import mouse
from pyglet import clock
from pyglet.gl import *
from pyglet.window import key
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import randint

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
select = image.load('imgs/color_select.png')
cellOn_pic = image.load('imgs/cell_on.png')
cellOff_pic = image.load('imgs/cell_off.png')
icon = image.load('imgs/icon.png')
back = image.load('imgs/back.png')
window = pyglet.window.Window(343, 500, 'digit recognition', fullscreen=False)
picture = np.zeros((28,28), order='C')
pictureDigit = 0
color = 1
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
window.set_icon(icon)
clk,pred = 0,0

net = Net()
net.load_state_dict(torch.load('model.pth'))

@window.event
def on_close():
    quit()

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if buttons & mouse.LEFT:
        y_offset = 450
        for i in range(28):
            x_offset = 30
            for j in range(28):
                if(x >= x_offset and x <= x_offset+10 and y >= y_offset-10 and y <= y_offset):
                    picture[i][j] = color
                x_offset += 10
            y_offset -= 10

@window.event
def on_mouse_press(x, y, buttons, modifiers):
    global color
    if buttons & mouse.LEFT:
        y_offset = 450
        for i in range(28):
            x_offset = 30
            for j in range(28):
                if(x >= x_offset and x <= x_offset+10 and y >= y_offset-10 and y <= y_offset):
                    picture[i][j] = color
                x_offset += 10
            y_offset -= 10
        if(x >= 25 and x <= 323 and y >= 21 and y <= 90):
            ClearPicture()
        if(x >= 158 and x <= 185 and y >= 137 and y <= 157):
            color = 1
        if(x >= 132 and x <= 160 and y >= 137 and y <= 157):
            color = 0
    
        
def ClearPicture():
    for i in range(28):
        for j in range(28):
            picture[i][j] = 0
    
@window.event
def on_draw():
    global clk,pred
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    clk += 1

    if(clk%5 == 0):
        inputPicture = picture.copy()
        inputPicture = torch.from_numpy(inputPicture)
        inputPicture = inputPicture.type(torch.FloatTensor)
        pred = torch.argmax(net(inputPicture.view(-1, 28*28))[0])

    window.clear()
    back.blit(0,0,0)
    label = pyglet.text.Label('PICTURE:',font_name='Times New Roman',font_size=30,color = (0,0,0,255),x=343//2,y=480,anchor_x='center', anchor_y='center')
    label.draw()
    predStr = ""
    try:
        predStr = "PREDICTION: " + str(pred)[7]
    except:
        predStr = "PREDICTION: 0"
    label = pyglet.text.Label(predStr, font_name='Times New Roman',font_size=25,color = (0,0,0,255),x=343//2-20,y=122,anchor_x='center', anchor_y='center')
    label.draw()

    y_offset = 440
    for i in range(28):
        x_offset = 30
        for j in range(28):
            if(picture[i][j] == 0):
                cellOff_pic.blit(x_offset, y_offset, 0)
            else:
                cellOn_pic.blit(x_offset, y_offset, 0)
            x_offset += 10
        y_offset -= 10

    if(color == 0):
         select.blit(132,137,0)
    elif(color == 1):
        select.blit(158,137,0)
        
def update(self):
    pass
    
pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()
