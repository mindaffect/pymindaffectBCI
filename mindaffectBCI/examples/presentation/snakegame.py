import pyglet
import random
import time

class SnakeGame:
    def __init__(self, window, grid_width:int=10, grid_height:int=10):
        self.window = window
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cells = []
        self.direction = 1
        self.startx, self.starty = max(1,self.grid_width//4), self.grid_height//2
        self.body = [(self.startx, self.starty),(self.startx, self.starty)]
        self.apple = (None, None)
        self.score = -1
        self.snake_clr = (0, 200, 0)
        self.apple_clr = (255, 0, 0)
        self.dead_clr = (255, 255, 255)
        self.extend = False
        self.death = False
        self.point = False
        self.watch = None
        self.turning = False
        self.gen_cell_base()
        self.fill_cells()

    def gen_cell_base(self):
        temp = []
        for row in range(0, self.grid_height):
            temp.append([])
            for col in range(0, self.grid_width):
                temp[row].append([0])
        return temp

    def draw_score(self):
        if self.point == True:
            self.score += 1
            self.point = False

        score = pyglet.text.Label(str(self.score),
                                  font_name = 'Times New Roman',
                                  font_size = 30,
                                  x = 0, y = self.window.height,
                                  anchor_x = 'left', anchor_y = 'top')
        score.draw()

    def fill_cells(self):
        # build the new display image
        head = self.body[-1]
        if -1 not in head and self.grid_width not in head and self.grid_height not in head and self.death == False:
            self.cells = self.gen_cell_base()
            self.fill_snake()
            self.fill_apple()

    def fill_snake(self):
        for seg in self.body:
            self.cells[seg[0]][seg[1]] = [1, self.snake_clr]

    def fill_apple(self):
        while self.apple in self.body or self.apple == (None, None):
            self.point = True
            self.extend = True
            self.apple = (random.randint(0, self.grid_height - 1), random.randint(0, self.grid_width - 1))
        
        self.cells[self.apple[0]][self.apple[1]] = [1, self.apple_clr]

    def turn(self, key):
        if not self.turning:
            if self.direction not in (0,1) and key not in (2,3):
                self.direction = key
                self.turning = True
            elif self.direction not in (2,3) and key not in (0,1):
                self.direction = key
                self.turning = True
                 
    def move(self):
        #Don't move if dead
        if self.death != True:  
            #LEFT
            if self.direction == 0:
                self.body.append((self.body[-1][0] - 1, self.body[-1][1]))
            #RIGHT
            elif self.direction == 1:
                self.body.append((self.body[-1][0] + 1, self.body[-1][1]))
            #UP
            elif self.direction == 2:
                self.body.append((self.body[-1][0], self.body[-1][1] + 1))
            #DOWN
            elif self.direction == 3:
                self.body.append((self.body[-1][0], self.body[-1][1] - 1))
            
            #Cut off end of tail unless apple just eaten
            if self.extend == False:
                self.body = self.body[1:]
            else:
                self.extend = False

        self.fill_cells()
        self.turning = False

    def pointer(self):
            if self.direction == 0:
                return (self.body[-1][0] - 1, self.body[-1][1])
            #RIGHT
            elif self.direction == 1:
                return (self.body[-1][0] + 1, self.body[-1][1])
            #UP
            elif self.direction == 2:
                return (self.body[-1][0], self.body[-1][1] + 1)
            #DOWN
            elif self.direction == 3:
                return (self.body[-1][0], self.body[-1][1] - 1)
        
    def run_rules(self):
        if self.watch != None:
            if self.body[-1] == self.watch:
                self.death = True
                if -1 not in self.watch and self.grid_height not in self.watch and self.grid_width not in self.watch:
                    self.cells[self.watch[0]][self.watch[1]] = [1, self.dead_clr]
            else:
                self.watch = None
        
        pointer = self.pointer()
        if pointer[0] <= self.grid_height and pointer[1] <= self.grid_width:
            if pointer[0] in (-1, self.grid_height) or pointer[1] in (-1, self.grid_width):
                self.watch = pointer
            elif self.cells[pointer[0]][pointer[1]][-1] == self.snake_clr:
                self.watch = pointer

    def draw(self):
        cell_width = self.window.width // self.grid_width
        cell_height = self.window.height // self.grid_height
        for row in range(0, self.grid_height):
            for col in range(0, self.grid_width):
                if self.cells[row][col][0] == 1: # occupied
                    #Get Color
                    color = self.cells[row][col][1] # color
                    #(0, 0) (0, 20) (20, 0) (20, 20)
                    square_coords = (col * cell_width,              row * cell_height,
                                     col * cell_width,              row * cell_height + cell_height,
                                     col * cell_width + cell_width, row * cell_height,
                                     col * cell_width + cell_width, row * cell_height + cell_height)

                    pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,                                                       
                                                [0, 1, 2, 1, 2, 3],
                                                ('v2i', square_coords),
                                                ('c3B', (color * 4))
                                                )

nframe=0
framespermove = 30
def draw(self, dt=0):
    global window, snake, nframe, framespermove
    nframe = nframe + 1
    if nframe % framespermove == 0 :
        snake.run_rules()
        snake.move()
    window.clear()
    snake.draw()
    snake.draw_score()

def on_key_press(symbol, modifiers):
    global snake
    if symbol in (pyglet.window.key.LEFT, pyglet.window.key.A):
        snake.turn(0)
    elif symbol in (pyglet.window.key.RIGHT, pyglet.window.key.D):
        snake.turn(1)
    elif symbol in (pyglet.window.key.UP, pyglet.window.key.W):
        snake.turn(2)
    elif symbol in (pyglet.window.key.DOWN, pyglet.window.key.S):
        snake.turn(3)

if __name__=="__main__":
    global snake
    window = pyglet.window.Window()
    window.push_handlers(on_key_press)
    snake = Snake(window.width,window.height,20)
    pyglet.clock.schedule(draw)
    pyglet.app.run()
