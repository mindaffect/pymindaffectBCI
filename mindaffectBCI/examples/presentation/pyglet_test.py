import pyglet
window = pyglet.window.Window()
label = pyglet.text.Label('Hello, world\n ☺ ☻ ☹ ☠ ✋ ✌ ☇ ✔	✖',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2, width=window.width,
                          multiline=True,
                          anchor_x='center', anchor_y='center')

@window.event
def on_draw():
    window.clear()
    label.draw()

pyglet.app.run()