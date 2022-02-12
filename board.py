import pygame as pg






pg.init()
screen = pg.display.set_mode((400, 400))
screen.fill((255,255,255))
pg.display.set_caption("digit recognition")

 
drawing = False
last_pos = None
w = 1
color = (0,0,0)
 
 
def draw(event):
    global drawing, last_pos, w
 
    if event.type == pg.MOUSEMOTION:
        if (drawing):
            mouse_position = pg.mouse.get_pos()
            if last_pos is not None:
                # pg.draw.line(screen, color, last_pos, mouse_position, w)
                pg.draw.circle(screen,color,(pg.mouse.get_pos()[0],pg.mouse.get_pos()[1]),7)
            last_pos = mouse_position
    elif event.type == pg.MOUSEBUTTONUP:
        mouse_position = (0, 0)
        drawing = False
        last_pos = None
    elif event.type == pg.MOUSEBUTTONDOWN:
        drawing = True
 
 
def mainloop():
    global screen
 
    loop = 1
    while loop:
        # checks every user interaction in this list
        for event in pg.event.get():
            if event.type == pg.QUIT:
                loop = 0
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    pg.image.save(screen, "image.png")
                    return
            draw(event)
        pg.display.flip()
    pg.quit()
 
 
mainloop() 