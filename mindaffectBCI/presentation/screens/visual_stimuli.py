#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pyglet
import math
import numpy as np

def polar2cart(cx,cy,theta,radius):
    return (cx+radius*math.sin(theta), cy+radius*math.cos(theta))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class TriangleStrip:
    '''object to hold a single graphics quad'''
    def __init__(self, vertices, color, batch=None, group=None, visible:bool=True):
        """object to hold a GL_TRIANGLE_STRIP based drawing segment

        Args:
            vertices (list-of-tuples-of-float): list of (x,y) vertices for this strip
            color (list-of-int): rgba float base color for this quad
            batch, group : pyglet drawing batch and group
        """
        if len(color)==3 : color=color + (255,)
        self.vertices, self._color, self._visible = (vertices, color, visible)
        if batch:
            # add degenerate vertices at start/end for pyglet / openGL reasons
            # https://pyglet.readthedocs.io/en/latest/modules/graphics/index.html
            bvertices = vertices[0:2] + vertices + vertices[-2:]
            N=len(bvertices)//2
            # N.B. triangle FAN seems to be buggy
            self.vertex_list = batch.add(N,pyglet.gl.GL_TRIANGLE_STRIP, group,
                            ('v2f',bvertices),
                            ('c4B/dynamic',(self._color)*N))

    @property
    def color(self): return self._color
    
    @color.setter
    def color(self,col):
        """set the color of the object

        Args:
            col (_type_): _description_
        """        
        if not(hasattr(col,'__iter__')): col=(col,)*4
        elif len(col)==1 : col = tuple(col)*4
        elif len(col)==3 : col = tuple(col) + (255,)
        self._color = col
        if hasattr(self,'vertex_list'):
            N=len(self.vertex_list.vertices)//2
            if not self._visible: col[3]=0
            self.vertex_list.colors = (col)*N

    @property
    def visible(self): return self._visible
    
    @visible.setter
    def visible(self,visible):
        """set the visibility status of the object.
        Note: currently not working correctly!

        Args:
            visible (_type_): _description_
        """        
        self._visible = visible
        self.color = self._color

    def draw(self):
        """manually draw the strip to the screen.
        Note: you should *never* need to do this, if you add the object to a drawing batch...
        """        
        N=len(self.vertices)//2
        pyglet.graphics.draw(N,pyglet.gl.GL_TRIANGLE_STRIP,
                            ('v2f',self.vertices),
                            ('c3B',(self._color)*N))




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class Rectangle(TriangleStrip):
    def __init__(self,x,y,w,h,color=(255,255,255),batch=None, group=None):
        """object to show a simple Rectangle on the screen

        Args:
            vertices (list-of-tuples-of-float): list of (x,y) vertices for this strip
            color (list-of-int): rgba float base color for this quad
            batch, group : pyglet drawing batch and group
        """
        vertices = self.get_vertices(x,y,w,h)
        super().__init__(vertices,color,batch,group)

    def get_vertices(self,x,y,w,h):
        """get the vertices needed to make a rectangle

        Args:
            x (float): starting x, relative to straight up
            y (float): starting y, relative to cx,cy
            w (float): segment width (so x -> x+w)
            h (float): segment width (so y -> y+h)
        """
        #           lb   +    lt   +    rb   +    rt
        vertices = [x,y] + [x,y+h] + [x+w,y] + [x+w,y+h]
        return vertices


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class PieSegment(TriangleStrip):
    def __init__(self,cx,cy,theta,radius,w,h,color=(255,255,255),n=None,batch=None, group=None):
        vertices = self.get_vertices(cx,cy,theta,radius,w,h,n)
        #print("Seg: ({},{}) {}-{}, {}-{} = vert={}\n".format(cx,cy,theta,theta+w,radius,radius+h,vertices))
        super().__init__(vertices,color,batch,group)

    def get_vertices(self,cx,cy,theta,radius,w,h,n):
        """get the vertices needed to make a pie-segment, or wedge

        Args:
            theta (float): starting angle, relative to straight up
            radius (float): starting radius, relative to cx,cy
            w (float): segment angular width (so theta -> theta+w)
            h (float): segment radial width (so radius -> radisu+h)
        """
        if n is None: n = max(4, int(30 * w / 2 / np.pi) ) 
        vertices = []        
        for t in np.linspace(theta,theta+w,n+1):
            it = polar2cart(cx,cy, t,   radius)         # inner
            ot = polar2cart(cx,cy, t,   radius+h)       # outer
            vertices += it + ot
        return vertices



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def invert_color(col, ave_color=(127,127,127)):
    # invert color, keep alpha
    col = tuple([min(255,max(0,ave_color[i]-(c-ave_color[i]))) for i,c in enumerate(col[:3])]) + col[3:]
    return col

class Checkerboard:
    def __init__(self,x,y,w,h, n=1, nx=5, ny=5, color=(255,255,255),ave_color=(127,127,127),visible:bool=True,batch=None, group=None):
        """make a checkerboard object, with nx rows and ny cols of alternating color, like a chess board

        Args:
            x,y,w,h (float): the bounding box for the checkerboard
            n (int, optional): _description_. Defaults to 1.
            nx (int, optional): the number of horizontial bands. Defaults to 5.
            ny (int, optional): the number of vertical bands. Defaults to 5.
            color (tuple, optional): the *base* color of the checks, used for the 'white' checks. Defaults to (255,255,255).
            ave_color (tuple, optional): the average color of the checks, used to compute the 'black' checks as : black = ave + (white-ave). Defaults to (127,127,127).
            visible (bool, optional): the initial visibility status of this checkerboard. Defaults to True.
            batch (_type_, optional): graphics batch for this check. Defaults to None.
            group (_type_, optional): graphics group for this check. Defaults to None.
        """        
        self.ave_color = tuple(ave_color)
        self._color, self._visible = (tuple(color), visible)
        self.make_squares(x,y,w,h,color,nx,ny,batch=batch,group=group)

    def make_squares(self,x,y,w,h,color,nx=3,ny=3,batch=None,group=None):
        """make a checkboard of squares

        Args:
            x,y,w,h (float): the bounding box for the checkerboard
            nx (int): nx>=0 - number of checks in the block, nx<= size in pixels of one-band. Default to 3
            ny (int): ny>=0 - number of check in the block, ny<=0 size in pixels of one check. Default to 3
            batch (): batch to add this checkerboard to
            group (): group to add this checkerboard to
        """
        if nx<0 : nx = int(w / -nx)
        dw = w/nx
        if ny<0 : ny = int(h / -ny)
        dh = h/ny
        self.checkerboard = [ [None for _ in range(ny)] for _ in range(nx) ] # list of lists
        for i in range(nx):
            xi  = x + i*dw
            for j in range(ny):
                yj = y + j*dh
                isblack = (i+j) % 2 == 0 # get the type of this check
                # set the color of this check
                col = color if isblack else invert_color(color,self.ave_color)
                # create the check

                check = Rectangle(xi,yj, dw, dh, color=col, batch=batch, group=group)
                # store it
                self.checkerboard[i][j] = check
        return self


    @property
    def color(self): return self._color
    
    @color.setter
    def color(self,color):
        """set the color of the checkerboard.  

        Args:
            color ((3/4-tuple)): This is the 'white' cell color, the 'black' cells have the appropriate inverse color.
        """        
        self._color = tuple(color) if len(color)==4 else tuple(color)+(255,)
        for i,row in enumerate(self.checkerboard):
            for j,seg in enumerate(row):
                isblack = (i+j) % 2 == 0 # get the type of this check
                # set the color of this check
                col = self._color if isblack else invert_color(self._color,self.ave_color)
                if not self._visible: col = col[:3]+(0,)
                seg.color = col

    @property
    def visible(self): return self._visible
    
    @visible.setter
    def visible(self,visible):
        self._visible = visible
        self.color = self._color

    def draw(self):
        """manually draw the strip to the screen.
        Note: you should *never* need to do this, if you add the object to a drawing batch...
        """        
        for row in self.checkerboard:
            for seg in row:
                seg.draw()



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class CheckerboardSegment:
    def __init__(self,cx,cy,theta,radius,w,h, n=1, nx=3, ny=3, color=(255,255,255),ave_color=(127,127,127),visible:bool=True,batch=None, group=None):
        """make a checkeboard type Pie-Segment or Wedge
        Args:
            cx,cy (float): the center of the circle which defines the 'pie'
            theta (float): starting angle, relative to straight up
            radius (float): starting radius, relative to cx,cy
            n (int, optional): _description_. Defaults to 1.
            nx (int, optional): the number of horizontial bands. Defaults to 5.
            ny (int, optional): the number of vertical bands. Defaults to 5.
            color (tuple, optional): the *base* color of the checks, used for the 'white' checks. Defaults to (255,255,255).
            ave_color (tuple, optional): the average color of the checks, used to compute the 'black' checks as : black = ave + (white-ave). Defaults to (127,127,127).
            visible (bool, optional): the initial visibility status of this checkerboard. Defaults to True.
            batch (_type_, optional): graphics batch for this check. Defaults to None.
            group (_type_, optional): graphics group for this check. Defaults to None.
        """        
        self.ave_color = ave_color
        self._color, self._visible = (color, visible)
        self.make_squares(cx,cy,theta,radius,w,h,color,n,nx,ny,batch=batch,group=group)

    def make_squares(self,cx,cy,theta,radius,w,h,color,n=1,nx=3,ny=3,batch=None,group=None):
        """make a pie-segment checkboard

        Args:
            cx,cy (float): the center of the circle which defines the 'pie'
            theta (float): starting angle, relative to straight up
            radius (float): starting radius, relative to cx,cy
            w (float): segment angular width (so theta -> theta+w)
            h (float): segment radial width (so radius -> radisu+h)
            nx (int): number of checks in the angular size. Default to 3
            ny (int): number of check in the radial size. Default to 3
            n (int): number points in the raster for each check
            batch (): batch to add this checkerboard to
            group (): group to add this checkerboard to

        TODO[x]: special case for central circle with radius==0 & angle>pi
        [] : better central bulls-eye implementation
        """
        if radius == 0 and w>np.pi: # center sector is special case
            self.make_bullseye(cx,cy,theta,radius,w,h,color,n,nx,ny,batch,group)
            return self

        dtheta = w/nx
        dradius = h/ny         
        self.checkerboard = [ [None for _ in range(ny)] for _ in range(nx) ] # list of lists
        for i in range(nx):
            t  = theta + i*dtheta
            for j in range(ny):
                r = radius + j*dradius
                isblack = (i+j) % 2 == 0 # get the type of this check
                # set the color of this check
                col = color if isblack else invert_color(color,self.ave_color)
                # create the check
                seg = PieSegment(cx, cy, t, r, dtheta, dradius, color=col,
                                n=n, batch=batch, group=group)
                # store it
                self.checkerboard[i][j] = seg
        return self


    def make_bullseye(self,cx,cy,theta,radius,w,h,color,n=1,nx=3,ny=3,batch=None,group=None):
        """make a pie-segment for the center, i.e. starting at radius = 0

        Args:
            cx,cy (float): the center of the circle which defines the 'pie'
            theta (float): starting angle, relative to straight up
            radius (float): starting radius, relative to cx,cy
            w (float): segment angular width (so theta -> theta+w)
            h (float): segment radial width (so radius -> radisu+h)
            nx (int): number of checks in the angular size. Default to 3
            ny (int): number of check in the radial size. Default to 3
            n (int): number points in the raster for each check
            batch (): batch to add this checkerboard to
            group (): group to add this checkerboard to

        TODO[]: special case for central circle with radius==0 & angle>pi
        """
        dtheta = w/nx
        dradius = h/ny         
        check_sz = h / np.sqrt(2)
        # BODGE: make a simple checkerboard square that fits in the circle
        cb = Checkerboard(cx-check_sz,cy-check_sz,2*check_sz,2*check_sz,
                          nx=nx, ny=ny, color=color,batch=batch,group=group)
        # and then extract and return the checks
        self.checkerboard = cb.checkerboard
        return self

    @property
    def color(self): return self._color
    
    @color.setter
    def color(self,color):
        self._color = tuple(color) if len(color)==4 else tuple(color)+(255,)
        for i,row in enumerate(self.checkerboard):
            for j,seg in enumerate(row):
                isblack = (i+j) % 2 == 0 # get the type of this check
                # set the color of this check
                col = self._color if isblack else invert_color(self._color,self.ave_color)
                if not self._visible : col = col[:3]+(0,)
                seg.color = col

    @property
    def visible(self): return self._visible
    
    @visible.setter
    def visible(self,visible):
        self._visible = visible
        self.color = self._color

    def draw(self):
        for row in self.checkerboard:
            for seg in row:
                seg.draw()
