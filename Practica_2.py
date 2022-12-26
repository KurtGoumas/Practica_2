# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:45:11 2022

@author: adelu
"""

#Vamos primero a importar toa la wea.
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
from scipy import signal
import matplotlib.animation as animation
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.mplot3d import Axes3D

#Definimos los datos 
Mt= 5.972e24#kg
Rt= 637000#m
G= 6.67e-11
H= 300000#m sobre la superficie
R= Rt+H
#f= 0.5#Órbita elíptica
f= 1#circular 
#f= 2#hipérbola
#f= 2**0.5#parábola
vc= f*(G*Mt/(R))**0.5#velocidad de la órbita. Si f=1 es circular
m= 1#kg
L= m*vc*R#momento angular
mu= Mt*m/(Mt+m)#masa reducida
'''
#Ejercicio 1. Potencial efectivo en función de r. Vef(r).

def Vef(r):
    return ((L**2)/(2*r) - G*Mt*m)/(mu*r)


r= np.linspace(R*0.1,R*2,2000)
ET= np.ones(2000)*Vef(R)#La energía total como un array coon todo el mismo valor
     
plt.figure()

plt.plot(r,Vef(r))#Potencial efectivo
plt.plot(r,ET)#Energía total
plt.xlabel('r')
plt.ylabel('Vef')

plt.show()

#Nota: Los puntos donde azul corta a naranja son el apocentro y pericentro respectivamente.

#Ejercicio 2. Órbita de la elipse.

def t_elipse(z,t):
    x,y,vx,vy= z
    dzdt= [vx,vy,-G*Mt*x/((x**2 + y**2)**(3/2)),-G*Mt*y/((x**2 + y**2)**(3/2))]
    return dzdt

#Condiciones iniciales.
x0= R
vx0= 0

y0= 0
vy0= vc

t= np.linspace(0,300,25000)#tiempo de integración.

c0= [x0,y0,vx0,vy0]#lista de las condiciones iniciales
#Hacemos odeint.
sol= odeint(t_elipse,c0,t)

x= sol[:,0]
y= sol[:,1]
vx= sol[:,2]
vy= sol[:,3]


plt.figure()

plt.plot(t,x)
plt.plot(t,y)

plt.xlabel('tiempo')
plt.ylabel('posiciones')

plt.figure()

plt.plot(t,vx)
plt.plot(t,vy)

plt.xlabel('tiempo')
plt.ylabel('velocidades')


#Esto para representar cualquier cosa.
plt.figure(figsize=(4, 4))

plt.plot(x,y)
plt.plot(0,0,'+')

plt.xlabel('x')
plt.ylabel('y')

plt.show()


#Esto para representar la elipse.

plt.figure()

plt.plot(x,y)
plt.plot(0,0,'+')

plt.xlabel('x')
plt.ylabel('y')


plt.show()

#Ejercicio 3. Variamos el factor f, mira arriba.

#Ejercicio 4. Esto es ir cambiando el tiempo de integración.
'''
#Ejercicio 5. Hay que usar la expresión F= -8L^2R^2/mr^5 para la circunferencia y la elipse.

#Sacamos la aceleración en x y en y usando esta fuerza y las sustituimos en...
#la funciónn para hacer un rico odeint.

L2= 2*m*vc*R

def Vef(r):
    return -2*((L2*R)**2)/(m*r**4)


r= np.linspace(R*0.1,R*2,2000)
ET= np.ones(2000)*Vef(R)#La energía total como un array coon todo el mismo valor
     
plt.figure()

plt.plot(r,Vef(r))#Potencial efectivo
plt.plot(r,ET)#Energía total
plt.xlabel('r')
plt.ylabel('Vef')

plt.show()

def t_elipse(z,t):
    x,y,vx,vy= z
    dzdt= [vx,vy,-8*(x+R)*((L2*R)**2)/(((x+R)**2 + y**2)**3),-8*y*((L2*R)**2)/(((x+R)**2 + y**2)**3)]
    return dzdt

x0= R
vx0= 0

y0= 0
vy0= vc

t= np.linspace(0,71.338,25000)#tiempo de integración.

c0= [x0,y0,vx0,vy0]#lista de las condiciones iniciales
#Hacemos odeint.
sol= odeint(t_elipse,c0,t)

x= sol[:,0]
y= sol[:,1]
vx= sol[:,2]
vy= sol[:,3]

plt.figure(figsize=(4, 4))

plt.plot(x,y)
plt.plot(0,0,'+')

plt.show()  
 

#Posiciones.
plt.figure()

plt.plot(t,x)
plt.plot(t,y)

plt.xlabel('tiempo')
plt.ylabel('posiciones')

#Velocidades.
plt.figure()

plt.plot(t,vx)
plt.plot(t,vy)

plt.xlabel('tiempo')
plt.ylabel('velocidades')


#Para la elipse.Pd: se ve mejor así.

plt.figure()

plt.plot(x,y)
plt.plot(0,0,'+')


plt.show()
#Vas cambiando el tiempo de integración y te sacas una animación.
'''
#Ejercicio 6. Vmuelle= K*r^2/2. Porque la fuerza es F=k*r
k= 5

def Vmuelle(r):#Potencial del muelle, hay que graficarlo.
    return k*(r**2)/2

r= np.linspace(-2*R,R*2,2000)
ET= np.ones(2000)*Vmuelle(R)#La energía total como un array coon todo el mismo valor
      
plt.figure()

plt.plot(r,Vmuelle(r))#Potencial efectivo
plt.plot(r,ET)#Energía total
plt.xlabel('r')
plt.ylabel('Vmuelle')

#Las trayectorias con odeint.Despejo kx=ma y saco ax y ay.
def t_elipse(z,t):
    x,y,vx,vy= z
    dzdt= [vx,vy,-k*x/m,-k*y/m]
    return dzdt

x0= R
vx0= 0

y0= 0
vy0= vc

t= np.linspace(0,5,25000)#tiempo de integración.

c0= [x0,y0,vx0,vy0]#lista de las condiciones iniciales
#Hacemos odeint.
sol= odeint(t_elipse,c0,t)

x= sol[:,0]
y= sol[:,1]
vx= sol[:,2]
vy= sol[:,3]

plt.figure(figsize=(4, 4))

plt.plot(x,y)
plt.plot(0,0,'+')

plt.show() 



#Para la elipse.Pd: se ve mejor así.

plt.figure()

plt.plot(x,y)
plt.plot(0,0,'+')
#Parece que solo está la órbita elíptica. Pero Luis decía que había conseguido una circular.

#La órbita circular. Usamos vyinicial= (Kr^2)^(1/2). Usa estas condiciones iniciales.

def t_elipse(z,t):
    x,y,vx,vy= z
    dzdt= [vx,vy,-k*x/m,-k*y/m]
    return dzdt

x0= R
vx0= 0

y0= 0
vy0= R*k**(1/2)

t= np.linspace(0,5,25000)#tiempo de integración.

c0= [x0,y0,vx0,vy0]#lista de las condiciones iniciales
#Hacemos odeint.
sol= odeint(t_elipse,c0,t)

x= sol[:,0]
y= sol[:,1]
vx= sol[:,2]
vy= sol[:,3]

plt.figure(figsize=(4, 4))

plt.plot(x,y)
plt.plot(0,0,'+')

#Ejercicio 7. Le sumamos un término multiplicado por r al cubo.
# r^3 = (x^2 + y^2)^(3/2). Pero además hay que multiplicarle el r versor.

k= 5
a= 2.5#Término que acompaña a r^3

def t_elipse(z,t):
    x,y,vx,vy= z
    dzdt= [vx,vy,-(k*x + a*x*(x**2 + y**2))/m,-(k*y + a*y*(x**2 + y**2))/m]
    return dzdt

x0= 1
vx0= 0

y0= 0
vy0= 1

t= np.linspace(0,100,25000)#tiempo de integración.

c0= [x0,y0,vx0,vy0]#lista de las condiciones iniciales
#Hacemos odeint.
sol= odeint(t_elipse,c0,t)

x= sol[:,0]
y= sol[:,1]
vx= sol[:,2]
vy= sol[:,3]

plt.figure()

plt.plot(x,y)
plt.plot(0,0,'+')
'''