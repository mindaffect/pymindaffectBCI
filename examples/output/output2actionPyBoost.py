#!/usr/bin/python3
from utopia2output import *
from time import sleep
from pyb00st.movehub import MoveHub
from pyb00st.constants import *

mymovehub = MoveHub('00:16:53:A5:03:67', 'Auto', 'hci0')
mymovehub.start()
while True:  
    try:
        mymovehub.connect()
    except Exception as e:
        print(e)
        print('Cannot connect to Boost, trying again..')
    else:
        break

# define our action functions
def head (objID):
    print('move head')
    try:
        mymovehub.run_motor_for_angle(MOTOR_D, 70, 20)
        sleep(2)
        mymovehub.run_motor_for_angle(MOTOR_D, 140, -10)
        sleep(1)
        mymovehub.run_motor_for_angle(MOTOR_D, 70, 20)
    except:
        print('Cannot connect to Boost.')

def left (objID):
    print('left')
    try:
        mymovehub.run_motors_for_angle(MOTOR_AB, 180, -10, 10)
        sleep(3)
        mymovehub.run_motors_for_angle(MOTOR_AB, 180, 10, -10)
        sleep(1)
    except:
        print('Cannot connect to Boost.')
def right (objID):
    print('right')
    try:
        mymovehub.run_motors_for_angle(MOTOR_AB, 180, 10, -10)
        sleep(3)
        mymovehub.run_motors_for_angle(MOTOR_AB, 180, -10, 10)
        sleep(1)
    except:
        print('Cannot connect to Boost.')


# IMPORTANT: Somewhere:
# mymovehub.stop()
#We then connect them to their trigger object identifiers
objectID2Action = { 52 :head, 50 :left, 51 :right}
# create the output object
utopia2output=Utopia2Output(outputPressThreshold=None)
# replace the default action dictionary with ours. N.B. be careful
utopia2output.objectID2Action=objectID2Action
# run the output module
utopia2output.run()
