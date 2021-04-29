# rcj_soccer_player controller - ROBOT B1

# Feel free to import built-in libraries
import math
import time
from time import sleep
# You can also import scripts that you put into the folder with controller
from rcj_soccer_robot import RCJSoccerRobot, TIME_STEP
import utils


class MyRobot1(RCJSoccerRobot):
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            if self.is_new_data():
                #get desired actions
                _, actions = data = self.get_new_data()
                while self.is_new_data():
                    _, actions = self.get_new_data()
                

                #since this is robot 1 actions are in indexes 0, 1
                #multiply by 10 to obtain values from -10 to 10
                left_speed = actions[0] * 10
                right_speed = actions[1] * 10


                # Set the speed to motors
                self.left_motor.setVelocity(left_speed)
                self.right_motor.setVelocity(right_speed)
