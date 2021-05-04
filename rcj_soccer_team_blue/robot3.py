# rcj_soccer_player controller - ROBOT B3

# Feel free to import built-in libraries
import math

# You can also import scripts that you put into the folder with controller
from rcj_soccer_robot import RCJSoccerRobot, TIME_STEP
import utils


class MyRobot3(RCJSoccerRobot):
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            if self.is_new_data():
                #get desired actions
                _, actions = data = self.get_new_data()
                while self.is_new_data():
                    _, actions = self.get_new_data()

                #since this is robot 3 actions are in indexes 4, 5
                #multiply by 10 to obtain values from -10 to 10
                left_speed = actions[4] * 10
                right_speed = actions[5] * 10

                # Set the speed to motors
                self.left_motor.setVelocity(left_speed)
                self.right_motor.setVelocity(right_speed)
