# rcj_soccer_player controller - ROBOT B2

# Feel free to import built-in libraries
import math

# You can also import scripts that you put into the folder with controller
from rcj_soccer_robot import RCJSoccerRobot, TIME_STEP
import utils


class MyRobot2(RCJSoccerRobot):
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            if self.is_new_data():
                #get desired actions
                _, actions = self.get_new_data()

                #since this is robot 2 actions are in indexes 2,3
                #also we multiply by 10 to obtain values from -10 to 10
                left_speed = actions[0] * 10
                right_speed = actions[1] * 10


                # Set the speed to motors
                self.left_motor.setVelocity(left_speed)
                self.right_motor.setVelocity(right_speed)
