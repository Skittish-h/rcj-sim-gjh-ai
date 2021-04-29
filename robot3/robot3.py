# rcj_soccer_player controller - ROBOT B2

###### REQUIRED in order to import files from B1 controller
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
# You can now import scripts that you put into the folder with your
# robot B1 controller class MyRobot(rcj_soccer_robot.RCJSoccerRobot):
from robot1 import utils
from controller import Robot

from robot1.rcj_soccer_robot import RCJSoccerRobot, TIME_STEP
from robot1.intercepts import interceptCalculator
######


# Feel free to import built-in libraries
# rcj_soccer_player controller - ROBOT B1

# Feel free to import built-in libraries
import math

# You can also import scripts that you put into the folder with controller

from robot1.CoordinateRecalculator import coor_recalc, robot_pos_recalc
from robot1.GoToFunc import goTo
from robot1.Goalie import goalie_angles, goalie_cal_Y, correct_rotation
from robot1.SupportOptiPos import support_position
from robot1.MovementCalculator import fit_parabola, get_tangent_point, passes_boundary

######

INTERCEPT_CONST = 0.03

# Feel free to import built-in libraries
import math
#robot class
class MyRobot(RCJSoccerRobot):
    
    #function that uses instacne of intercept calculator to find our robots intercepts and the intercepts for all
    def getIntercepts(self, data, Team):
        #get our robots
        r1 = robot_pos_recalc(data[f"{self.team}1"], Team=Team)
        r2 = robot_pos_recalc(data[f"{self.team}2"], Team=Team)
        r3 = robot_pos_recalc(data[f"{self.team}3"], Team=Team)

        #prep dictionary
        intercepts = {"r1":0,"r2":0 ,"r3":0}
        
        #get all the intercepts, with a lota samples
        intercepts["r1"] = self.intercept_c.calculateOptimumIntercept(r1, Team, sample_count=200)
        intercepts["r2"] = self.intercept_c.calculateOptimumIntercept(r2, Team, sample_count=200)
        intercepts["r3"] = self.intercept_c.calculateOptimumIntercept(r3, Team, sample_count=200)

        #dict with times
        ts = {"r1":intercepts["r1"]["t"],"r2":intercepts["r2"]["t"],"r3":intercepts["r3"]["t"]}

        #dict with our robots intercepts
        my_intercept = intercepts[f"r{self.player_id}"]

        return my_intercept, ts
    
    def role_decision(self, intercept_times, data):
        my_list = list(sorted(intercept_times.items(), key = lambda a:a[1]))
        my_list = [x[0] for x in my_list]
        if self.player_id == 1:
            if my_list[0] == "r1":
                return "att"
            else:
                my_list.pop(0)
                if (data[f'{self.team}1']['x'] > data[f'{self.team}{"2" if "r2" in my_list else "3"}']['x']) if self.team=="B" else (data[f'{self.team}1']['x'] < data[f'{self.team}{"2" if "r2" in my_list else "3"}']['x']):
                    return "goal"
                else:
                    return "back" 

        elif self.player_id == 2:
            if my_list[0] == "r2":
                return "att"
            else:
                my_list.pop(0)
        
                if (data[f'{self.team}2']['x'] > data[f'{self.team}{"1" if "r1" in my_list else "3"}']['x']) if self.team=="B" else (data[f'{self.team}2']['x'] < data[f'{self.team}{"1" if "r1" in my_list else "3"}']['x']):
                    return "goal"
                else:
                    return "back" 

        elif self.player_id == 3:
            if my_list[0] == "r3":
                return "att"
            else:
                my_list.pop(0)
        
                if (data[f'{self.team}3']['x'] > data[f'{self.team}{"1" if "r1" in my_list else "2"}']['x']) if self.team=="B" else (data[f'{self.team}3']['x'] < data[f'{self.team}{"1" if "r1" in my_list else "2"}']['x']):
                    return "goal"
                else:
                    return "back" 

    def be_attacker(self, myi, robot_pos, team):
        stuff = 0
        point = 0

        if (myi['x'] < robot_pos['x']) if team else (myi['x'] > robot_pos['x']):
            
            #print(intercepts)
            x = fit_parabola(myi, robot_pos ,{'x':(0.0 if team else 1.0),"y":0.5})
            if not passes_boundary(x):
                point = get_tangent_point(robot_pos, x, team)
                ball_angle, robot_angle = self.get_angles(point, robot_pos)
                
                return goTo(point['x'], point['y'], robot_pos, robot_angle)
                
            else:
                ball_angle, robot_angle = self.get_angles(myi, robot_pos)
                
                return goTo(myi['x'], myi['y'], robot_pos, robot_angle)
                
        else:
            ball_angle, robot_angle = self.get_angles(myi, robot_pos)
            
            return goTo(myi['x'], myi['y'], robot_pos, robot_angle)
        


    def be_goalie(self, ball_pos, robot_pos, team):
        Designated_pos = [[0.58, 0]] if team else [[-0.58,0]]
        DesiredPos = coor_recalc(Designated_pos[0][0],Designated_pos[0][1], team=team)
        if (ball_pos['x'] > DesiredPos['x']) if team else (ball_pos['x'] < DesiredPos['x']):
            DesiredPos = ball_pos
        else:
            DesiredPos['y'] = goalie_cal_Y(ball_pos)
        
        ball_angle, robot_angle = self.get_angles(ball_pos, robot_pos)
        return goTo(DesiredPos["x"], DesiredPos["y"], robot_pos, robot_angle) #0 right motor, 1 left motor 

    def be_backup(self, robot_pos, data, Team):
        DesiredPos = support_position(data, Team)

        ball_angle, robot_angle = self.get_angles(DesiredPos, robot_pos)
        return goTo(DesiredPos["x"], DesiredPos["y"], robot_pos, robot_angle) #0 right motor, 1 left motor 
    


    def run(self):
        #create interceptcalc instance
        self.intercept_c = interceptCalculator(3)

        Team = (self.team == "B")
        while self.robot.step(TIME_STEP) != -1:
            if self.is_new_data():
                data = self.get_new_data()
                while self.is_new_data():
                    data = self.get_new_data()
                
                robot_pos = robot_pos_recalc(data[self.name], Team)
                # Get & recalculate the position of the ball
                ball_pos = coor_recalc(data['ball']['x'], data['ball']['y'], Team)
                self.intercept_c.pushPoint(ball_pos)
                
                myi, intercepts = self.getIntercepts(data, Team)
                
                
                role = self.role_decision(intercepts, data)
                out=[]
               
                # if roles att is 1 the B1 will execute attacker code
                if role == "att":                
                    out = self.be_attacker(myi, robot_pos, Team)

                # if goalie will be 1 B1 will execute goalie code
                elif role == "goal":
                    out = self.be_goalie(ball_pos, robot_pos, Team)

                #if support is 1 B1 will execute backup code
                elif role == "back":
                    out = self.be_backup(robot_pos, data, Team)
                    pass
                self.left_motor.setVelocity(out[1])
                self.right_motor.setVelocity(out[0])
               
robot = Robot()          
             
my_robot = MyRobot(robot)
my_robot.run()