import math as m
import random as r
from simple_geometry import *
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle, Rectangle
import numpy as np


class Car():
    def __init__(self) -> None:
        self.radius = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def diameter(self):
        return self.radius/2

    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.radius/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / (self.radius*1.5))) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground():
    def __init__(self):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]

        self.car = Car()
        self.reset()

    def _setDefaultLine(self):
        print('use default lines')
        # default lines
        # Destination area: top-left (18, 40) and bottom-right (30, 37)
        self.destination_topleft = Point2D(18, 40)
        self.destination_bottomright = Point2D(30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination area corners
                self.destination_topleft = Point2D(*[float(v) for v in lines[1].split(',')])
                self.destination_bottomright = Point2D(*[float(v) for v in lines[2].split(',')])

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception as e:
            print(f"Error reading path file: {e}")
            self._setDefaultLine()

    def is_in_destination_area(self, point):
        """Check if a point is within the destination area"""
        return (self.destination_topleft.x <= point.x <= self.destination_bottomright.x and
                self.destination_bottomright.y <= point.y <= self.destination_topleft.y)

    def predictAction(self, state):
        '''
        此function為模擬時，給予車子隨機數字讓其走動。
        不需使用此function。
        '''
        return r.randint(0, self.n_actions-1)

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        diameter = self.car.diameter

        # Check if car is in destination area
        isAtDestination = self.is_in_destination_area(cpos)
        done = False if not isAtDestination else True
        self.reached_goal = isAtDestination  # Track if the goal was reached for rendering

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # check every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (
                dToLine < diameter and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True
                    self.reached_goal = False  # If we're done but not at destination, we hit a wall

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.reached_goal = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None):
        '''
        請更改此處code，依照自己的需求撰寫。
        '''
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            self.car.tick()

            self._checkDoneIntersects()
            return self.state
        else:
            return self.state

    def render(self, ax):
        """
        Render the current state of the simulation on the provided axes
        """
        ax.clear()
        
        # Draw the walls/boundaries
        for line in self.lines:
            ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'k-', linewidth=2)
        
        # Draw decorative lines
        for line in self.decorate_lines:
            ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'b--', linewidth=1)
        
        # Draw destination area as a rectangle
        dest_width = self.destination_bottomright.x - self.destination_topleft.x
        dest_height = self.destination_topleft.y - self.destination_bottomright.y
        dest_rect = Rectangle((self.destination_topleft.x, self.destination_bottomright.y), 
                             dest_width, dest_height, fill=True, color='g', alpha=0.3)
        ax.add_patch(dest_rect)
        
        # Draw the destination area outline
        dest_points = [
            (self.destination_topleft.x, self.destination_topleft.y),
            (self.destination_bottomright.x, self.destination_topleft.y),
            (self.destination_bottomright.x, self.destination_bottomright.y),
            (self.destination_topleft.x, self.destination_bottomright.y),
            (self.destination_topleft.x, self.destination_topleft.y)
        ]
        dest_xs, dest_ys = zip(*dest_points)
        ax.plot(dest_xs, dest_ys, 'g-', linewidth=2)
        
        # Draw car
        car_pos = self.car.getPosition()
        car_circle = Circle((car_pos.x, car_pos.y), self.car.diameter, fill=False, color='r')
        ax.add_patch(car_circle)
        
        # Draw car direction
        front_pos = self.car.getPosition('front')
        ax.plot([car_pos.x, front_pos.x], [car_pos.y, front_pos.y], 'r-', linewidth=2)
        
        # Draw wheel direction
        wheel_pos = self.car.getWheelPosPoint()
        ax.plot([car_pos.x, wheel_pos.x], [car_pos.y, wheel_pos.y], 'y-', linewidth=1)
        
        # Draw sensors
        if self.front_intersects:
            front_int = self.front_intersects[0]
            ax.plot([front_pos.x, front_int.x], [front_pos.y, front_int.y], 'b-', alpha=0.5)
        
        right_pos = self.car.getPosition('right')
        if self.right_intersects:
            right_int = self.right_intersects[0]
            ax.plot([right_pos.x, right_int.x], [right_pos.y, right_int.y], 'g-', alpha=0.5)
        
        left_pos = self.car.getPosition('left')
        if self.left_intersects:
            left_int = self.left_intersects[0]
            ax.plot([left_pos.x, left_int.x], [left_pos.y, left_int.y], 'y-', alpha=0.5)
        
        # Set axis limits to focus on relevant area
        x_min, x_max = -10, 55
        y_min, y_max = -10, 55
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add labels
        ax.set_title('Self-Driving Car Simulation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add state information
        state_text = f"Distances - Front: {self.state[0]:.2f}, Right: {self.state[1]:.2f}, Left: {self.state[2]:.2f}"
        ax.text(0.05, 0.95, state_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Add car position information
        pos_text = f"Position: ({car_pos.x:.2f}, {car_pos.y:.2f}), Angle: {self.car.angle:.2f}°"
        ax.text(0.05, 0.90, pos_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Add done status
        if self.done:
            if self.reached_goal:
                status = "SUCCESS: Reached Destination!"
                color = 'green'
            else:
                status = "FAILED: Collision!"
                color = 'red'
                
            ax.text(0.5, 0.5, status, transform=ax.transAxes, fontsize=20,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))


def run_example():
    # Setup the simulation
    p = Playground()
    
    # Create figure and axes for visualization
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Run simulation
    state = p.reset()
    step_count = 0
    max_steps = 200  # Prevent infinite loops
    
    while not p.done and step_count < max_steps:
        # Print state and position
        car_pos = p.car.getPosition('center')
        print(f"Step {step_count} - State: {state}, Position: ({car_pos.x:.2f}, {car_pos.y:.2f})")
        
        # Render the current state
        p.render(ax)
        plt.draw()
        plt.pause(0.1)  # Small pause to make animation visible
        
        # Select action randomly or use your own policy here
        action = p.predictAction(state)
        
        # Take action
        state = p.step(action)
        step_count += 1
    
    # Final render after completion
    p.render(ax)
    plt.draw()
    
    # Add a bit more detail to the final status
    if p.done:
        if step_count >= max_steps:
            print(f"Simulation ended after maximum steps ({max_steps})")
        elif p.reached_goal:
            print(f"SUCCESS: Car reached destination area at step {step_count}")
            print(f"Final position: ({p.car.getPosition().x:.2f}, {p.car.getPosition().y:.2f})")
        else:
            print(f"FAILED: Car collision at step {step_count}")
            print(f"Final position: ({p.car.getPosition().x:.2f}, {p.car.getPosition().y:.2f})")
    
    # Keep the plot open until closed manually
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_example()