from collections import deque
import random

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints, get_speed
from .local_planner import LocalPlanner


class CustomPlanner(LocalPlanner):
    def __init__(self, vehicle, opt_dict={}, map_inst=None):
        super().__init__(vehicle, opt_dict, map_inst)
        print('CustomPlanner initialized')

    
    def display_alternates(self):
        for waypoint, _ in self._waypoints_queue:
            for waypoint_alternate in waypoint.next(1.0):
                if waypoint_alternate.is_junction:
                    continue
                self._world.debug.draw_point(
                    waypoint_alternate.transform.location,
                    size=0.1,
                    color=carla.Color(255, 0, 0),  # green
                    life_time=0.1,
                    persistent_lines=False
                )
