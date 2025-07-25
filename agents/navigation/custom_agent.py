import carla
import networkx as nx
import numpy as np

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal
from agents.tools.misc import get_speed
from .custom_planner import CustomPlanner



class CustomAgent(BehaviorAgent):
    def __init__(self, vehicle, opt_dict = {}, **kwargs):
        super().__init__(vehicle, **kwargs)

        print('CustomAgent initialized with options:', opt_dict)
        self._local_planner = CustomPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        self._visible_vehicle_list = None
        self._occupied_edges = []

        #Expose this at some point, maybe to opt_dict
        self.max_wp_distance_to_npc = 3.0

        
    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        #print('PASSED START LOCATION:', start_location, type(start_location))
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
        else:
            start_location = self._vehicle.get_location()

        #print('ACTUAL START LOCATION:', start_location, type(start_location))

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=True)
        self._destination = end_location

    def get_destination(self):
        return self._destination

    def run_step(self, new_visible_vehicles = None, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information(new_visible_vehicles)

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.2: Car following behaviors

        #if debug
        self.draw_vehicle_filter_debug(self._visible_vehicle_list)
        self.draw_behavior_agent_route_debug()
        self.display_map_graph_debug()


        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp, self._visible_vehicle_list)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            # print('DISTANCE TO VEHICLE:', distance, 'BRAKING DISTANCE:', self._behavior.braking_distance)
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint is not None and self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control
    
    def trace_route(self, start_waypoint, end_waypoint, weighted = False):
        """
        This method returns list of (carla.Waypoint, RoadOption)
        from origin to destination
        """
        origin = start_waypoint.transform.location
        destination = end_waypoint.transform.location

        # print('TRACING ROUTE FROM {} TO {}'.format(
        #     origin, destination))
        
        route_trace = []
        route = self._path_search(origin, destination, weighted = weighted)
        # print('ROUTE TRACE: {}'.format(route), type(route[0]))
        current_waypoint = self._global_planner._wmap.get_waypoint(origin)
        destination_waypoint = self._global_planner._wmap.get_waypoint(destination)

        for i in range(len(route) - 1):
            road_option = self._global_planner._turn_decision(i, route)
            edge = self._global_planner._graph.edges[route[i], route[i+1]]
            path = []

            if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
                route_trace.append((current_waypoint, road_option))
                exit_wp = edge['exit_waypoint']
                n1, n2 = self._global_planner._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self._global_planner._graph.edges[n1, n2]
                if next_edge['path']:
                    closest_index = self._global_planner._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path'])-1, closest_index+5)
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                route_trace.append((current_waypoint, road_option))

            else:
                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                closest_index = self._global_planner._find_closest_in_list(current_waypoint, path)
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*self._global_planner._sampling_resolution:
                        break
                    elif len(route)-i <= 2 and current_waypoint.road_id == destination_waypoint.road_id \
                        and current_waypoint.section_id == destination_waypoint.section_id \
                            and current_waypoint.lane_id == destination_waypoint.lane_id:
                        destination_index = self._global_planner._find_closest_in_list(destination_waypoint, path)
                        if closest_index > destination_index:
                            break

        return route_trace

    def draw_vehicle_filter_debug(self, all_vehicle_list):
        def draw_bbox(vehicle, color):
            bb = vehicle.bounding_box
            bb.location = vehicle.get_transform().transform(bb.location)  # move to world space
            rotation = vehicle.get_transform().rotation

            self._world.debug.draw_box(
                bb,
                rotation,
                thickness=0.2,
                color=color,
                life_time=0.1,
                persistent_lines=False
            )
        
        draw_bbox(self._vehicle, carla.Color(255,0,0))
        # print('ALL VEHICLE LIST:', len(all_vehicle_list), 'VISIBLE VEHICLE LIST:', len(visible_vehicle_list))
        visible_ids = {vehicle.id for vehicle in self._visible_vehicle_list}
        for v in all_vehicle_list:
            # Transform to world space
            # bb_transform = carla.Transform(v.get_transform().transform(bb_loc), bb_rot)
            if v.id != self._vehicle.id:
                color = carla.Color(0, 0, 255) if v.id in visible_ids else carla.Color(255, 255, 255)
                draw_bbox(v, color)        

    def display_map_graph_debug(self):
        graph = self.get_global_planner()._graph
        for ind in graph.nodes:
            
            x, y, z = graph.nodes[ind]['vertex']
            self._world.debug.draw_point(
                carla.Location(x,y,z),
                size=0.2,
                color=carla.Color(1, 66, 1),  # dark green
                life_time=0.1,
                persistent_lines=False
            )

        occupied_edges = self._find_occupied_edges()
        for edge in graph.edges:
            ind1, ind2 = edge
            x1, y1, z1 = graph.nodes[ind1]['vertex']
            x2, y2, z2 = graph.nodes[ind2]['vertex']
            if edge in occupied_edges:
                color = carla.Color(255, 0, 0)
            else:
                color = carla.Color(1, 66, 1)
            self._world.debug.draw_line(
                carla.Location(x1, y1, z1),
                carla.Location(x2, y2, z2),
                color=color,
                life_time=0.1,
                persistent_lines=False
            )
    
    def draw_behavior_agent_route_debug(self):
        if not hasattr(self, "_local_planner"):
            print("Agent missing local planner, cannot draw route.")
            return

        life_time = 0.1

        # Draw local buffer in green
        plan = self._local_planner.get_plan()
        occupied_waypoints = self._find_occupied_waypoints()
        for wp, _ in plan:
            if wp in occupied_waypoints:
                self._world.debug.draw_point(
                    wp.transform.location,
                    size=0.1,
                    color=carla.Color(255, 0, 0),  # red for occupied waypoints
                    life_time=life_time,
                    persistent_lines=False
                )
            else:
                self._world.debug.draw_point(
                    wp.transform.location,
                    size=life_time,
                    color=carla.Color(0, 255, 0),  # green
                    life_time=life_time,
                    persistent_lines=False
                )
        
        if len(plan) > 0:
            self._world.debug.draw_point(
                #self._destination.location,
                plan[0][0].transform.location,
                size=0.5,
                color=carla.Color(r=255, g=0, b=0),
                life_time=life_time,
                persistent_lines=False
            )

            self._world.debug.draw_point(
                #self._destination.location,
                plan[-1][0].transform.location,
                size=0.5,
                color=carla.Color(r=0, g=0, b=255),
                life_time=life_time,
                persistent_lines=False
            )
        # self._local_planner.display_alternates() 
  


    def collision_and_car_avoid_manager(self, waypoint, vehicle_list):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance
    
    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance
    
    def _update_information(self, new_visible_vehicles = None):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        
        if new_visible_vehicles is None:
            all_vehicle_list = self._world.get_actors().filter("*vehicle*")
            self._visible_vehicle_list = list(all_vehicle_list)#[
            #    v for v in all_vehicle_list 
            #    if v.get_location().distance(self._vehicle.get_location()) < 45 and v.id != self._vehicle.id
            #]
        else:
            self._visible_vehicle_list = new_visible_vehicles

        # Waypoint occupancy check
        self._occupied_edges = self._find_occupied_edges()
        if len(self._occupied_edges) > 0:
            start_location  = self._vehicle.get_location()
            start_waypoint = self._map.get_waypoint(start_location)
            end_waypoint = self._map.get_waypoint(self._destination)

            route_trace = self.trace_route(start_waypoint, end_waypoint, weighted = True)
            if len(route_trace) > 0:
                self._local_planner.set_global_plan(route_trace, clean_queue=True)

        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW
    
    def _find_occupied_waypoints(self):
        """
        Find waypoints that are occupied by vehicles in the visible vehicle list.
        
        :param visible_vehicle_list: List of vehicles that are currently visible to the agent.
        :return: List of waypoints that are occupied by the vehicles in the visible vehicle list.
        """
        occupied_waypoints = []
        for waypoint in self._local_planner.get_plan():
            for vehicle in self._visible_vehicle_list:
                if waypoint[0].transform.location.distance(vehicle.get_location()) < self.max_wp_distance_to_npc:
                    occupied_waypoints.append(waypoint[0])
        return occupied_waypoints
    
    def _find_occupied_edges(self):
        """
        Find edges that are occupied by vehicles in the visible vehicle list.
        
        :param visible_vehicle_list: List of vehicles that are currently visible to the agent.
        :return: List of edges that are occupied by the vehicles in the visible vehicle list.
        """
        occupied_waypoints = self._find_occupied_waypoints()
        occupied_edges = [self._global_planner._localize(waypoint.transform.location) for waypoint in occupied_waypoints]
        return occupied_edges
    
    def _path_search(self, origin, destination, weighted = False):#, weight = 'length'):
        """
        This function finds the shortest path connecting origin and destination
        using A* search with distance heuristic.
        origin      :   carla.Location object of start position
        destination :   carla.Location object of of end position
        return      :   path as list of node ids (as int) of the graph self._graph
        connecting origin and destination
        """
        def occupied_weight(u, v, d):
            # print('U:', u, 'V:', v, 'D:', d)
            return np.inf if (u,v) in self._occupied_edges or (v,u) in self._occupied_edges else d['length']
        if weighted:
            weight = occupied_weight
        else:
            weight = 'length'

        start, end = self._global_planner._localize(origin), self._global_planner._localize(destination)

        route = nx.astar_path(
            self._global_planner._graph, source=start[0], target=end[0],
            heuristic=self._global_planner._distance_heuristic, weight=weight)
        route.append(end[1])
        return route
    