# ScenarioRunner/Scenarios/SimpleDrive/SimpleDrive.py

import math
import carla
import py_trees
import random
import time
import sys
import traceback
import zmq
import json
import networkx as nx
import sys
import os

# Retrieve the project root from the environment variable
project_root = os.environ.get('PROJECT_ROOT')

if project_root:
    print(f"Project root is: {project_root}")
    # Use the project root to construct paths
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        # Add the src directory to the system path
        print(f"Adding {src_path} to sys.path")
        sys.path.append(src_path)
else:
    print("Error: PROJECT_ROOT environment variable is not set.")

from src.data_interface import DataInterface
from src.filters.filter_interface import FilterInterface
from src.scenarios.scenario_manager import CustomScenarioManager
from src.behaviors.drive_agent import DriveWithAgent
from src.behaviors.npc_controller import NPCActorsController
from src.behaviors.reset_scenario import ResetScenarioNode
from src.behaviors.loop_behavior import LoopBehavior
from src.behaviors.check_arrival import CheckArrival
from src.behaviors.location_timer import ArriveToLocationOnTime

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.custom_agent import CustomAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.timer import TimeOut
# from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
#                                                                       ActorDestroy,
#                                                                       KeepVelocity,
#                                                                       StopVehicle,
#                                                                       WaypointFollower)




from agents.navigation.local_planner import RoadOption

DEBUG_LIFETIME = 300 # seconds
class SimpleDrive(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False):

        # ========== GET THESE VARIABLES FROM CONFIG IN THE FUTURE ========== #
        self.timeout=100000
        random.seed(42)

        self.n_loops = 10 # Expose this at some point
        self.num_parked = 10
        self.num_background = 10
        self.max_scenario_time = 90
        self.get_detection_area = True

        # self.filter_params = {
        #     'threshold': 45.0,
        # }
        self.filter_params = {
            'model_path': 'saved_models/random_forest/traffic_actor_classifier_0.joblib',
            'egocentric' : True
        }
        self.filter_type = 'binary_classifier'  # Use 'data_collector' for collecting data without filtering

        # ========== END OF CONFIG VARIABLES ========== #
                
        print('Debug Mode:', debug_mode)
        self.data_interface = DataInterface(interface_type='sender')  # Initialize the data interface
        
        self.spawn_points = world.get_map().get_spawn_points()
        self.scenario_manager = CustomScenarioManager()
        
        self.global_route_planner = GlobalRoutePlanner(wmap = world.get_map(), 
                                                      sampling_resolution=2.0)
        super().__init__(
            name="SimpleDrive",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=True,
        )

        self._set_spectator_camera()
        self.display_detection_area()
        if debug_mode:
            self.display_spawn_ids()
            self.display_grid()

        topology_json = self.get_topology()
        self.data_interface.send_data(topology_json)


    def get_graph_topology(self):     
        
        '''
        Returns the graph topology of the map as a JSON
        '''
        nodes = {}
        edges = []
        graph = self._agent.get_global_planner()._graph
        for ind in graph.nodes:
            x, y, z = graph.nodes[ind]['vertex']
            nodes[ind] = [x, y, z]  # Store the vertex coordinates
        for edge in graph.edges:
            ind1, ind2 = edge
            edges.append((ind1, ind2))  # Store the edge as a tuple of node indices

        graph_topology = {
            "nodes": nodes,
            "edges": edges
        }
        return graph_topology
    
    def get_topology(self):     
        
        '''
        Returns the graph topology of the map as a JSON
        '''
        # nodes = {}
        paths = []
        graph_topology = self._agent.get_global_planner()._topology
        for edge in graph_topology:
            path = []
            entry_x, entry_y, _ = edge['entryxyz']
            path.append([entry_x, entry_y])
            for waypoint in edge['path']:
                wp_x, wp_y = waypoint.transform.location.x, waypoint.transform.location.y
                path.append([wp_x, wp_y])
            exit_x, exit_y, _ = edge['exitxyz']
            path.append([exit_x, exit_y])
            paths.append(path)
        
        return {'topology' : paths}
    
    def _initialize_actors(self, config):
        #Base for now
        scenario_type = 'base'
        ego_route, npc_params_list, detection_area, _ = self.scenario_manager.generate_scenario(
            get_detection_area = self.get_detection_area,
            scenario_type = scenario_type,
            num_parked = self.num_parked,
            num_background = self.num_background)
        
        self._ego_start_location_dict, self._destination_dict = ego_route 
        self._ego_start_location = self._dict_to_transform(self._ego_start_location_dict)
        self._destination = self._dict_to_transform(self._destination_dict)

        self._detection_area = detection_area

        if not hasattr(self, 'filter_interface'):
            self.filter_params['detection_area'] = self._detection_area
            self.filter_interface = FilterInterface(filter_type=self.filter_type, filter_params=self.filter_params)
        
        #Publish the ego start location and destination to the data interface
        self.data_interface.send_data({
            'ego_start_location': self._ego_start_location_dict,
            'destination': self._destination_dict
        }, type = 'send')
    

        blueprint_library = self.world.get_blueprint_library().filter("vehicle.*")
        
        # ego_spawn_points = scenario._get_all_ego_spawn_points()
        # self.validate_ego_spawn_points(ego_spawn_points)

        self._set_ego_location(self._ego_start_location)

        ego = CarlaDataProvider.get_hero_actor()
        self._vehicle = ego
        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = CustomAgent(ego, behavior="normal")
        else:
            self._agent._vehicle = ego  # Update the agent's vehicle reference
       
        #Clear the waypoints queue and set a new destination
        self._agent.set_destination(self._destination.location, start_location = self._ego_start_location)

        #Spawn NPC Actors
        self._npc_actor_configs = [] #Keep track of actors and specified behaviors
        
        if not hasattr(self, '_npc_actor_controller'):
            self._npc_actor_controller = NPCActorsController(self.global_route_planner, behavior_list = [])

        for npc_params in npc_params_list:
            #Get NPC spawn transform
            transform_dict = npc_params['spawn']
            transform = self._dict_to_transform(transform_dict)
            
            #Spawn NPC
            type_exclude = npc_params.get('type_exclude', [])
            filtered_blueprint_library = [bp for bp in blueprint_library \
                        if bp.has_attribute('base_type') and bp.get_attribute('base_type').as_str().lower() not in type_exclude]
            npc_model = random.choice(filtered_blueprint_library)
            try:
                rolename = npc_params['role']
                controller_type = npc_params['behavior']['controller']
                use_autopilot = True if controller_type == 'autopilot' else False
                npc_actor = CarlaDataProvider.request_new_actor(npc_model.id, transform, 
                                                                rolename = rolename, autopilot = use_autopilot)
                # print('SPAWNED NPC ACTOR:', npc_actor, 'ROLENAME:', npc_actor.attributes['role_name'])
                                                                # random_location = True)
                if controller_type == 'bt' and npc_actor is not None:
                    npc_params['behavior']['route']['start'] = carla.Location(**npc_params['behavior']['route']['start'])
                    npc_params['behavior']['route']['end'] = carla.Location(**npc_params['behavior']['route']['end'])
                    cfg = {
                        'actor' : npc_actor,
                        'behavior': npc_params['behavior']
                    }
                    self._npc_actor_configs.append(cfg)
                    
            except Exception as e:
                continue
            self.other_actors.append(npc_actor)

        #Initialize the NPC controller with the generated behaviors
        if self._npc_actor_configs:
            self._npc_actor_controller.update_behavior_list(self._npc_actor_configs)
    
    def _create_behavior(self):
        # root = py_trees.composites.Parallel(
        #     "DriveLoop",
        #     policy = py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        # )
        criteria = self._create_test_criteria()
        root = LoopBehavior(
            "DriveLoop", 
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, 
            n_loops=self.n_loops,
            criteria = criteria,
            filter_interface = self.filter_interface
        )

        # timer_node = ArriveToLocationOnTime(
        #     self._vehicle,
        #     self._destination.location,
        #     time_limit=90  # seconds
        # )
        episode_sequence = py_trees.composites.Parallel(
            "OneEpisode",
            policy = py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        episode_sequence.add_child(DriveWithAgent(self._agent, self.filter_interface))
        episode_sequence.add_child(CheckArrival(self._vehicle, self._agent.get_destination))
        episode_sequence.add_child(TimeOut(timeout = self.max_scenario_time, name="TimeOut"))
        #Add something here to check collision as well

        episode_sequence.add_child(self._npc_actor_controller)

        reset_node = ResetScenarioNode(self)

        loop = py_trees.composites.Sequence('LoopEpisode')
        loop.add_child(episode_sequence)
        loop.add_child(reset_node)
        
        root.add_child(loop)

        py_trees.display.print_ascii_tree(root)
        return root

    def _create_test_criteria(self):
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    
    def _setup_scenario_trigger(self, config):
        return None
    
    def _dict_to_transform(self, transform_dict):
        '''
        Converts a dictionary with keys 'location' and 'rotation' to a carla.Transform object
        '''
        location_dict = transform_dict['location']
        rotation_dict = transform_dict['rotation']
        location = carla.Location(**location_dict)
        rotation = carla.Rotation(**rotation_dict)
        return carla.Transform(location, rotation)
    
    def _set_ego_location(self, transform):
        '''
        Sets the ego vehicle to a specific spawn point
        '''
        if isinstance(transform, dict):
            transform = self._dict_to_transform(transform)
        elif not isinstance(transform, carla.Transform):
            raise TypeError("Transform must be a dictionary or carla.Transform object")
        
        self.ego_vehicles[0].set_transform(transform)

        # Tick world once to allow physics to update. DOES NOT WORK IN ASYNCHRONOUS MODE.
        # You would likely need to either run on synchronous mode temporarily or use time.sleep
        # instead 
        self.world.tick()
    
    def _set_spectator_camera(self):
        # position spectator overhead
        spectator = self.world.get_spectator()
        x_coord, y_coord, z_coord = -45, 23, 70
        spec_tf = carla.Transform(carla.Location(x_coord,y_coord,z_coord), 
                                  carla.Rotation(pitch = -90, yaw = -90))
        spectator.set_transform(spec_tf)

    def display_spawn_ids(self):
        '''
        Displays the spawn points of the map
        '''
        spawn_points = self.world.get_map().get_spawn_points()
        # spawn_ids = [spawn.id for spawn in spawn_points]
        for i, wp in enumerate(spawn_points):
            # print('WAYPOINT TRANSFORM',wp.transform)
            self.world.debug.draw_point(
                wp.location,
                size=0.1,
                color=carla.Color(255, 0, 0),  # red for occupied waypoints
                life_time=DEBUG_LIFETIME,
                persistent_lines=False
            )
            self.world.debug.draw_string(
                location = wp.location,
                text = str(i),
                draw_shadow=False, 
                color=carla.Color(255,0,0), 
                life_time=DEBUG_LIFETIME, 
                persistent_lines=False
            )
            

    def display_grid(self):
        '''
        Displays grid in viewing window
        '''
        max_coord = 100
        grid_size = 5
        default_z = 0.5

        for x in range(-max_coord, max_coord + 1, grid_size):
            for y in range(-max_coord, max_coord + 1, grid_size):
                location = carla.Location(x = x, y = y, z = default_z)
                self.world.debug.draw_string(location, f'({x},{y})',
                                        life_time = DEBUG_LIFETIME, 
                                        color = carla.Color(0, 255, 0))
        
        arrow_len = 15.0
        yaw_colors = {
            0: carla.Color(255, 0, 0),
            90: carla.Color(0, 255, 0),
            180: carla.Color(0, 0, 255),
            270: carla.Color(160, 32, 240)
        }

        origin = carla.Location(x = 0, y = 0, z = default_z)
        for deg, color in yaw_colors.items():
            rad = math.radians(deg)
            dx = math.cos(rad) * arrow_len
            dy = math.sin(rad) * arrow_len

            target = carla.Location(x = origin.x + dx, 
                                    y = origin.y + dy,
                                    z = default_z)
            self.world.debug.draw_arrow(origin, target, 
                                        thickness = 0.1, arrow_size = 0.3,
                                        color = color, life_time = DEBUG_LIFETIME)
            self.world.debug.draw_string(target, str(deg), 
                                         color = color, life_time = DEBUG_LIFETIME)

    def display_detection_area(self):
        x_top, y_top = self._detection_area['top_left']['x'], self._detection_area['top_left']['y']
        x_bot, y_bot = self._detection_area['bottom_right']['x'], self._detection_area['bottom_right']['y']
        z_default = 0.3
        corners = [
            carla.Location(*(x_top, y_top, z_default)),
            carla.Location(*(x_top, y_bot, z_default)),
            carla.Location(*(x_bot, y_bot, z_default)),
            carla.Location(*(x_bot, y_top, z_default)),
            carla.Location(*(x_top, y_top, z_default))
            # Closing the rectangle by returning to the first corner
        ]

        for i in range(len(corners) - 1):
            self.world.debug.draw_line(
                corners[i],
                corners[i+1],
                thickness=0.5,
                color=carla.Color(0, 255, 0),
                life_time=0
            )


    def validate_ego_spawn_points(self,spawn_points):
        '''
        Validates the spawn points of the map
        '''
        blueprint_library = self.world.get_blueprint_library().filter("vehicle.*")

        for spawn in spawn_points:
            if isinstance(spawn, dict):
                spawn = self._dict_to_transform(spawn)
            elif not isinstance(spawn, carla.Transform):
                raise TypeError("Spawn point must be a dictionary or carla.Transform object")
            
            npc_model = random.choice(blueprint_library)
            npc_actor = CarlaDataProvider.request_new_actor(npc_model.id, spawn, 
                                                        rolename = 'npc', autopilot = False)
            print('Spawning NPC Actor at:', spawn.location)
        

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
  