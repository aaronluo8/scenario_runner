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
    print(f"Adding {src_path} to sys.path")
    sys.path.append(src_path)
else:
    print("Error: PROJECT_ROOT environment variable is not set.")

from src.data_interface import DataInterface
from src.scenarios import CustomScenarioManager
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.custom_agent import CustomAgent
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTimeToArrivalToLocation
from srunner.scenariomanager.timer import TimeOut
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity,
                                                                      StopVehicle,
                                                                      WaypointFollower)


from agents.navigation.local_planner import RoadOption

class SimpleDrive(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False):
        self.timeout=100000
        random.seed(42)
                
        print('Debug Mode:', debug_mode)
        self.data_interface = DataInterface(interface_type='sender')  # Initialize the data interface
        
        self.spawn_points = world.get_map().get_spawn_points()
        self.scenario_manager = CustomScenarioManager(self.spawn_points)
        
        super().__init__(
            name="SimpleDrive",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=True,
        )

        # self._agent: BehaviorAgent = None
        # self._destination = None

        #Move ego vehicle to a random spawn point

        #Set random destination for ego vehicle
        # self.display_map_graph()
        self._set_spectator_camera()
        if debug_mode:
            self.display_spawn_ids()
            self.display_grid()

        # for ind in range(5):
        #     topology_edge = self._agent.get_global_planner()._topology[ind]
        #     print('TOPOLOGY {ind}')
            
        #     print(f'ENTRY IDS -> ID: {topology_edge["entry"].id}, ROAD ID: {topology_edge["entry"].road_id}, LANE ID: {topology_edge["entry"].lane_id}')
        #     print(f'EXIT IDS -> ID: {topology_edge["exit"].id}, ROAD ID: {topology_edge["exit"].road_id}, LANE ID: {topology_edge["exit"].lane_id}')
        #     for waypoint in topology_edge['path']:
        #         print(f'WAYPOINT: {waypoint.id}, ROAD ID: {waypoint.road_id}, LANE ID: {waypoint.lane_id}')

        # print(f'ID MAP {self._agent.get_global_planner()._id_map}')
        # print(f'ROAD ID TO EDGE MAP {self._agent.get_global_planner()._road_id_to_edge}') 

        # graph_json = self.get_graph_topology()
        topology_json = self.get_topology()
        # self.data_interface.send_data(graph_json)  # Send the graph topology to the receiver
        self.data_interface.send_data(topology_json)

        #graph_json = self.get_graph_topology()
        #self.send_data(graph_json)  # Send the graph topology to the receiver
        # print('TOPOLOGY:',self._agent._global_planner._topology[0])
        # print('WMAP:', self._agent._global_planner._wmap)
        


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
        #Random for now
        self._ego_start_location, self._destination = self.scenario_manager.select_route()
        self._set_ego_location(self._ego_start_location)
        # self._ego_start_location = self._ego_random_spawn()

        ego = CarlaDataProvider.get_hero_actor()
        self._vehicle = ego
        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = CustomAgent(ego, behavior="normal")
        else:
            self._agent._vehicle = ego  # Update the agent's vehicle reference

        print('START LOCATION:', self._ego_start_location, \
              'CURRENT EGO LOCATION:', self._agent._vehicle.get_location())
        
        # self._destination = self._set_random_ego_destination()

        #Clear the waypoints queue and set a new destination
        self._agent.set_destination(self._destination.location, start_location = self._ego_start_location)


        #Spawn NPC Actors
        actor_npc_transforms, background_npc_transforms = self.scenario_manager.generate_npc_behavior()
        for transform_dict in actor_npc_transforms:
            print('ACTOR NPC DICT',actor_npc_transforms)

            location_dict = transform_dict['location']
            rotation_dict = transform_dict['rotation']
            location = carla.Location(x = location_dict['x'],
                                      y = location_dict['y'],
                                      z = 0.3)
            rotation = carla.Rotation(pitch = 0,
                                      yaw = rotation_dict['yaw'],
                                      roll = 0)
            transform = carla.Transform(location, rotation)
            npc_model = random.choice(self.world.get_blueprint_library().filter("vehicle.*"))
            try:
                npc_actor = CarlaDataProvider.request_new_actor(npc_model.id, transform, 
                                                                rolename = "actor", autopilot = False)
                                                                # random_location = True)
                print('MODEL ID:',npc_model.id)
            except Exception as e:
                continue
            self.other_actors.append(npc_actor)

        for transform in background_npc_transforms:
            # transform = random.choice(spawn_points,)
            npc_model = random.choice(self.world.get_blueprint_library().filter("vehicle.*"))
            try:
                npc_actor = CarlaDataProvider.request_new_actor(npc_model.id, transform, 
                                                                rolename = "npc", autopilot = True)
                                                                # random_location = True)
                print('MODEL ID:',npc_model.id)
            except Exception as e:
                continue
            self.other_actors.append(npc_actor)
            # npc = self.world.try_spawn_actor(bp, transform)
            # if npc:
            #     npc.set_autopilot(True)
            #     self.  
        
    def _create_behavior(self):
        # root = py_trees.composites.Parallel(
        #     "DriveLoop",
        #     policy = py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        # )
        criteria = self._create_test_criteria()
        root = LoopBehavior(
            "DriveLoop", 
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, 
            n_loops=10,
            criteria = criteria,
            data_interface = self.data_interface
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
        episode_sequence.add_child(DriveWithAgent(self._agent, self.data_interface))
        episode_sequence.add_child(CheckArrival(self._vehicle, self._agent.get_destination))
        episode_sequence.add_child(TimeOut(timeout = 180, name="TimeOut"))

        reset_node = ResetScenarioNode(self)

        loop = py_trees.composites.Sequence('LoopEpisode')
        loop.add_child(episode_sequence)
        loop.add_child(reset_node)
        
        root.add_child(loop)
        #root.add_child(DriveWithAgent(self._agent))
        # for actor in self.other_actors:
        #     root.add_child(ActorDestroy(actor))
        py_trees.display.print_ascii_tree(root)
        return root


    def _create_test_criteria(self):
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        # ego = self.ego_vehicles[0]
        # destination = self._destination.location  # should be a carla.Location

        # time_limit = 90  # seconds
        # criteria.append(
        #     InTimeToArrivalToLocation(ego, time_limit, destination)
        # )

        return criteria

    
    def _setup_scenario_trigger(self, config):
        return None
    
    def _set_ego_location(self, transform):
        '''
        Sets the ego vehicle to a specific spawn point
        '''
        self.ego_vehicles[0].set_transform(transform)

        # Tick world once to allow physics to update. DOES NOT WORK IN ASYNCHRONOUS MODE.
        # You would likely need to either run on synchronous mode temporarily or use time.sleep
        # instead 
        self.world.tick()
    
    # def _ego_random_spawn(self):
    #     '''
    #     Relocates the ego vehicle to a new position randomly chosen from a set of spawn points.
    #     Assumes only one ego vehicle
    #     '''
    #     ego_spawn_inds = [80,81,91,94,0,1,137,79,50,49,52,51,139,138,110,89,102,99]
        
    #     spawn_points = self.world.get_map().get_spawn_points()
    #     new_spawn = spawn_points[random.choice(ego_spawn_inds)]

    #     # self.ego_vehicles = []
    #     self.ego_vehicles[0].set_transform(new_spawn)

    #     # Tick world once to allow physics to update. DOES NOT WORK IN ASYNCHRONOUS MODE.
    #     # You would likely need to either run on synchronous mode temporarily or use time.sleep
    #     # instead 
    #     self.world.tick()

    #     return new_spawn
    
    # def _set_random_ego_destination(self):
    #     ego_dest_inds = [2,3,77,96,111,115,67,95,27,26,68,122,53,55,56,57]
    #     spawn_points = self.world.get_map().get_spawn_points()
    #     destination = spawn_points[random.choice(ego_dest_inds)]

    #     return destination

    def _set_spectator_camera(self):
        # position spectator overhead
        spectator = self.world.get_spectator()
        x_coord, y_coord, z_coord = -45, 23, 70
        spec_tf = carla.Transform(carla.Location(x_coord,y_coord,z_coord), 
                                  carla.Rotation(pitch = -90, yaw = 0))
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
                life_time=300,
                persistent_lines=False
            )
            self.world.debug.draw_string(
                location = wp.location,
                text = str(i),
                draw_shadow=False, 
                color=carla.Color(255,0,0), 
                life_time=300, 
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
                                        life_time = 30.0, 
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
                                        color = color, life_time = 30.0)
            self.world.debug.draw_string(target, str(deg), 
                                         color = color, life_time = 30.0)




    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class DriveWithAgent(py_trees.behaviour.Behaviour):
    '''
    Hook for the custom control agent
    '''
    def __init__(self, agent: CustomAgent, data_interface: DataInterface):
        super().__init__("DriveWithAgent")
        self.agent = agent
        self.vehicle = agent._vehicle
        self.world = self.vehicle.get_world()
        self.data_interface = data_interface  # Data interface for sending data

    def _prepare_vehicle_information(self):
        '''
        Prepares the vehicle information to be sent to the data interface
        '''
        ego_location = self.vehicle.get_transform().location
        ego_x, ego_y, ego_z = ego_location.x, ego_location.y, ego_location.z

        actor_list = []
        for actor in self.world.get_actors().filter("*vehicle*"):
            if actor.id != self.vehicle.id:
                actor_transform = actor.get_transform()
                actor_location = actor_transform.location
                actor_x, actor_y, actor_z = actor_location.x, actor_location.y, actor_location.z
                actor_info = {
                    'id': actor.id,
                    'location': [actor_x, actor_y, actor_z],
                    'rotation': [actor_transform.rotation.pitch,
                                 actor_transform.rotation.yaw,
                                 actor_transform.rotation.roll],
                    # 'velocity': actor.get_velocity(),
                    # 'acceleration': actor.get_acceleration(),
                    # 'angular_velocity': actor.get_angular_velocity(),
                }
                actor_list.append(actor_info)
        ego_info = {
            'location': [ego_x, ego_y, ego_z],
            'rotation': [
                self.vehicle.get_transform().rotation.roll,
                self.vehicle.get_transform().rotation.pitch,
                self.vehicle.get_transform().rotation.yaw
                ],
            # 'velocity': self.vehicle.get_velocity(),
            # 'acceleration': self.vehicle.get_acceleration(),
            # 'angular_velocity': self.vehicle.get_angular_velocity(),
        }

        vehicle_info = {
            'ego': ego_info,
            'actors': actor_list
        }
        return vehicle_info

    def update(self):
        try:
            # Update ego and npc vehicle locations
            # output = self._prepare_vehicle_information()
            # # Send the vehicle information to the data interface
            # curated_vehicle_list = self.data_interface.send_data(output, type = 'request')
            control = self.agent.run_step()
            self.vehicle.apply_control(control)
            return py_trees.common.Status.RUNNING
        except Exception as e:
            print(f"Error in DriveWithAgent: {e}")
            traceback.print_exc(0)
            raise
    
class ResetScenarioNode(py_trees.behaviour.Behaviour):
    '''
    Node to reset the scenario
    '''
    def __init__(self, scenario):
        super().__init__("ResetScenario")
        self.scenario = scenario

    def update(self):
        print("Resetting scenario...")
        ego = self.scenario.ego_vehicles[0]

        # # Pick new random destination
        # self.scenario._start_location = self.scenario._ego_random_spawn()
        # self.scenario._destination = self.scenario._set_random_ego_destination()

        # # Move vehicle and reset planner
        # ego.set_transform(self.scenario._start_location)
        ego.set_target_velocity(carla.Vector3D(0, 0, 0))  # Stop the vehicle
        ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))  # Stop any rotation
        # ego.apply_control(carla.VehicleControl()) 

        # self.scenario._agent.set_destination(self.scenario._destination.location)
        
        self.scenario.remove_all_actors()
        self.scenario._initialize_actors(self.scenario.config)
        return py_trees.common.Status.SUCCESS
    
class CheckArrival(py_trees.behaviour.Behaviour):
    '''
    Checks if the goal has been reached
    '''
    def __init__(self, vehicle, get_destination, threshold=5.0):
        super().__init__("CheckArrival")
        self.vehicle = vehicle
        self.get_destination = get_destination #Function to retrieve the current destination
        self.threshold = threshold

    def update(self):
        destination = self.get_destination()
        dist = self.vehicle.get_location().distance(destination)
        # print(f"Distance to destination: {dist:.2f} m")
        if dist < self.threshold:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING
    
class ArriveToLocationOnTime(py_trees.behaviour.Behaviour):
    def __init__(self, vehicle, destination, time_limit=60):
        super().__init__("ArriveToLocationOnTime")
        self.vehicle = vehicle
        self.destination = destination
        self.time_limit = time_limit
        self.start_time = time.time()
        self.behavior_function = InTimeToArrivalToLocation(
            self.vehicle, self.time_limit, self.destination
        )
        # print("INITIALIZED ARRIVE TO LOCATION ON TIME")
    def update(self):
        new_status = self.behavior_function.update()
        return new_status

class LoopBehavior(py_trees.composites.Parallel):
    '''
    Loop behavior for repeating the driving sequence
    '''
    def __init__(self, name="CustomLoopParallel", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, \
                 children=None, n_loops=3, criteria = None, data_interface=None):
        super().__init__(name=name, policy=policy, children=children)
        self.n_loops = n_loops
        self.current_loop = 0

        self.start_time = time.time()
        self.criteria = criteria  # List of criteria to evaluate
        self.data_interface = data_interface  # Data interface for sending data
        # print('HELLO IN LOOP')

    def tick(self):
        # Call the original Parallel tick
        # print('HELLO IN LOOP:',self.current_loop, self.status)
        for node in super().tick():
            # print('TICKING NODE:', node.name, 'Status:', node.status)
            yield node
        # print('--END OF TICK--')

        # Custom logic after children have been ticked
        if self.status == py_trees.common.Status.SUCCESS:
            results = {}
            if self.criteria is not None:
                for crit in self.criteria:
                    # Assumes each criterion exposes its actual_value, adjust as needed
                    crit_name = crit.name
                    results[crit_name] = crit.actual_value
                    
                    #Reset actual value for next loop
                    crit.actual_value = 0

            # Add loop-specific information
            results["current_loop"] = self.current_loop
            results["elapsed_time"] = time.time() - self.start_time
            # Send the results through the data interface, if provided
            print(results)
            if self.data_interface is not None:
                self.data_interface.send_data(results, type="send")

            self.start_time = time.time()

            self.current_loop += 1
            if self.current_loop < self.n_loops:
                # Reset children for next loop
                for child in self.children:
                    child.stop(py_trees.common.Status.INVALID)
                self.status = py_trees.common.Status.RUNNING
            else:
                self.status = py_trees.common.Status.SUCCESS
        yield self