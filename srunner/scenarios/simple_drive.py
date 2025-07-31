# ScenarioRunner/Scenarios/SimpleDrive/SimpleDrive.py

import carla
import py_trees
import random
import time
import sys
import sys
import os
import signal

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
from src.util.carla_utils.debug import display_spawn_ids, display_location, display_grid, display_detection_area
from src.util.carla_utils.nx_vis import get_topology
from src.util.carla_utils.spawn_tools import initialize_scenario, spawn_npc_actors
from src.wrappers.wrapped_object_registry import WrappedObjectRegistry
from src.behaviors.drive_agent import DriveWithAgent
from src.behaviors.reset_scenario import ResetScenarioNode
from src.behaviors.loop_behavior import LoopBehavior
from src.behaviors.check_arrival import CheckArrival
from src.behaviors.location_timer import ArriveToLocationOnTime
from src.behaviors.npc_controller import NPCActorsController
from src.behaviors.walker_controller import WalkerActorController
from src.util.ros2_nodes.ros_interface import ROSInterface

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

class SimpleDrive(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False):

        signal.signal(signal.SIGINT, self.graceful_shutdown)

        # ========== GET THESE VARIABLES FROM CONFIG IN THE FUTURE ========== #
        self.timeout=100000
        self.seed = 45
        random.seed(self.seed)

        self.n_loops = 10 # Expose this at some point
        self.num_parked = 10
        self.num_background = 0#10
        self.num_walkers = 0#10
        self.max_scenario_time = 120
        self.get_detection_area = True
        self.scenario_type = 'obstacle'  # Options: 'obstacle', 'pedestrian', 'base'
        self.ego_control_type = 'Autoware'  # Options: 'CustomAgent', 'Autoware'
        self.reveal_actors = True

        # self.filter_params = {
        #     'threshold': 45.0,
        # }
        # self.filter_params = {
        #     'model_path': 'saved_models/random_forest/traffic_actor_classifier_0.joblib',
        #     'egocentric' : True
        # }
        self.filter_params = {}
        self.filter_type = 'base'  # Use 'data_collector' for collecting data without filtering

        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.01
        # ========== END OF CONFIG VARIABLES ========== #
                
        print('Debug Mode:', debug_mode)
        
        self.world_map = world.get_map()

        #Phasing this out
        self.data_interface = DataInterface(interface_type='sender')  # Initialize the data interface

        super().__init__(
            name="SimpleDrive",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=True,
        )
        # print('PHYSICS:',self._vehicle.get_physics_control())
        # sys.exit(0)
        self.spawn_points = self.world_map.get_spawn_points()

        self.blueprint_library = self.world.get_blueprint_library()
        self.walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        
        self.reset_scenario()

        self._set_spectator_camera()
        display_detection_area(self.world, self._detection_area)
        # display_walker_spawns()

        if debug_mode:
            display_spawn_ids(self.world, self.spawn_points)
            display_grid(self.world)

        topology_json = get_topology(self._agent)
        self.data_interface.send_data(topology_json)

    def _initialize_actors(self, config):
        self.ros_interface = ROSInterface(node_name='autoware')

        self.scenario_manager = CustomScenarioManager(seed=self.seed)
        
        self.filter_interface = FilterInterface(filter_type=self.filter_type, filter_params=self.filter_params)

        self.global_route_planner = GlobalRoutePlanner(wmap = self.world_map, 
                                                      sampling_resolution=2.0)
        
        ego = CarlaDataProvider.get_hero_actor()

        if 'Autoware' in self.ego_control_type:
            ackermann_settings = carla.AckermannControllerSettings(
                speed_kp = 1.0,
                speed_ki = 0.1,
                speed_kd = 0.0,
                accel_kp = 0.01,
                accel_ki = 0.0,
                accel_kd = 0.01
            )
            ego.apply_ackermann_controller_settings(ackermann_settings)
            print(f"Applied Ackermann controller settings: {ego.get_ackermann_controller_settings()}")
        
        self._vehicle = ego

        self._agent = CustomAgent(ego, behavior="normal", reveal_actors=self.reveal_actors)
        self._npc_actor_controller = NPCActorsController(self.global_route_planner, behavior_list = [])
        self._walker_actor_controller = WalkerActorController(walker_list = [], world_map = self.world_map,
                                                                ego_vehicle = None)
        
        #For walker pathing debug
        self.destination_id = 0  


    def _create_behavior(self):
        criteria = self._create_test_criteria()
        root = LoopBehavior(
            "DriveLoop", 
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, 
            n_loops=self.n_loops,
            criteria = criteria,
            filter_interface = self.filter_interface
        )

        episode_sequence = py_trees.composites.Parallel(
            "OneEpisode",
            policy = py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        episode_sequence.add_child(DriveWithAgent(self._agent, self.filter_interface, 
                                                  self.ros_interface, self.ego_control_type))
        episode_sequence.add_child(CheckArrival(self._vehicle, self._agent.get_destination))
        episode_sequence.add_child(TimeOut(timeout = self.max_scenario_time, name="TimeOut"))
        #Add something here to check collision as well

        episode_sequence.add_child(self._npc_actor_controller)
        episode_sequence.add_child(self._walker_actor_controller)

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
    
    def _set_spectator_camera(self):
        # position spectator overhead
        spectator = self.world.get_spectator()
        x_coord, y_coord, z_coord = -45, 23, 70
        spec_tf = carla.Transform(carla.Location(x_coord,y_coord,z_coord), 
                                  carla.Rotation(pitch = -90, yaw = -90, roll = 0))
        spectator.set_transform(spec_tf)

    def reset_scenario(self):
        #Initialize the NPC controller with the generated behaviors
        scenario_output = initialize_scenario(
            self.scenario_manager,
            get_detection_area=self.get_detection_area,
            scenario_type=self.scenario_type,
            num_parked=self.num_parked,
            num_background=self.num_background,
            num_walkers=self.num_walkers
        )
        self._ego_start_location, self._destination, self._detection_area, npc_params_list = scenario_output

        self.filter_params['detection_area'] = self._detection_area
        self.filter_interface.update_filter_params(self.filter_params)

        # npc_actors already includes walker actors, walker_actors is to be passed to the walker controller
        # behavior node for fine tuned control
        npc_actor_configs, npc_actors, walker_actors = spawn_npc_actors(npc_params_list, self.world_map,
                                                         self.blueprint_library, self.walker_controller_bp)

        self.other_actors.extend(npc_actors)
        for walker in walker_actors:
            self.other_actors.append(walker['walker'])
            # self._npc_actor_controller.add_behavior(walker['walker'], walker['route'], walker['controller'])

        print(f"Spawned {len(npc_actors)} NPC actors and {len(walker_actors)} walkers.")
        self._walker_actor_controller.update_walker_list(walker_actors)
        # walker_routes = [walker['route'] for walker in walker_actors]
        # # for i, route in enumerate(walker_routes):
        # #     display_location(self.world, route[0], f'Walker Start {self.destination_id}')
        # #     display_location(self.world, route[1], f'Walker Destination {self.destination_id}')
        # #     self.destination_id += 1

        if npc_actor_configs:
            self._npc_actor_controller.update_behavior_list(npc_actor_configs)

        #TODO: Publish the ego start location and destination to the ros2
        self._set_ego_location(self._ego_start_location)

        self._agent._vehicle = self._vehicle  # Update the agent's vehicle reference  

        self._walker_actor_controller.update_ego_vehicle(self._vehicle)

        #Clear the waypoints queue and set a new destination
        self._agent.set_destination(self._destination.location, start_location = self._ego_start_location) 


        if 'Autoware' in self.ego_control_type:  
            self.init_autoware_startup()

        self.world.tick()


    def init_autoware_startup(self):
        def wait_for_subscribers(input, topic):
            while self.ros_interface.check_subscribers(topic) == 0:
                print(f"Waiting for subscribers on topic: {topic}")
                time.sleep(0.1)
            self.ros_interface.send_message(
                input=input,
                topic=topic
            )
            
        print("Initializing Autoware startup...")

        wait_for_subscribers(
            input = {
                'pose' : self._ego_start_location
            },
            topic = 'initialpose',
        )
        wait_for_subscribers(
            input = {
                'goal' : self._destination
            },
            topic = 'goal',
        )

        self.ros_interface.send_message(
            input = {},
            topic = 'engage'
        )

    def remove_all_actors(self):
        """
        Remove all actors
        """
        if not hasattr(self, 'other_actors'):
            return
        print('Removing all actors...')
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    try:
                        CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                        WrappedObjectRegistry.unregister(self.other_actors[i].id)
                    except Exception as e:
                        print(f"Error removing actor {self.other_actors[i].id}: {e}")
                self.other_actors[i] = None
        self.other_actors = []
        print(f'Registry has {WrappedObjectRegistry.count()} objects')

    def graceful_shutdown(self, signum, frame):
        """
        Gracefully shutdown the scenario
        """
        print("Shutting down scenario...")
        self.remove_all_actors()
        self.ros_interface.shutdown()
        CarlaDataProvider.cleanup()
        print("Scenario shutdown complete.")
        exit(0) 

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
        self.ros_interface.shutdown()
  