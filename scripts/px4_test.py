#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
from rclpy.node import Node
import time
import threading
import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV')
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand
from px4_rl_uav.agent import DDPG,TD3,SAC,SoftQ
from px4_rl_uav.pytorch_model import GaussianPolicyNetwork, PolicyNetwork,QNetwork,VNetwork,PhasicPolicyNetwork,PhasicQNetwork,ConstraintNetwork,MultiplierNetwork,SafePolicyNetwork,RealNVP,FeatureExtractor

from px4_rl_uav.replay_buffer.Uniform_RB import ReplayBuffer,VisionReplayBuffer
from px4_rl_uav.replay_buffer.Auxiliary_RB import AuxReplayBuffer
from px4_rl_uav.replay_buffer.Constraint_RB import ConstReplayBuffer,CostReplayBuffer

from px4_rl_uav.exploration.OUActionNoise import OUActionNoise
from px4_rl_uav.environment.GazeboEnv.Quadrotor.BaseGazeboUAVVelObsEnvSimpPX4 import BaseGazeboUAVVelObsEnvSimpPX4

class StatePublisher(Node):

    def __init__(self):
        super().__init__('uam_state_pubslisher')
        self.uam_state_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.uam_offboard_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand,"/fmu/in/vehicle_command",10)
        self.position = [0.0,0.0,-2.0]
        self.velocity = None
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.offboard_setpoint_counter_ = 0

    def timer_callback(self):
        if self.offboard_setpoint_counter_ == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

            self.arm()

        self.publish_offboard_mode()
        self.publish_state()

        if self.offboard_setpoint_counter_ < 200:
            self.offboard_setpoint_counter_+=1

    def arm(self):

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def publish_vehicle_command(self,command, param1 = 0.0,param2 = 0.0):

        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.vehicle_command_publisher.publish(msg)
    
    def publish_state(self):

        msg = TrajectorySetpoint()

        if self.velocity is None:
            msg.position = [0.0,0.0,-2.0]
            msg.yaw = -3.14
        else:
            msg.velocity = self.velocity
        
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.uam_state_publisher.publish(msg)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.uam_offboard_mode_publisher.publish(msg)

def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("Environment",nargs="?",type=str,default="uav_vel_obs_gazebo1",help="Name of OPEN AI environment")
    parser.add_argument("input_shape",nargs="?",type=int,default=[],help="Shape of environment state")
    parser.add_argument("n_actions",nargs="?",type=int,default=[],help="shape of environment action")
    parser.add_argument("max_action",nargs="?",type=float,default=[],help="Max possible value of action")
    parser.add_argument("min_action",nargs="?",type=float,default=[],help="Min possible value of action")

    parser.add_argument("Algorithm",nargs="?",type=str,default="SAC",help="Name of RL algorithm")
    parser.add_argument('tau',nargs="?",type=float,default=0.005)
    parser.add_argument('gamma',nargs="?",default=0.99)
    parser.add_argument('actor_lr',nargs="?",type=float,default=0.0001,help="Learning rate of Policy Network")
    parser.add_argument('critic_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the Q Network")
    parser.add_argument('mult_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the LAG constraint")

    parser.add_argument("mem_size",nargs="?",type=int,default=100000,help="Size of Replay Buffer")
    parser.add_argument("batch_size",nargs="?",type=int,default=64,help="Batch Size used during training")
    parser.add_argument("n_episodes",nargs="?",type=int,default=50000,help="Total number of episodes to train the agent")
    parser.add_argument("target_update",nargs="?",type=int,default=2,help="Iterations to update the target network")
    parser.add_argument("vision_update",nargs="?",type=int,default=5,help="Iterations to update the vision network")
    parser.add_argument("delayed_update",nargs="?",type=int,default=100,help="Iterations to update the second target network using delayed method")
    parser.add_argument("enable_vision",nargs="?",type=bool,default=False,help="Whether you want to integrate sensor data")
    
    # SOFT ACTOR PARAMETERS
    parser.add_argument("temperature",nargs="?",type=float,default=0.2,help="Entropy Parameter")
    parser.add_argument("log_std_min",nargs="?",type=float,default=np.log(1e-4),help="")
    parser.add_argument("log_std_max",nargs="?",type=float,default=np.log(4),help="")
    parser.add_argument("aux_step",nargs="?",type=int,default=8,help="How often the auxiliary update is performed")
    parser.add_argument("aux_epoch",nargs="?",type=int,default=6,help="How often the auxiliary update is performed")
    parser.add_argument("target_entropy_beta",nargs="?",type=float,default=-3,help="")
    parser.add_argument("target_entropy",nargs="?",type=float,default=-3,help="")

    # MISC VARIABLES 
    parser.add_argument("save_rl_weights",nargs="?",type=bool,default=True,help="save reinforcement learning weights")
    parser.add_argument("save_results",nargs="?",type=bool,default=True,help="Save average rewards using pickle")

    # USL 
    parser.add_argument("eta",nargs="?",type=float,default=0.05,help="USL eta")
    parser.add_argument("delta",nargs="?",type=float,default=0.1,help="USL delta")
    parser.add_argument("Niter",nargs="?",type=int,default=20,help="Iterations")
    parser.add_argument("cost_discount",nargs="?",type=float,default=0.99,help="Iterations")
    parser.add_argument("kappa",nargs="?",type=float,default=5,help="Iterations")
    parser.add_argument("cost_violation",nargs="?",type=int,default=20,help="Save average rewards using pickle")

    # Safe RL parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("safe_max_action",nargs="?",type=float,default=[],help="Max possible value of safe action")
    parser.add_argument("safe_min_action",nargs="?",type=float,default=[],help="Min possible value of safe action")

    # Environment Teaching parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("teach_alg",nargs="?",type=str,default="alp_gmm",help="How to change the environment")

    # Environment parameters List
    parser.add_argument("max_obstacles",nargs="?",type=int,default=10,help="Maximum number of obstacles need in the environment")
    parser.add_argument("obs_region",nargs="?",type=float,default=6,help="Region within which obstacles should be added")

    # ALP GMM parameters
    parser.add_argument('gmm_fitness_fun',nargs="?", type=str, default="aic")
    parser.add_argument('warm_start',nargs="?", type=bool, default=False)
    parser.add_argument('nb_em_init',nargs="?", type=int, default=1)
    parser.add_argument('min_k', nargs="?", type=int, default=2)
    parser.add_argument('max_k', nargs="?", type=int, default=11)
    parser.add_argument('fit_rate', nargs="?", type=int, default=250)
    parser.add_argument('alp_buffer_size', nargs="?", type=int, default=500)
    parser.add_argument('random_task_ratio', nargs="?", type=int, default=0.2)
    parser.add_argument('alp_max_size', nargs="?", type=int, default=None)

    args = parser.parse_args("")

    return args

def train(args,env,agent):

    running_counter = 0
    publisher_obj = StatePublisher()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher_obj)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    time.sleep(10.0)

    s = env.reset_test(np.array([1.5,1.5,2]),30,args.Algorithm)
    agent.load(args.Environment)
    start_time = time.time()

    while True:

        start_time = time.time()
        action = agent.choose_action(s,"testing")
        print(f"Time in seconds : {time.time() - start_time}")
        next_state,rwd,done,info = env.step(action)

        publisher_obj.velocity = [env.vel[0],env.vel[1],-2.0]

        if done:
            break
            
        s = next_state
        time.sleep(0.07)
        # print(env.check_contact)

    # f = open("config/saves/velocity_nine.pkl","wb")
    # pickle.dump(velocity_traj,f)
    # f.close()

if __name__=="__main__":

    rclpy.init(args=None)

    args = build_parse()
    env = BaseGazeboUAVVelObsEnvSimpPX4()

    if args.enable_vision:
        vision_model = FeatureExtractor(None,None,12)
        replay_buffer = VisionReplayBuffer
    else:
        vision_model = None
        replay_buffer = ReplayBuffer
    
    args.state_size = env.state_size
    args.input_shape = env.state_size
    args.n_actions = env.action_space.shape[0]
    args.max_action = env.action_space.high
    args.min_action = env.action_space.low
    args.safe_max_action = env.safe_action_max
    args.safe_min_action = -env.safe_action_max

    for i in ["TD3"]:

        args.Algorithm = i

        if args.Algorithm == "DDPG":
            agent = DDPG.DDPG(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = replay_buffer,exploration = OUActionNoise,vision = vision_model)
        elif args.Algorithm == "TD3":
            agent = TD3.TD3(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
        elif args.Algorithm == "SAC":
            agent = SAC.SAC(args = args,policy = GaussianPolicyNetwork,critic = QNetwork,valueNet=VNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
        elif args.Algorithm == "SoftQ":
            agent = SoftQ.SoftQ(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)

        train(args,env,agent)