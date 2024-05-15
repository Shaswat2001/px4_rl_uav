## PX4 Simulation for Path Planning of UAV using RL

### Installing PX4

To setup PX4 with ROS2 foxy and gazebo, follow the installation steps outlined in [link](https://docs.px4.io/main/en/ros/ros2_comm.html). While installing ```PX4-Autopilot```, make a small change if you are working on ARM architecture based system -

```
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
```

Adding ```--no-nuttx``` is essential as it there are few libraries which are not available for ARM.

### Simulation Setup

To setup the simulation, perform the following steps - 

```
$ mkdir -p ~/colcon_ws/src
$ cd ~/colcon_ws/src
$ git clone https://github.com/Shaswat2001/px4_rl_uav.git
$ git clone https://github.com/PX4/px4_msgs.git
```

To build the workspace -

```
$ cd ~/colcon_ws
$ colcon build
```

### Demo 

To run the PX4 demo in three seperate terminals run the following commands - 

```
$ cd ~/PX4-Autopilot
$ make px4_sitl gazebo-classic_iris_rplidar
```

```
$ cd ~/Micro-XRCE-DDS-Agent
$ MicroXRCEAgent udp4 -p 8888
```

```
$ cd ~/ros_ws
$ source install/setup.bash
$ ros2 run px4_rl_uav px4_test.py
```
