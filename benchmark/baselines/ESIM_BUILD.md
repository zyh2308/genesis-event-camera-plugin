# ESIM build record

**Status:** `supported_but_not_reproducibly_built`  
**Date:** 2026-09-29  
**Source:** `/home/科研/Eventbased_WAM/code/rpg_esim`  
**Commit:** `4cf0b8952e9f58f674c3098f1b027a4b6db53427`  
**Official repository:** <https://github.com/uzh-rpg/rpg_esim>

## Source and intended build route

The checkout is the official ESIM ROS/catkin source tree. Its README directs
users to the official wiki for installation, and `dependencies.yaml` lists
catkin_simple, ze_oss, gflags/glog/eigen catkin packages, minkindr, rpg_dvs_ros,
assimp and yaml-cpp dependencies. The repository does not contain a standalone
pure-Python event generator that can be substituted for the official build.

## Build attempt

Environment:

```text
Python 3.14.6 (/root/miniconda3/bin/python)
Ubuntu host tools: cmake and make present
ROS/catkin: catkin_make not found; roscore/roslaunch not found
```

Exact command attempted:

```bash
cd /home/科研/Eventbased_WAM/code/rpg_esim
catkin_make
```

Observed result:

```text
/bin/bash: line 6: catkin_make: command not found
EXIT_CODE=127
```

No dependency clone, system package installation, or unrelated ROS setup was
performed. There is no `build/`, `devel/`, or `install/` artifact in this
checkout. The failure is therefore an environment/build-availability failure,
not an ESIM algorithm failure.

## Benchmark decision

ESIM is **not included in the executable formal result table** at this point.
The benchmark may use only Ours vs V2E until an official ESIM build and a real
ESIM smoke output pass. The wrapper in `esim_wrapper.py` is a fail-closed I/O
adapter; it does not implement “ESIM-like” event generation.

