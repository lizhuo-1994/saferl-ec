# Safe Reinforcement Learning via Episodic Control
  SEC is based on [tianshou](https://tianshou.readthedocs.io) platform and safe RL algorithm benchmark [FSRL](https://fsrl.readthedocs.io). Please refer to the original repo for details.


## Environment
  Ubuntu 20.04.6 LTS, NVIDIA GeForce GTX 1080 Ti, cuda 12.2, nvidia driver 535.86.05, and python 3.8.5

## Install
  * pip3 install wheel==0.38.4
  * pip3 install setuptools==68.0.0
  * pip3 install -r requirements.txt

## Implementation:
  The implementation of episodic memory, state measurement, and reward shaping are in fsrl/data/abstracter.py

## Execution:
  * python  train_ddpgl_agent.py --task SafetyBallCircle-v0 --epoch 50  --episodic True --episodic_step 3 

    or

  * bash scripts/train_navigation_episodic.sh

## Experiment results:
  Data will be automatically saved into ./result