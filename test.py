import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper

# create environment instance

robots = ["Panda", "Sawyer", "Jaco"]
env_names = ["Lift", "Stack", "Door"]
for robot in robots:
    for env_name in env_names:
        env = GymWrapper(
            suite.make(
                env_name=env_name,  # try with other tasks like "Stack" and "Door"
                robots=robot,  # try with other robots like "Sawyer" and "Jaco"
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                reward_shaping=True,
            ))

        # reset the environment
        state = env.reset()

        print(state[0])

        for i in range(1):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            # obs, reward, done, info = env.step(action)  # take action in the environment
            print(robot, env_name, 'obs shape:', observation.shape, 'action shape:', action.shape, reward)
# env = GymWrapper(
#     suite.make(
#     env_name="Lift", # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     has_renderer=False,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
# ))

# # reset the environment
# env.reset()

# for i in range(1):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     # obs, reward, done, info = env.step(action)  # take action in the environment
#     print('obs shape:', observation.shape, 'action shape:', action.shape)