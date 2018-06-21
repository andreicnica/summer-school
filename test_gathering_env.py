import torch
import cv2
import time
import numpy as np

from gathering_mae.single_agent_wrapper import SingleAgentGatheringEnv
from configs.utils import load_config

np.set_printoptions(threshold=np.nan, linewidth=np.nan, precision=2)


if __name__ == '__main__':

    # Get default config
    cfg = load_config("configs/default_env.yaml")

    EVAL_STEPS = 100000
    return_span = 20

    no_agents = cfg.no_agents
    # Force visual
    cfg.visualize = True
    visualize = cfg.visualize

    env = SingleAgentGatheringEnv(cfg)
    env_step = 0
    ep_r = 0

    done = 1

    agent0_r = []
    start_time = time.time()

    while env_step < EVAL_STEPS:
        # check if env needs reset
        if done:
            obs, r, done = env.restart_game()
            if visualize:
                env.render(imshow=True)

            print("Episode finished:")
            print(f"Return per episode {return_span}:".format(sum(agent0_r[-return_span:])))
            print("Agent0 mean reward: {}".format(np.mean(agent0_r)))

        actions = np.random.randint(7, size=no_agents)

        if visualize:
            # actions[1] = 4
            actions[0] = -1
            while actions[0] == -1:
                key = cv2.waitKey(0) & 0xFF

                # if the 'ESC' key is pressed, Quit
                if key == 27:
                    quit()
                elif key == 82:   # Arrow up
                    actions[0] = 0
                elif key == 83:  # Arrow right
                    actions[0] = 1
                elif key == 84:  # Arrow down
                    actions[0] = 2
                elif key == 81:  # Arrow left
                    actions[0] = 3
                elif key == 113:  # Null action q
                    actions[0] = 4
                elif key == 119:  # Turn clockwise w
                    actions[0] = 5
                elif key == 101:  # Turn counterclockwise e
                    actions[0] = 6
                elif key == 114:  # Action - laser r
                    actions[0] = 7
                else:
                    print("Unknown key: {}".format(key))

        obs, r, done, _ = env.step(actions[0])

        env_step += 1
        agent0_r.append(r)

        if visualize:
            print(f"Step: {env_step};\t Reward: {r} | Return: {sum(agent0_r[-return_span:])}\r")
            env.render(imshow=True)

        if env_step % 10000 == 0:
            print(time.time() - start_time)
            start_time = time.time()
