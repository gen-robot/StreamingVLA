import collections
import dataclasses
import logging
import math
import pathlib
import time 

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import streaming_websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from typing import Optional, List, Tuple

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256 # resolution used to render training data

@dataclasses.dataclass
class Args:
    
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10
    timing_output_path: str = "your_timing_path" 
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  
    num_trials_per_task: int = 50  
    video_out_path: str = "your_video_path"  

    seed: int = 7  # Random Seed (for reproducibility)


    use_judger: bool = True # To decide whether to use the judger's feedback for replanning
    action_left: int = 2 # Number of actions left threshold to trigger the judger check
    threshold: float = 4.43 # Threshold for the judger's confidence to decide whether to jump to a new observation
    
   

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220 
    elif args.task_suite_name == "libero_object":
        max_steps = 280 
    elif args.task_suite_name == "libero_goal":
        max_steps = 300 
    elif args.task_suite_name == "libero_10":
        max_steps = 520 
    elif args.task_suite_name == "libero_90":
        max_steps = 400 
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    suite_total_episodes, suite_total_successes = 0, 0
    suite_total_time_success, suite_total_actions_success = 0.0, 0
    
    pathlib.Path(args.timing_output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.timing_output_path, 'w', encoding='utf-8') as timing_file:
        timing_file.write(f"--- Timing Results for Task Suite: {args.task_suite_name} (Success-Only Metrics) ---\n")
        # Note: Only SUCCESS lines will be written below
        timing_file.write("Task | Episode | Success | Total Time (s) | Actions | Speed (s/action)\n")
        timing_file.write("-" * 80 + "\n")
    
        for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Overall Task Suite"):
            # Get task
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            
            logging.info(f"\n--- Starting Task {task_id+1}/{num_tasks_in_suite}: {task_description} ---")

            task_episode_results: List[Tuple[bool, float, int]] = [] 
            task_successful_episodes_data: List[Tuple[float, int]] = []
            task_sim_time_list: List[float] = [] 

            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description} | Episode: {episode_idx + 1}")
               
                # Reset environment
                env.reset()
                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                steps_since_replan = 0 
                action_states = np.zeros(7, dtype=np.float32)
                episode_start_time = 0.0 
                episode_actions = 0
                jump_times = 0
                obs_times = 0
                episode_success = False
                norm_new_obs=True
                new_task = True
                
                while t < max_steps + args.num_steps_wait:
                    try:
                        # Stabilization steps
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Start timing after stabilization
                        if episode_start_time == 0.0:
                            episode_start_time = time.monotonic()
                        
                        # Get preprocessed image and prepare element dict
                        # ... (image preprocessing and element dict creation logic remains unchanged)
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
                        replay_images.append(img)
                        if (args.use_judger and steps_since_replan == (args.replan_steps-args.action_left) % args.replan_steps ):
                            
                            action_left_sum = client.get_left_queue_actions()
                            print(f"action_left_sum: {action_left_sum}")
                            if not np.any(action_left_sum):
                                action_left_sum = None
                                norm_new_obs=True
                            if action_left_sum is not None:
                                element = {
                                    "observation/image": img,
                                    "observation/wrist_image": wrist_img,
                                    "observation/state": np.concatenate(
                                        (
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )
                                    ),
                                    "prompt": str(task_description),
                                    "observation/action_left_sum": action_left_sum,
                                    "observation/action_states": action_states,
                                    "observation/threshold": args.threshold,
                                }
                                jump_times+=1
                                obs_times+=1
                                norm_new_obs=False
                                client.infer(element,new_task)
                                
                        if (steps_since_replan == 0 and norm_new_obs) :
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                                "observation/action_left_sum": None,
                                "observation/action_states": action_states,
                                "observation/threshold": 100000.0, # default: normal inference without judger feedback
                            }

                            obs_times+=1
                            client.infer(element,new_task)
                            new_task=False
                            
                        while True:                      
                            action_data = client.get_next_action(timeout=20)                       
                            if action_data is not None:
                                break
                            
                        if "norm_exceeded" in action_data:
                            norm_new_obs = True                         
                            continue

                        if "actions" in action_data:
                            episode_actions=episode_actions+1
                            action = action_data["actions"]
                            action_states+=action

                            sim_time=time.monotonic()
                            obs, reward, done, info = env.step(action.tolist())
                            sim_end_time = time.monotonic()-sim_time
                            task_sim_time_list.append(sim_end_time)
                            steps_since_replan = (steps_since_replan + 1) % args.replan_steps
                            
                            t += 1
                         
                        if done:
                            episode_success = True

                            break

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                episode_end_time = time.monotonic()
                suffix = "success" if done else "failure"
                
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"{task_description}_{task_id}_{(episode_idx)%3}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=20,
                )

                if episode_start_time > 0:
                    episode_total_time = episode_end_time - episode_start_time
                else:
                    episode_total_time = 0.0
                    episode_actions = 0 # Actions before start time are dummy and not counted

                
                task_episode_results.append((episode_success, episode_total_time, episode_actions))

                if episode_success:
                    task_successful_episodes_data.append((episode_total_time, episode_actions))
                    action_speed = episode_total_time / episode_actions if episode_actions > 0 else 0.0
                    timing_file.write(f"{episode_idx + 1:7d} | SUCCESS | {episode_total_time:14.2f} | {episode_actions:7d} | {action_speed:14.4f}\n")
                
                success_status = "SUCCESS" if episode_success else "FAILURE"
                logging.info(f"Episode {episode_idx + 1} Result: {success_status} | Time: {episode_total_time:.2f} s | Actions: {episode_actions}")
            
           
            num_trials = len(task_episode_results)
            num_successes = len(task_successful_episodes_data)

            if num_trials > 0:
                task_avg_success_rate = num_successes / num_trials
                if num_successes > 0:
                    task_success_only_time = sum(t for t, _ in task_successful_episodes_data)
                    task_success_only_actions = sum(a for _, a in task_successful_episodes_data)
                    
                    task_avg_time_success = task_success_only_time / num_successes 
                    task_avg_actions_success = task_success_only_actions / num_successes 
                    task_avg_speed = task_success_only_actions / task_success_only_time if task_success_only_time > 0 else 0.0 
                    
                    suite_total_time_success += task_success_only_time
                    suite_total_actions_success += task_success_only_actions
                else:
                    task_avg_time_success, task_avg_actions_success, task_avg_speed = 0.0, 0.0, 0.0

                task_avg_sim_time = sum(task_sim_time_list) / len(task_sim_time_list) if task_sim_time_list else 0.0
                
                timing_file.write("-" * 80 + "\n")
                timing_file.write(f"Task Summary: {task_description}\n")
                timing_file.write(f"  Avg Success Rate (All Trials): {task_avg_success_rate:.4f} ({num_successes}/{num_trials})\n")
                timing_file.write(f"  Avg Episode Time (Success Only): {task_avg_time_success:.2f} seconds\n")
                timing_file.write(f"  Avg Actions/Episode (Success Only): {task_avg_actions_success:.2f} actions\n")
                timing_file.write(f"  Avg Action Speed (Success Only): {task_avg_speed:.2f} actions/second\n")
                timing_file.write(f"  Avg Sim Step Time: {task_avg_sim_time * 1000:.4f} ms\n")
                timing_file.write("=" * 80 + "\n\n")

                suite_total_episodes += num_trials
                suite_total_successes += num_successes

                logging.info(f"Task Success Rate: {task_avg_success_rate:.4f}")

        current_suite_avg_success_rate = suite_total_successes / suite_total_episodes if suite_total_episodes > 0 else 0.0
        logging.info(f"Current Total Success Rate: {current_suite_avg_success_rate * 100:.1f}%")

    
        if suite_total_successes > 0:
            suite_avg_success_rate = suite_total_successes / suite_total_episodes
            suite_avg_time_success = suite_total_time_success / suite_total_successes
            suite_avg_actions_success = suite_total_actions_success / suite_total_successes
            suite_avg_speed_success = suite_total_actions_success / suite_total_time_success if suite_total_time_success > 0 else 0.0
        else:
            suite_avg_success_rate, suite_avg_time_success, suite_avg_actions_success, suite_avg_speed_success = 0.0, 0.0, 0.0, 0.0
            
        timing_file.write("\n\n")
        timing_file.write("################################################################################\n")
        timing_file.write("### OVERALL SUITE SUMMARY ###\n")
        timing_file.write("################################################################################\n")
        timing_file.write(f"Total Tasks Completed: {num_tasks_in_suite}\n")
        timing_file.write(f"Total Episodes Attempted: {suite_total_episodes}\n")
        timing_file.write(f"Total Successful Episodes: {suite_total_successes}\n")
        timing_file.write(f"Overall Success Rate (All Trials): {suite_avg_success_rate:.4f} ({suite_total_successes}/{suite_total_episodes})\n")
        timing_file.write(f"Overall Avg Episode Time (Success Only): {suite_avg_time_success:.2f} seconds\n")
        timing_file.write(f"Overall Avg Actions/Episode (Success Only): {suite_avg_actions_success:.2f} actions\n")
        timing_file.write(f"Overall Avg Action Speed (Success Only): {suite_avg_speed_success:.2f} actions/second\n")

        logging.info(f"Total success rate: {suite_avg_success_rate:.4f}")
        logging.info(f"Total episodes: {suite_total_episodes}")    

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
