import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.sac import SAC
from ray.tune.registry import register_env
from rl_environment import CodeGenerationEnv
import os
import json
from datetime import datetime

def env_creator(env_config):
    """Environment creator function for Ray."""
    return CodeGenerationEnv(env_config.get("problems", None))

def train_rl_agent(algorithm="PPO", num_iterations=100, checkpoint_freq=10):
    """
    Train an RL agent for code generation.

    Args:
        algorithm: RL algorithm to use ("PPO", "DQN", "SAC")
        num_iterations: Number of training iterations
        checkpoint_freq: How often to save checkpoints
    """

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env("code_generation_env", env_creator)

    # Configuration based on algorithm
    if algorithm == "PPO":
        config = {
            "env": "code_generation_env",
            "env_config": {
                "problems": [
                    "Write a function to add two numbers",
                    "Create a function to check if a string is a palindrome",
                    "Write a function to compute factorial",
                    "Implement a simple calculator",
                    "Write a function to find the maximum in a list",
                    "Create a function to reverse a string",
                    "Write a function to check if a number is prime",
                    "Implement bubble sort",
                    "Write a function to calculate fibonacci numbers",
                    "Create a function to validate email addresses"
                ]
            },
            # PPO specific config
            "framework": "torch",
            "num_workers": 2,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 200,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 30,
            "lr": 5e-5,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
        }
        trainer_class = PPO

    elif algorithm == "DQN":
        config = {
            "env": "code_generation_env",
            "env_config": {
                "problems": [
                    "Write a function to add two numbers",
                    "Create a function to check if a string is a palindrome",
                    "Write a function to compute factorial",
                    "Implement a simple calculator"
                ]
            },
            # DQN specific config
            "framework": "torch",
            "num_workers": 1,
            "buffer_size": 10000,
            "learning_starts": 1000,
            "train_batch_size": 32,
            "target_network_update_freq": 500,
            "gamma": 0.99,
            "lr": 1e-4,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 10000,
            },
        }
        trainer_class = DQN

    elif algorithm == "SAC":
        config = {
            "env": "code_generation_env",
            "env_config": {
                "problems": [
                    "Write a function to add two numbers",
                    "Create a function to check if a string is a palindrome",
                    "Write a function to compute factorial"
                ]
            },
            # SAC specific config
            "framework": "torch",
            "num_workers": 1,
            "buffer_size": 10000,
            "learning_starts": 1000,
            "train_batch_size": 128,
            "gamma": 0.99,
            "tau": 0.005,
            "target_entropy": "auto",
            "optimization": {
                "actor_learning_rate": 3e-4,
                "critic_learning_rate": 3e-4,
                "entropy_learning_rate": 3e-4,
            },
        }
        trainer_class = SAC

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Create trainer
    trainer = trainer_class(config=config)

    # Training results
    results = []

    print(f"Starting {algorithm} training...")
    print(f"Environment: Code Generation")
    print(f"Training for {num_iterations} iterations")

    for i in range(num_iterations):
        # Train for one iteration
        result = trainer.train()

        # Store results
        results.append({
            "iteration": i,
            "episode_reward_mean": result.get("episode_reward_mean", 0),
            "episode_len_mean": result.get("episode_len_mean", 0),
            "episodes_total": result.get("episodes_total", 0),
            "timesteps_total": result.get("timesteps_total", 0),
        })

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations}")
            print(f"  Mean Reward: {result.get('episode_reward_mean', 0):.2f}")
            print(f"  Mean Episode Length: {result.get('episode_len_mean', 0):.2f}")
            print(f"  Total Episodes: {result.get('episodes_total', 0)}")

        # Save checkpoint
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_path = trainer.save()
            print(f"Checkpoint saved at: {checkpoint_path}")

    # Final checkpoint
    final_checkpoint = trainer.save()
    print(f"Training completed! Final checkpoint: {final_checkpoint}")

    # Save training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"rl_training_results_{algorithm}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            "algorithm": algorithm,
            "config": config,
            "results": results,
            "final_checkpoint": final_checkpoint
        }, f, indent=2)

    print(f"Training results saved to: {results_file}")

    # Cleanup
    trainer.stop()
    ray.shutdown()

    return final_checkpoint, results

def evaluate_trained_agent(checkpoint_path: str, algorithm="PPO", num_episodes=10):
    """
    Evaluate a trained RL agent.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        algorithm: Algorithm used for training
        num_episodes: Number of episodes to evaluate
    """

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    register_env("code_generation_env", env_creator)

    # Load configuration
    if algorithm == "PPO":
        trainer_class = PPO
    elif algorithm == "DQN":
        trainer_class = DQN
    elif algorithm == "SAC":
        trainer_class = SAC
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Restore trainer from checkpoint
    trainer = trainer_class.from_checkpoint(checkpoint_path)

    print(f"Evaluating trained {algorithm} agent...")
    print(f"Running {num_episodes} evaluation episodes")

    # Run evaluation episodes
    evaluation_results = []

    for episode in range(num_episodes):
        # Create environment
        env = CodeGenerationEnv()

        # Reset environment
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        episode_info = {
            "episode": episode + 1,
            "problem": env.current_problem,
            "steps": [],
            "total_reward": 0,
            "success": False
        }

        while not done and steps < env.max_attempts:
            # Get action from trained policy
            action = trainer.compute_single_action(state)

            # Take step
            next_state, reward, done, _, info = env.step(action)

            # Record step
            episode_info["steps"].append({
                "step": steps + 1,
                "action": action,
                "reward": reward,
                "test_success": info['test_result']['returncode'] == 0,
                "code_length": len(info['code'])
            })

            total_reward += reward
            state = next_state
            steps += 1

        episode_info["total_reward"] = total_reward
        episode_info["success"] = any(step["test_success"] for step in episode_info["steps"])

        evaluation_results.append(episode_info)

        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Success={episode_info['success']}, Steps={steps}")

    # Calculate summary statistics
    success_rate = sum(1 for ep in evaluation_results if ep["success"]) / len(evaluation_results)
    avg_reward = sum(ep["total_reward"] for ep in evaluation_results) / len(evaluation_results)
    avg_steps = sum(len(ep["steps"]) for ep in evaluation_results) / len(evaluation_results)

    summary = {
        "success_rate": success_rate,
        "average_reward": avg_reward,
        "average_steps": avg_steps,
        "episodes": evaluation_results
    }

    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_file = f"rl_evaluation_results_{algorithm}_{timestamp}.json"

    with open(eval_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("
Evaluation Summary:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"Results saved to: {eval_file}")

    # Cleanup
    trainer.stop()
    ray.shutdown()

    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent for code generation")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN", "SAC"],
                       help="RL algorithm to use")
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                       help="Checkpoint frequency")
    parser.add_argument("--evaluate", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")

    args = parser.parse_args()

    if args.evaluate:
        # Evaluation mode
        evaluate_trained_agent(args.evaluate, args.algorithm, args.eval_episodes)
    else:
        # Training mode
        train_rl_agent(args.algorithm, args.iterations, args.checkpoint_freq)
