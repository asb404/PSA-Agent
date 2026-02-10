#!/usr/bin/env python3
"""
Demonstration of Reinforcement Learning for Problem Solving Agent.

This script shows how to:
1. Test the RL environment
2. Train a simple RL agent
3. Evaluate trained policies
"""

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from rl_environment import CodeGenerationEnv
import json
from datetime import datetime

def env_creator(env_config):
    """Environment creator for Ray."""
    return CodeGenerationEnv(env_config.get("problems", None))

def demo_environment():
    """Demonstrate the RL environment."""
    print("=" * 60)
    print("RL ENVIRONMENT DEMO")
    print("=" * 60)

    env = CodeGenerationEnv()

    print("Testing RL environment with random actions...")

    # Run a few episodes with random actions
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")

        state, _ = env.reset()
        print(f"Initial state: {state}")

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 3:  # Limit steps for demo
            # Take random action
            action = env.action_space.sample()

            # Step environment
            next_state, reward, done, _, info = env.step(action)

            print(f"Step {steps + 1}: Action={action}, Reward={reward:.2f}, Done={done}")
            print(f"  Code length: {len(info['code'])}")
            print(f"  Test success: {info['test_result']['returncode'] == 0}")

            total_reward += reward
            state = next_state
            steps += 1

        print(f"Episode finished - Total Reward: {total_reward:.2f}")

    env.close()
    print("\nEnvironment demo completed!")

def quick_training_demo():
    """Demonstrate quick RL training."""
    print("\n" + "=" * 60)
    print("QUICK RL TRAINING DEMO")
    print("=" * 60)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=2)

    # Register environment
    register_env("code_generation_env", env_creator)

    # Simple PPO config for quick demo
    config = {
        "env": "code_generation_env",
        "env_config": {
            "problems": [
                "Write a function to add two numbers",
                "Create a function to check if a string is a palindrome"
            ]
        },
        "framework": "torch",
        "num_workers": 0,  # Run on main process for demo
        "rollout_fragment_length": 50,
        "train_batch_size": 100,
        "sgd_minibatch_size": 50,
        "num_sgd_iter": 5,
        "lr": 1e-4,
        "gamma": 0.99,
    }

    print("Initializing PPO trainer...")
    trainer = PPO(config=config)

    print("Training for 5 iterations (this may take a moment)...")

    training_results = []
    for i in range(5):
        result = trainer.train()

        reward = result.get("episode_reward_mean", 0)
        episodes = result.get("episodes_total", 0)

        training_results.append({
            "iteration": i + 1,
            "mean_reward": reward,
            "total_episodes": episodes
        })

        print(f"Iteration {i + 1}: Mean Reward = {reward:.2f}, Episodes = {episodes}")

    # Quick evaluation
    print("\nEvaluating trained policy...")
    env = CodeGenerationEnv()

    total_reward = 0
    successes = 0
    num_episodes = 3

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < env.max_attempts:
            action = trainer.compute_single_action(state)
            next_state, reward, done, _, info = env.step(action)

            episode_reward += reward
            state = next_state
            steps += 1

            if info['test_result']['returncode'] == 0:
                successes += 1
                break

        total_reward += episode_reward
        print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {info['test_result']['returncode'] == 0}")

    avg_reward = total_reward / num_episodes
    success_rate = successes / num_episodes

    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"rl_demo_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            "training_results": training_results,
            "evaluation": {
                "average_reward": avg_reward,
                "success_rate": success_rate,
                "num_episodes": num_episodes
            }
        }, f, indent=2)

    print(f"\nDemo results saved to: {results_file}")

    # Cleanup
    trainer.stop()
    env.close()
    ray.shutdown()

def show_rl_concepts():
    """Explain RL concepts used in this implementation."""
    print("\n" + "=" * 60)
    print("RL CONCEPTS EXPLAINED")
    print("=" * 60)

    concepts = {
        "State": "Encoded representation of current problem, attempt count, previous test success, code metrics",
        "Actions": [
            "0: Direct generation",
            "1: Step-by-step reasoning",
            "2: Modify previous attempt",
            "3: Use simpler approach",
            "4: Use more complex approach"
        ],
        "Rewards": [
            "+10.0 for test success",
            "-2.0 for test failure",
            "Efficiency bonus (fewer attempts = higher reward)",
            "Code quality bonuses/penalties",
            "Strategy diversity bonuses"
        ],
        "Environment": "Gym-compatible environment that generates code, tests it, and provides feedback",
        "Algorithms": [
            "PPO: Proximal Policy Optimization (good for continuous control)",
            "DQN: Deep Q-Network (good for discrete actions)",
            "SAC: Soft Actor-Critic (good for exploration)"
        ]
    }

    for concept, explanation in concepts.items():
        print(f"\n{concept}:")
        if isinstance(explanation, list):
            for item in explanation:
                print(f"  • {item}")
        else:
            print(f"  {explanation}")

def main():
    """Run the complete RL demonstration."""
    print("🤖 Problem Solving Agent - Reinforcement Learning Demo")
    print("This demo shows how RL can improve code generation strategies.")

    try:
        # Show concepts first
        show_rl_concepts()

        # Demo environment
        demo_environment()

        # Quick training demo
        quick_training_demo()

        print("\n" + "=" * 60)
        print("🎉 RL DEMO COMPLETED!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("• RL can learn optimal strategies for code generation")
        print("• The agent improves from random actions to learned policies")
        print("• Success rate and efficiency improve with training")
        print("• Different RL algorithms can be applied to different problems")

        print("\nNext steps:")
        print("• Run 'python rl_trainer.py' for full training")
        print("• Try different algorithms: --algorithm DQN or --algorithm SAC")
        print("• Increase training iterations for better performance")

    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure Ollama is running and all dependencies are installed.")

if __name__ == "__main__":
    main()
