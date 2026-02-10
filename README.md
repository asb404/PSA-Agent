# Problem Solving Agent (PSA)

A problem solving agent built using LangChain and Ollama that can generate Python code solutions to problems and automatically test them.

## Features

- Generates Python code solutions using Ollama LLMs
- Automatically tests generated code using pytest in a sandboxed environment
- Provides analysis and feedback on the generated solutions
- Uses resource limits to prevent runaway processes
- **Reinforcement Learning Integration**: RL agents can learn optimal code generation strategies

## Prerequisites

- Python 3.8+
- Ollama installed and running with a model (e.g., `gemma3:4b`)
- Required Python packages (installed automatically)

## Installation

1. Install Ollama: https://ollama.ai/

2. Pull a model:
```bash
ollama pull gemma3:4b
```

3. Start Ollama server:
```bash
ollama serve
```

4. Install Python dependencies:
```bash
pip install langchain langchain-ollama
```

## Usage

### Web Interface (Recommended)

Start the web application for an interactive experience:

```bash
python web_app.py
```

Then visit http://localhost:8000 in your browser. The web interface provides:
- User-friendly form to input problems
- Real-time loading indicators
- Formatted display of generated code, tests, and analysis
- Example problem buttons to try

### Python API

Use the agent programmatically:

```python
from psa_agent import solve_problem

# Solve a problem
result = solve_problem("Write a function to add two numbers.")
print(result)
```

### Command Line

#### Demo Script
Run the demo script to see the agent solve multiple problems:

```bash
python demo.py
```

#### Direct Script
Run the main script with a simple example:

```bash
python psa_agent.py
```

## Architecture

- `psa_agent.py`: Main agent implementation using LangChain and Ollama
- `llm_adapter.py`: LLM client abstraction (supports multiple backends)
- `runner.py`: Sandboxed test execution with resource limits
- `demo.py`: Demonstration script with multiple example problems

## How It Works

1. **Problem Input**: User provides a natural language problem description
2. **Code Generation**: LLM generates Python code to solve the problem
3. **Code Cleaning**: Removes markdown formatting from generated code
4. **Testing**: Code is written to a temporary file and tested using pytest
5. **Analysis**: LLM analyzes the test results and provides feedback
6. **Output**: Returns generated code, test results, and analysis

## Security

- Code execution is sandboxed with CPU and memory limits
- Uses subprocess isolation for testing
- Prevents infinite loops and excessive resource usage

## Examples

The agent can solve various coding problems:

- Mathematical functions (factorial, fibonacci, etc.)
- String manipulation (palindromes, anagrams, etc.)
- Data structures and algorithms
- Simple calculators and utilities

## Customization

- Change the LLM model in `psa_agent.py`
- Modify prompts for different solution styles
- Add more sophisticated testing frameworks
- Extend with additional tools for the agent

## Reinforcement Learning (RL) Integration

The PSA includes RL capabilities using RLlib to learn optimal code generation strategies.

### RL Features

- **Custom Gym Environment**: Code generation environment with state/action/reward definitions
- **Multiple Algorithms**: Support for PPO, DQN, and SAC algorithms
- **Strategy Learning**: RL agents learn which generation approaches work best for different problems
- **Performance Tracking**: Comprehensive training and evaluation metrics

### RL Usage

#### Quick Demo
Run the RL demonstration:
```bash
python rl_demo.py
```

#### Full Training
Train an RL agent with custom parameters:
```bash
# Train with PPO (default)
python rl_trainer.py --iterations 100 --algorithm PPO

# Train with DQN
python rl_trainer.py --iterations 100 --algorithm DQN

# Train with SAC
python rl_trainer.py --iterations 100 --algorithm SAC
```

#### Evaluation
Evaluate a trained model:
```bash
python rl_trainer.py --evaluate /path/to/checkpoint --eval-episodes 20
```

### RL Concepts

- **State**: Problem hash, attempt count, previous success, code metrics
- **Actions**: Direct generation, step-by-step reasoning, modification strategies
- **Rewards**: Test success (+10), efficiency bonuses, code quality penalties
- **Environment**: Gym-compatible environment for RL training

### RL Architecture

- `rl_environment.py`: Gym environment for code generation
- `rl_trainer.py`: Training and evaluation scripts
- `rl_demo.py`: Interactive demonstration of RL concepts
