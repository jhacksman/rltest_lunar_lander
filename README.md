# Lunar Lander RL Simulation

This project implements an interactive Lunar Lander simulation and reinforcement learning (RL) environment using Python, Pygame, Stable-Baselines3, and Matplotlib.

## Folder Structure

- **main.py**: Entry point.
- **env/**: Contains the custom OpenAI Gym environment and a Monitor wrapper.
- **rl/**: Contains RL training functions and custom callbacks.
- **ui/**: Contains the Pygame-based UI code (widgets, main menu, demo mode).
- **utils/**: Contains utility functions (e.g. plotting).

## Requirements

Install dependencies with:

pip install -r requirements.txt

-------------------------------

lunar_lander_rl/
├── README.md
├── requirements.txt
├── main.py
├── env/
│   ├── __init__.py
│   ├── lunar_lander.py
│   └── monitor.py
├── rl/
│   ├── __init__.py
│   ├── callbacks.py
│   └── train.py
├── ui/
│   ├── __init__.py
│   ├── widgets.py
│   ├── menu.py
│   └── demo.py
└── utils/
    ├── __init__.py
    └── plot_utils.py

