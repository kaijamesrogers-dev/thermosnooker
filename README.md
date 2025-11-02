# ThermoSnooker

ThermoSnooker is a Python project for simulating and analyzing elastic collisions in a 2D circular container. It includes simulations of:

- **SingleBallSimulation:** A single ball bouncing inside a circular container.
- **MultiBallSimulation:** Multiple balls colliding with each other and the container.
- **BrownianSimulation:** Brownian motion, where a larger 'big ball' interacts with smaller balls to model stochastic dynamics.

The project provides tools to run these simulations, visualize animations of collisions, and analyze physical properties like momentum, trajectories, and thermodynamic behavior.

## Features

- Simulate elastic collisions for single and multiple balls in a 2D container
- Model Brownian motion with a large particle
- Visualize ball trajectories and collision animations
- Analyze kinetic energy, momentum, pressure, and temperature evolution
- Compare simulation outputs to statistical physics models (Ideal Gas Law, Van der Waals, Maxwell-Boltzmann distribution)

## Installation

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/kaijamesrogers-dev/thermosnooker.git
cd thermosnooker
pip install -r requirements.txt
```

## Usage

Basic usage example:

```bash
python main.py
```

Or run specific analysis tasks (see `analysis.py` for details):

```bash
python analysis.py
```

## Documentation

- Simulation logic is implemented in `simulations.py`
- Particle and container definitions are in `balls.py`
- Analysis and visualization scripts are in `analysis.py`

See docstrings and comments throughout the code for additional guidance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License

[Specify your license here, e.g., MIT License]

## Author

Created by [kaijamesrogers-dev](https://github.com/kaijamesrogers-dev)

---

*This project is written entirely in Python.*
