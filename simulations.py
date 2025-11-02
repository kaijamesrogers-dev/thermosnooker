"""Simulation module for thermosnooker project.

This module contains simulation classes for modelling elastic collisions
between balls in a 2D circular container. It includes both single-ball and
multi-ball simulations, as well as a special Brownian motion simulation with
a larger "big ball" particle.
"""

import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.balls import Ball, Container


class Simulation:
    """
    Abstract base class for all simulations. Defines interface methods that must
    be implemented by derived classes.
    """

    def next_collision(self):
        """
        Advance the simulation by one collision event.
        Must be implemented by derived classes.
        """
        raise NotImplementedError('next_collision() needs to be implemented in derived classes')

    def setup_figure(self):
        """
        Create and return the figure and axes for animation.
        Must be implemented by derived classes.
        """
        raise NotImplementedError('setup_figure() needs to be implemented in derived classes')

    def run(self, num_collisions, animate=False, pause_time=0.01):
        """
        Run the simulation for a given number of collisions.

        Args:
            num_collisions (int): Number of collision events to simulate.
            animate (bool): Whether to animate the simulation.
            pause_time (float): Time between frames in the animation.
        """
        self._num_collisions = num_collisions
        if animate:
            fig, axes = self.setup_figure()
        for _ in range(num_collisions):
            self.next_collision()
            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()

    def num_collisions(self):
        """
        Return the total number of collisions executed.

        Returns:
            int: Number of collisions.
        """
        return self._num_collisions


class SingleBallSimulation(Simulation):
    """
    A simulation with a single ball bouncing inside a circular container.
    """

    def __init__(self, container, ball):
        """
        Initialise the simulation with one container and one ball.

        Args:
            container (Container): The circular container.
            ball (Ball): The ball inside the container.
        """
        self._ball = ball
        self._container = container

    def container(self):
        """Return the container object."""
        return self._container

    def ball(self):
        """Return the ball object."""
        return self._ball

    def setup_figure(self):
        """
        Set up the figure for animation.

        Returns:
            tuple: (Figure, Axes) for plotting.
        """
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        ax.add_patch(self.ball().patch())
        return fig, ax

    def next_collision(self):
        """Advance the simulation by one collision between the ball and the container."""
        time = self._container.time_to_collision(self._ball)
        self._ball.move(time)
        self._container.collide(self._ball)


class MultiBallSimulation(Simulation):
    """
    A simulation of multiple balls colliding with each other and with a container.
    """

    def __init__(self, c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8.,
                nrings=3, multi=6):
        """
        Initialise the simulation with multiple balls arranged in concentric rings.

        Args:
            c_radius (float): Radius of the container.
            b_radius (float): Radius of each ball.
            b_speed (float): Initial speed of each ball.
            b_mass (float): Mass of each ball.
            rmax (float): Maximum radius for ring generation.
            nrings (int): Number of rings.
            multi (int): Number of balls per ring level.
        """
        self._c_radius = float(c_radius)
        self._b_radius = float(b_radius)
        self._b_speed = float(b_speed)
        self._b_mass = float(b_mass)
        self._rmax = float(rmax)
        self._nrings = int(nrings)
        self._multi = int(multi)
        self._container = Container(radius=c_radius)
        self._balls = []
        self._time_total = 0.0

        for pos in self.rtrings():
            angle = np.random.uniform(0, 2 * np.pi)
            vel = b_speed * np.array([np.cos(angle), np.sin(angle)])
            ball = Ball(pos=pos, vel=vel, radius=b_radius, mass=b_mass)
            self._balls.append(ball)

    def container(self):
        """Return the container object."""
        return self._container

    def balls(self):
        """Return the list of ball objects."""
        return self._balls

    def rtrings(self):
        """
        Generate positions for balls in concentric rings.

        Yields:
            tuple: (x, y) coordinates for ball positions.
        """
        yield (0.0, 0.0)
        if self._multi != 0:
            for i in range(1, self._nrings + 1):
                r = i * (self._rmax / self._nrings)
                n_points = i * self._multi
                sep = 2 * np.pi / n_points
                for j in range(n_points):
                    theta = j * sep
                    yield (r * np.cos(theta), r * np.sin(theta))

    def setup_figure(self):
        """
        Set up the figure for animation.

        Returns:
            tuple: (Figure, Axes) for plotting.
        """
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        for ball in self._balls:
            ax.add_patch(ball.patch())

        return fig, ax

    def next_collision(self):
        """
        Advance the simulation by one collision, updating positions, velocities, and time.
        """
        time = float(np.inf)
        object_colliding = []
        for i, ball_i in enumerate(self._balls):
            for j, ball_j in enumerate(self._balls[i + 1:], start=i + 1):
                t = ball_i.time_to_collision(ball_j)
                if t is not None:
                    if t < time:
                        time = t
                        object_colliding = [(i, j)]
                    elif t == time:
                        object_colliding.append((i, j))

            t = self.container().time_to_collision(ball_i)
            if t is not None:
                if t < time:
                    time = t
                    object_colliding = [(i,)]
                elif t == time:
                    object_colliding.append((i,))

        for ball in self._balls:
            ball.move(time)

        self._container.move(time)
        self._time_total += time

        for col in object_colliding:
            if len(col) == 2:
                self._balls[col[0]].collide(self._balls[col[1]])
            else:
                self._container.collide(self._balls[col[0]])

    def kinetic_energy(self):
        """
        Return the total kinetic energy of the system.

        Returns:
            float: Total kinetic energy.
        """
        ke_balls_total = 0
        for ball in self._balls:
            ke_balls_total += 0.5 * ball.mass() * np.dot(ball.vel(), ball.vel())

        ke_container = 0.5 * self._container.mass() * np.dot(self._container.vel(),
                                                             self._container.vel())
        return ke_balls_total + ke_container

    def momentum(self):
        """
        Return the total momentum vector of the system.

        Returns:
            ndarray: Total momentum as a 2-element array.
        """
        p_balls_total = np.zeros(2)
        for ball in self._balls:
            p_balls_total += ball.mass() * ball.vel()

        return p_balls_total + self._container.mass() * self._container.vel()

    def time(self):
        """Return the total simulation time elapsed."""
        return self._time_total

    def pressure(self):
        """
        Compute the average pressure on the container wall.

        Returns:
            float: Pressure in Pascals.
        """
        if self._time_total == 0:
            return 0.0
        else:
            return self._container.dp_tot() / (self.time() * 2 * np.pi * self._container.radius())

    def t_equipartition(self):
        """
        Compute the temperature using the equipartition theorem.

        Returns:
            float: Temperature in Kelvin.
        """
        avg_v2 = np.mean([np.dot(ball.vel(), ball.vel()) for ball in self._balls])
        return avg_v2 * self._b_mass / (2 * 1.380649e-23)

    def t_ideal(self):
        """
        Compute the temperature using the ideal gas law.

        Returns:
            float: Ideal temperature in Kelvin.
        """
        return self.pressure() * self._container.volume() / (len(self._balls) * 1.380649e-23)

    def speeds(self):
        """
        Return a list of scalar speeds of all balls.

        Returns:
            list[float]: Speeds of all balls.
        """
        return [np.linalg.norm(ball.vel()) for ball in self._balls]

    def t_vdw(self):
        """
        Compute the temperature using the Van der Waals equation.

        Returns:
            float: Van der Waals temperature.
        """
        return self.pressure() * (self.container().volume() - 2 * len(self._balls) *
                                  np.pi * self._balls[0].radius() *
                                  self._balls[0].radius()) / (len(self._balls) * 1.380649e-23)


class BrownianSimulation(MultiBallSimulation):
    """
    A simulation including a larger 'big ball' to model Brownian motion.
    """

    def __init__(self, c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8.,
                 nrings=3, multi=6, bb_radius=2., bb_mass=10.):
        """
        Initialise a Brownian motion simulation with one large ball among smaller ones.

        Args:
            c_radius, b_radius, b_speed, b_mass, rmax, nrings, multi: parameters for small balls
            bb_radius (float): Radius of big ball.
            bb_mass (float): Mass of big ball.
        """
        super().__init__(c_radius=c_radius, b_radius=b_radius, b_speed=b_speed,
                         b_mass=b_mass, rmax=rmax, nrings=nrings, multi=multi)
        self._balls.pop(0)  # remove central small ball
        self._bb = Ball(pos=[0., 0.], vel=[0., 0.], radius=bb_radius, mass=bb_mass)
        self._balls.append(self._bb)
        self._bb_positions = [self._bb.pos().copy()]
        self._trace_line = None

    def bb_positions(self):
        """
        Return list of big ball positions over time.

        Returns:
            list[np.ndarray]: Positions of the big ball.
        """
        return self._bb_positions

    def setup_figure(self):
        """
        Set up figure with big ball highlighted and trajectory plotted.

        Returns:
            tuple: (Figure, Axes)
        """
        fig, ax = super().setup_figure()
        patch = self._bb.patch()
        patch.set_facecolor('blue')
        patch.set_alpha(0.6)
        ax.add_patch(patch)
        self._trace_line, = ax.plot([], [], '-', lw=1, label='Big ball path', color='blue')
        return fig, ax

    def next_collision(self):
        """
        Advance simulation by one collision and update big ball's path.
        """
        super().next_collision()
        self._bb_positions.append(self._bb.pos().copy())

        if self._trace_line is not None:
            traj = np.array(self._bb_positions)
            self._trace_line.set_data(traj[:, 0], traj[:, 1])


