"""Module defining Ball and Container classes for 2D gas simulations."""

import numpy as np
from matplotlib.patches import Circle


class Ball:
    """
    Represents a 2D hard-sphere particle with elastic collisions.

    Attributes:
        _pos (np.ndarray): Position vector of the ball.
        _vel (np.ndarray): Velocity vector of the ball.
        _radius (float): Radius of the ball.
        _mass (float): Mass of the ball.
        _patch (Circle): Graphical representation of the ball.
    """

    def __init__(self, pos=None, vel=None, radius=1., mass=1.):
        """
        Initialize a Ball object.

        Args:
            pos (list): Initial position of the ball [x, y].
            vel (list): Initial velocity of the ball [vx, vy].
            radius (float): Radius of the ball.
            mass (float): Mass of the ball.

        Raises:
            ValueError: If pos or vel are not 2-dimensional.
        """
        if pos is None:
            pos = [0, 0]
        if vel is None:
            vel = [1., 0.]
        self._pos = np.array(pos, dtype=float)
        self._vel = np.array(vel, dtype=float)
        self._radius = float(radius)
        self._mass = float(mass)
        self._patch = Circle(tuple(self._pos), self._radius, fc='firebrick')
        if len(self._pos) != 2:
            raise ValueError("pos must be a 2-dimensional vector")
        if len(self._vel) != 2:
            raise ValueError("vel must be a 2-dimensional vector")

    def pos(self):
        """Returns the current position of the ball.

        Returns:
            np.ndarray: Position vector.
        """
        return self._pos

    def vel(self):
        """Returns the current velocity of the ball.

        Returns:
            np.ndarray: Velocity vector.
        """
        return self._vel

    def radius(self):
        """Returns the radius of the ball.

        Returns:
            float: Radius.
        """
        return self._radius

    def mass(self):
        """Returns the mass of the ball.

        Returns:
            float: Mass.
        """
        return self._mass

    def set_vel(self, vel):
        """
        Sets a new velocity for the ball.

        Args:
            vel (list or np.ndarray): New velocity vector.

        Returns:
            np.ndarray: Updated velocity vector.

        Raises:
            ValueError: If vel is not 2-dimensional.
        """
        self._vel = np.array(vel)
        if len(self._vel) != 2:
            raise ValueError("vel must be a 2-dimensional vector")
        return self._vel

    def move(self, dt):
        """
        Updates position based on velocity and time increment.

        Args:
            dt (float): Time step.

        Returns:
            np.ndarray: Updated position.
        """
        self._pos += self._vel * dt
        self._patch.center = tuple(self._pos)
        return self._pos

    def patch(self):
        """Returns the patch object for rendering.

        Returns:
            Circle: Matplotlib patch representing the ball.
        """
        return self._patch

    def volume(self):
        """Returns the 2D volume (area) of the ball.

        Returns:
            float: Area.
        """
        return np.pi * self._radius**2

    def surface_area(self):
        """Returns the perimeter (2D surface area) of the ball.

        Returns:
            float: Perimeter.
        """
        return 2 * np.pi * self._radius

    def time_to_collision(self, other):
        """
        Computes the time to collision with another ball or container.

        Args:
            other (Ball or Container): The other object to check collision with.

        Returns:
            float or None: Time to collision, or None if no valid collision.
        """
        if isinstance(self, Container):
            relative_pos = other.pos() - self.pos()
            relative_vel = other.vel() - self.vel()

            a = np.dot(relative_vel, relative_vel)
            b = 2 * np.dot(relative_pos, relative_vel)
            c = np.dot(relative_pos, relative_pos) - \
                (self.radius() - other.radius())**2

            discriminant = b*b - 4*a*c

            if discriminant < 0 or a == 0:
                return None

            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2 * a)
                t2 = (-b + sqrt_disc) / (2 * a)

                valid_times = [t for t in (t1, t2) if t > 1e-12]
                if valid_times:
                    return float(min(valid_times))
                else:
                    return None

        if isinstance(self, Ball) and isinstance(other, Ball):

            relative_pos = self.pos() - other.pos()
            relative_vel = self.vel() - other.vel()

            a = np.dot(relative_vel, relative_vel)
            b = 2 * np.dot(relative_pos, relative_vel)
            c = np.dot(relative_pos, relative_pos) - \
                (self.radius() + other.radius())**2

            discriminant = b*b - 4*a*c

            if discriminant < 0 or a == 0:
                return None

            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2 * a)
                t2 = (-b + sqrt_disc) / (2 * a)

                valid_times = [t for t in (t1, t2) if t > 1e-12]
                if valid_times:
                    return float(min(valid_times))
                else:
                    return None

    def collide(self, other):
        """
        Resolves elastic collision with another ball.

        Args:
            other (Ball): Another ball.
        """
        if isinstance(other, Ball):

            r_rel = self.pos() - other.pos()

            distance = np.linalg.norm(r_rel)
            n = r_rel / distance
            v1_n = np.dot(self.vel(), n)
            v2_n = np.dot(other.vel(), n)

            v1_n_new = (v1_n * (self._mass - other.mass()) + 2 *
                        other.mass() * v2_n) / (self._mass + other.mass())
            v2_n_new = (v2_n * (other.mass() - self._mass) + 2 *
                        self._mass * v1_n) / (self._mass + other.mass())

            delta_v1 = (v1_n_new - v1_n) * n
            delta_v2 = (v2_n_new - v2_n) * n

            self.set_vel(self._vel + delta_v1)
            other.set_vel(other.vel() + delta_v2)


class Container(Ball):
    """
    Represents a circular container that inherits from Ball.

    Attributes:
        _change_mom (float): Cumulative change in momentum from collisions.
    """
    change_mom = 0

    def __init__(self, radius=10., mass=1e7):
        """
        Initialize a Container.

        Args:
            radius (float): Radius of the container.
            mass (float): Mass of the container.
        """
        super().__init__(pos=[0, 0], vel=[0, 0], radius=radius, mass=mass)
        patch = self.patch()
        patch.set_facecolor('none')
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
        self._change_mom = 0.0

    def dp_tot(self):
        """Return total momentum change accumulated.

        Returns:
            float: Total change in momentum.
        """
        return self._change_mom

    def collide(self, other):
        """
        Resolves collision with another ball and accumulates momentum change.

        Args:
            other (Ball): The ball colliding with the container.
        """
        if isinstance(other, Ball):

            r_rel = self.pos() - other.pos()

            distance = np.linalg.norm(r_rel)
            n = r_rel / distance
            v1_n = np.dot(self.vel(), n)
            v2_n = np.dot(other.vel(), n)

            v1_n_new = (v1_n * (self._mass - other.mass()) + 2 *
                        other.mass() * v2_n) / (self._mass + other.mass())
            v2_n_new = (v2_n * (other.mass() - self._mass) + 2 *
                        self._mass * v1_n) / (self._mass + other.mass())

            delta_v1 = (v1_n_new - v1_n) * n
            delta_v2 = (v2_n_new - v2_n) * n

            other_old_vel = other.vel().copy()
            self.set_vel(self._vel + delta_v1)
            other.set_vel(other.vel() + delta_v2)

            self._change_mom += np.linalg.norm(
                (other.vel() - other_old_vel) * other.mass())
