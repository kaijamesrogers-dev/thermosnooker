"""Analysis Module."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
from thermosnooker.simulations import SingleBallSimulation, MultiBallSimulation, BrownianSimulation
from thermosnooker.balls import Ball
from thermosnooker.balls import Container
from thermosnooker.physics import maxwell


def task9():
    """
    Task 9.

    In this function, you should test your animation. To do this, create a container
    and ball as directed in the project brief. Create a SingleBallSimulation object from these
    and try running your animation. Ensure that this function returns the balls final position and
    velocity.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The balls final position and velocity
    """
    c = Container(radius=10.)
    b = Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
    sbs = SingleBallSimulation(container=c, ball=b)
    sbs.run(num_collisions=1, animate=True, pause_time=0.5)

    return b.pos(), b.vel()


def task10():
    """
    Task 10.

    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.

    Watch the resulting animation carefully and make sure you aren't 
    seeing errors like balls sticking together or escaping the container.
    """
    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
    mbs.run(num_collisions=1, animate=True, pause_time=0.001)


def task11():
    """
    Task 11.

    In this function we shall be quantitatively checking that the balls aren't escaping or sticking.
    To do this, create the two histograms as directed in the project script. Ensure that these two
    histogram figures are returned.

    Returns:
        tuple[Figure, Firgure]: The histograms (distance from centre, inter-ball spacing).
    """
    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
    mbs.run(num_collisions=1, animate=False, pause_time=0.001)
    balls = mbs.balls()
    container = mbs.container()

    center = container.pos()
    dist_from_center = []
    for b in balls:
        pos = b.pos()
        d = np.linalg.norm(pos - center)
        dist_from_center.append(d)

    dist_from_center = np.array(dist_from_center)

    between_ball = []
    n = len(balls)
    for i in range(n):
        for j in range(i + 1, n):
            pi = balls[i].pos()
            pj = balls[j].pos()
            dij = np.linalg.norm(pi - pj)
            between_ball.append(dij)

    between_ball = np.array(between_ball)

    fig1 = plt.figure()
    ax1 = fig1.subplots()
    ax1.hist(dist_from_center, bins=20, edgecolor='black')
    ax1.set_xlabel("Distance from container center")
    ax1.set_ylabel("Number of balls")
    ax1.set_title(
        f"ball distances from center (after {mbs.num_collisions()} collisions)")

    fig2 = plt.figure()
    ax2 = fig2.subplots()
    ax2.hist(between_ball, bins=30, edgecolor='black')
    ax2.set_xlabel("Pairwise distance between balls")
    ax2.set_ylabel("Number of ball pairs")
    ax2.set_title(
        f"Histogram of pairwise ball distances (after {mbs.num_collisions()} collisions)")

    return fig1, fig2


def task12():
    """
    Task 12.

    In this function we shall check that the fundamental quantities of energy and momentum are 
    conserved. Additionally we shall investigate the pressure evolution of the system. Ensure 
    that the 4 figures outlined in the project script are returned.

    Returns:
        tuple[Figure, Figure, Figure, Figure]: matplotlib Figures of the KE, momentum_x, momentum_y 
        ratios as well as pressure evolution.
    """
    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
    num_collision = 1
    ke_list = [mbs.kinetic_energy()]
    p_list = [mbs.momentum()]
    pressure_list = [mbs.pressure()]
    time_list = [mbs.time()]
    for _ in range(num_collision):
        mbs.next_collision()
        ke_list.append(mbs.kinetic_energy())
        p_list.append(mbs.momentum())
        pressure_list.append(mbs.pressure())
        time_list.append(mbs.time())
    ke_ratio = ke_list / ke_list[0]
    px_ratio = [p[0] for p in p_list]/([p[0] for p in p_list][0])
    py_ratio = [p[1] for p in p_list]/([p[1] for p in p_list][0])

    fig_ke, ax_ke = plt.subplots()
    ax_ke.plot(time_list, ke_ratio)
    ax_ke.set_xlabel("time")
    ax_ke.set_ylabel("kinetic energy ratio (KE(t)/KE(0))")
    ax_ke.set_ylim(0.95, 1.05)

    fig_px, ax_px = plt.subplots()
    ax_px.plot(time_list, px_ratio)
    ax_px.set_xlabel("time")
    ax_px.set_ylabel("momentum ratio in x direction")
    ax_px.set_ylim(0.95, 1.05)

    fig_py, ax_py = plt.subplots()
    ax_py.plot(time_list, py_ratio)
    ax_py.set_xlabel("time")
    ax_py.set_ylabel("momentum ratio in y direction")
    ax_py.set_ylim(0.95, 1.05)

    fig_pressure, ax_pressure = plt.subplots()
    ax_pressure.plot(time_list, np.array(pressure_list))
    ax_pressure.set_xlabel("time")
    ax_pressure.set_ylabel("pressure")

    return fig_ke, fig_px, fig_py, fig_pressure


def task13():
    """
    Task 13.

    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Create the 3 figures directed by the project script, namely:
    1) PT plot
    2) PV plot
    3) PN plot
    Ensure that this function returns the three matplotlib figures.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """

    temp = []
    pressure = []
    for i in np.arange(0.1, 300., 3.):
        mbs = MultiBallSimulation(
            c_radius=10., b_radius=1., b_speed=float(i), b_mass=1., rmax=8., nrings=3, multi=6)
        mbs.run(num_collisions=4, animate=False, pause_time=0.001)
        temp.append(mbs.t_equipartition())
        pressure.append(mbs.pressure())

    fig_pressure_temp, ax_pressure_temp = plt.subplots()
    ax_pressure_temp.plot(temp, len(mbs.balls()) * 1.380649e-23 /
                          mbs.container().volume() * np.array(temp),
                          label="IGL")
    ax_pressure_temp.plot(
        temp,
        pressure,
        label=f'V = {mbs.container().volume()} N = {len(mbs.balls())}')
    ax_pressure_temp.set_xlabel("Temperature")
    ax_pressure_temp.set_ylabel("Pressure")
    ax_pressure_temp.legend()

    volume = []
    pressure = []
    for i in np.arange(10, 20, 0.1):
        mbs = MultiBallSimulation(
            c_radius=float(i), b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
        mbs.run(num_collisions=4, animate=False, pause_time=0.001)
        volume.append(mbs.container().volume())
        pressure.append(mbs.pressure())

    fig_pressure_vol, ax_pressure_vol = plt.subplots()
    ax_pressure_vol.plot(volume, len(mbs.balls()) * 1.380649e-23 *
                         mbs.t_equipartition() / np.array(volume),
                         label="IGL")
    ax_pressure_vol.plot(
        volume,
        pressure,
        label=f'T = {mbs.t_equipartition()} N = {len(mbs.balls())}')
    ax_pressure_vol.set_xlabel("Volume")
    ax_pressure_vol.set_ylabel("Pressure")
    ax_pressure_vol.legend()

    number_balls = []
    pressure = []
    for i in np.arange(0, 12, 0.1):
        mbs = MultiBallSimulation(
            c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=1, multi=i)
        mbs.run(num_collisions=4, animate=False, pause_time=0.001)
        number_balls.append(len(mbs.balls()))
        pressure.append(mbs.pressure())

    fig_pressure_num, ax_pressure_num = plt.subplots()
    ax_pressure_num.plot(number_balls, 1.380649e-23 * mbs.t_equipartition() /
                         mbs.container().volume() * np.array(number_balls),
                         label="IGL")
    ax_pressure_num.plot(
        number_balls,
        pressure,
        label=(
            f'T = {mbs.t_equipartition()} '
            f'V = {mbs.container().volume()}'))
    ax_pressure_num.set_xlabel("Number of Particles")
    ax_pressure_num.set_ylabel("Pressure")
    ax_pressure_num.legend()

    return fig_pressure_temp, fig_pressure_vol, fig_pressure_num


def task14():
    """
    Task 14.

    In this function we shall be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio defined
    in the project brief.

    Returns:
        Figure: The temperature ratio figure.
    """
    ball_radius = np.arange(0.01, 1, 0.01)
    t_equipartition = []
    t_ideal = []
    for i in ball_radius:
        mbs = MultiBallSimulation(
            c_radius=10., b_radius=float(i), b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
        mbs.run(num_collisions=500, animate=False, pause_time=0.001)
        t_equipartition.append(mbs.t_equipartition())
        t_ideal.append(mbs.t_ideal())

    fig_tt, ax_tt = plt.subplots()
    ax_tt.plot(ball_radius, np.array(t_equipartition) /
               np.array(t_ideal), linestyle="None", marker=".")
    ax_tt.set_xlabel("Ball Radius")
    ax_tt.set_ylabel("equipartition tempurature / ideal tempurature")

    return fig_tt


def task15():
    """
    Task 15.

    In this function we shall plot a histogram to investigate how the speeds of the balls evolve
    from the initial value. We shall then compare this to the Maxwell-Boltzmann distribution.
    Ensure that this function returns the created histogram.

    Returns:
        Figure: The speed histogram.
    """
    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=4, multi=6)
    mbs.run(num_collisions=500, animate=False, pause_time=0.001)
    speed10 = mbs.speeds()
    kbt = 1.380649e-23 * mbs.t_equipartition()

    fig_speed, ax_speed = plt.subplots()
    ax_speed.hist(speed10, bins=20, density=True)
    ax_speed.plot(np.arange(0, 1.5 * max(float(x) for x in speed10), 0.1),
                  maxwell(np.arange(0, 1.5 * max(float(x) for x in speed10), 0.1), kbt, mass=1.))
    ax_speed.set_xlabel("speed")
    ax_speed.set_ylabel("pdf")

    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=20., b_mass=1., rmax=8., nrings=4, multi=6)
    mbs.run(num_collisions=500, animate=False, pause_time=0.001)
    speed20 = mbs.speeds()
    kbt = 1.380649e-23 * mbs.t_equipartition()

    ax_speed.hist(speed20, bins=20, density=True)
    ax_speed.plot(np.arange(0, 1.5 * max(float(x) for x in speed20), 0.1),
                  maxwell(np.arange(0, 1.5 * max(float(x) for x in speed20), 0.1), kbt, mass=1.))

    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=30., b_mass=1., rmax=8., nrings=4, multi=6)
    mbs.run(num_collisions=500, animate=False, pause_time=0.001)
    speed30 = mbs.speeds()
    kbt = 1.380649e-23 * mbs.t_equipartition()

    ax_speed.hist(speed30, bins=20, density=True)
    ax_speed.plot(np.arange(0, 1.5 * max(float(x) for x in speed30), 0.1),
                  maxwell(np.arange(0, 1.5 * max(float(x) for x in speed30), 0.1), kbt, mass=1.))

    return fig_speed


def task16():
    """
    Task 16.

    In this function we shall also be looking at the divergence of our simulation from
    the IGL. We shall quantify the ball radii dependence of this divergence by
    plotting the temperature ratio and volume fraction defined in the project brief.
    We shall fit this temperature ratio before plotting the VDW b parameters radii dependence.

    Returns:
        tuple[Figure, Figure]: The ratio figure and b parameter figure.
    """
    ball_radius = np.arange(0.01, 1, 0.01)
    t_equipartition = []
    t_vdw = []
    vol_fraction_list = []
    valid_radius = []
    for i in ball_radius:
        mbs = MultiBallSimulation(
            c_radius=10., b_radius=i, b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
        mbs.run(num_collisions=500, animate=False, pause_time=0.001)
        t_equip = mbs.t_equipartition()
        t_v = mbs.t_vdw()
        if np.isfinite(t_equip) and np.isfinite(t_v) and t_v != 0:
            t_equipartition.append(t_equip)
            t_vdw.append(t_v)
            valid_radius.append(i)
            vol_fraction = 1 / ((mbs.container().volume() - 2 * len(mbs.balls()) * np.pi *
                                mbs.balls()[0].radius() * mbs.balls()[0].radius()) /
                                mbs.container().volume())
            vol_fraction_list.append(vol_fraction)

    valid_radius = np.array(valid_radius)

    def fit_func(r, a, b):
        return a * r**2 + b
    popt, _ = spo.curve_fit(fit_func, valid_radius, np.array(
        t_equipartition) / np.array(t_vdw))
    fit_vals = fit_func(valid_radius, *popt)

    fig_16, ax_16 = plt.subplots()
    ax_16.plot(valid_radius, np.array(t_equipartition) /
               np.array(t_vdw), linestyle="None", marker=".")
    ax_16.plot(valid_radius, vol_fraction_list, label="Volume Fraction")
    ax_16.plot(valid_radius, fit_vals, label='Fit: a*r^2 + b')
    ax_16.set_xlabel("Ball Radius")
    ax_16.set_ylabel("equipartition tempurature / Van Der Waals tempurature")
    ax_16.legend()

    b_val = (mbs.container().volume() / len(mbs.balls())) * (1 - 1 / fit_vals)
    b_param_geom = 2 * np.pi * valid_radius * valid_radius

    fig_16_2, ax2 = plt.subplots()
    ax2.plot(valid_radius, b_val, label='b from fit')
    ax2.plot(valid_radius, b_param_geom, label='Geometric b (2πr²)')
    ax2.set_xlabel('Ball Radius')
    ax2.set_ylabel('b parameter')
    ax2.legend()

    return fig_16, fig_16_2


def task17():
    """
    Task 17.

    In this function we shall run a Brownian motion simulation and plot the resulting trajectory of 
    the 'big' ball.
    """

    bds = BrownianSimulation(c_radius=10., b_radius=1., b_speed=10.,
                             b_mass=1., rmax=8., nrings=2, multi=6, bb_radius=2., bb_mass=10.)
    fig, ax = bds.setup_figure()

    num_steps = 200
    for _ in range(num_steps):
        bds.next_collision()
        plt.pause(0.01)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Brownian Motion: Big Ball Trajectory')
    ax.legend()
    return fig


if __name__ == "__main__":

    # Run task 9 function
    # BALL_POS, BALL_VEL = task9()

    # Run task 10 function
    # task10()

    # Run task 11 function
    # FIG11_BALLCENTRE, FIG11_INTERBALL = task11()

    # Run task 12 function
    #FIG12_KE, FIG12_MOMX, FIG12_MOMY, FIG12_PT = task12()

    # Run task 13 function
    #FIG13_PT, FIG13_PV, FIG13_PN = task13()

    # Run task 14 function
    #FIG14 = task14()

    # Run task 15 function
    #FIG15 = task15()

    # Run task 16 function
    #FIG16_RATIO, FIG16_BPARAM = task16()

    # Run task 17 function
    #task17()

    plt.show()
