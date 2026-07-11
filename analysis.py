"""Analysis module for the thermosnooker simulation.

Runs demonstrations and produces the figures used in the accompanying report:
animation demos, sanity-check histograms, conservation checks, ideal-gas
comparisons, Maxwell-Boltzmann speed distributions, Van der Waals fitting,
and a Brownian motion trajectory.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
from thermosnooker.simulations import SingleBallSimulation, MultiBallSimulation, BrownianSimulation
from thermosnooker.balls import Ball
from thermosnooker.balls import Container
from thermosnooker.physics import maxwell

# Which demo to run when executing this file directly.
# Options: "single_ball", "multi_ball", "histograms", "conservation",
#          "ideal_gas", "temperature_ratio", "speeds", "van_der_waals",
#          "brownian"
RUN = "brownian"  # Change this to run different demos


def single_ball_demo():
    """
    Animate a single ball bouncing inside the container.

    Verifies collision detection and elastic response in the simplest case.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The ball's final position and velocity.
    """
    c = Container(radius=10.)
    b = Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
    sbs = SingleBallSimulation(container=c, ball=b)
    sbs.run(num_collisions=10, animate=True, pause_time=0.5)

    return b.pos(), b.vel()


def multi_ball_animation():
    """
    Animate a multi-ball gas simulation for 500 collisions.

    Used for visually checking that balls do not stick together or
    escape the container.
    """
    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
    mbs.run(num_collisions=500, animate=True, pause_time=0.001)


def plot_distance_histograms():
    """
    Quantitatively check that balls are not escaping or overlapping.

    Produces two histograms: distance of each ball from the container centre,
    and pairwise distance between all balls.

    Returns:
        tuple[Figure, Figure]: The histograms (distance from centre, inter-ball spacing).
    """
    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
    mbs.run(num_collisions=500, animate=False, pause_time=0.001)
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


def plot_conservation_checks():
    """
    Check conservation of kinetic energy and momentum, and plot pressure evolution.

    Plots the ratio of each quantity at time t to its initial value; a flat
    line at 1 confirms conservation.

    Returns:
        tuple[Figure, Figure, Figure, Figure]: matplotlib Figures of the KE, momentum_x, momentum_y
        ratios as well as pressure evolution.
    """
    mbs = MultiBallSimulation(
        c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6)
    num_collisions = 500
    ke_list = [mbs.kinetic_energy()]
    p_list = [mbs.momentum()]
    pressure_list = [mbs.pressure()]
    time_list = [mbs.time()]
    for _ in range(num_collisions):
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


def plot_ideal_gas_comparisons():
    """
    Compare simulated pressure with the ideal gas law.

    Produces three figures: pressure vs temperature (PT), pressure vs volume (PV),
    and pressure vs number of particles (PN), each alongside the IGL prediction.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 figures: (PT, PV, PN)
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
    temps_nb = []
    volumes_nb = []
    # iterate integer particle counts (avoid zero)
    for n in range(1, 12):
        mbs = MultiBallSimulation(
            c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=1, multi=n)
        mbs.run(num_collisions=4, animate=False, pause_time=0.001)
        number_balls.append(len(mbs.balls()))
        pressure.append(mbs.pressure())
        temps_nb.append(mbs.t_equipartition())
        volumes_nb.append(mbs.container().volume())

    number_balls = np.array(number_balls)
    pressure = np.array(pressure)
    temps_nb = np.array(temps_nb)
    volumes_nb = np.array(volumes_nb)

    # compute ideal gas pressure elementwise: P = N kB T / V
    ideal_pressure = 1.380649e-23 * temps_nb / volumes_nb * number_balls

    fig_pressure_num, ax_pressure_num = plt.subplots()
    ax_pressure_num.plot(number_balls, ideal_pressure, label="IGL")
    ax_pressure_num.plot(
        number_balls,
        pressure,
        label=(
            f'T={np.mean(temps_nb):.3e} '
            f'V={volumes_nb[0]:.3e}'
        ))
    ax_pressure_num.set_xlabel("Number of Particles")
    ax_pressure_num.set_ylabel("Pressure")
    ax_pressure_num.legend()

    return fig_pressure_temp, fig_pressure_vol, fig_pressure_num


def plot_temperature_ratio():
    """
    Quantify divergence from the ideal gas law as ball radius increases.

    Plots the ratio of equipartition temperature to ideal-gas temperature
    against ball radius, showing excluded-volume effects.

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


def plot_speed_distributions():
    """
    Compare equilibrium speed histograms with the Maxwell-Boltzmann distribution.

    Runs the simulation at three initial speeds (10, 20, 30) and overlays the
    theoretical Maxwell-Boltzmann curve on each histogram.

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


def plot_van_der_waals_fit():
    """
    Fit the temperature ratio to extract an effective Van der Waals b parameter.

    Plots the equipartition/Van der Waals temperature ratio against ball radius
    with a quadratic fit, then compares the fitted b parameter with the
    geometric estimate 2*pi*r^2.

    Returns:
        tuple[Figure, Figure]: The ratio figure and b parameter figure.
    """
    ball_radius = np.arange(0.01, 1, 0.05)
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

    fig_ratio, ax_ratio = plt.subplots()
    ax_ratio.plot(valid_radius, np.array(t_equipartition) /
                  np.array(t_vdw), linestyle="None", marker=".")
    ax_ratio.plot(valid_radius, vol_fraction_list, label="Volume Fraction")
    ax_ratio.plot(valid_radius, fit_vals, label='Fit: a*r^2 + b')
    ax_ratio.set_xlabel("Ball Radius")
    ax_ratio.set_ylabel("equipartition tempurature / Van Der Waals tempurature")
    ax_ratio.legend()

    b_val = (mbs.container().volume() / len(mbs.balls())) * (1 - 1 / fit_vals)
    b_param_geom = 2 * np.pi * valid_radius * valid_radius

    fig_bparam, ax_bparam = plt.subplots()
    ax_bparam.plot(valid_radius, b_val, label='b from fit')
    ax_bparam.plot(valid_radius, b_param_geom, label='Geometric b (2πr²)')
    ax_bparam.set_xlabel('Ball Radius')
    ax_bparam.set_ylabel('b parameter')
    ax_bparam.legend()

    return fig_ratio, fig_bparam


def plot_brownian_trajectory():
    """
    Run a Brownian motion simulation and animate the big ball's trajectory.

    Returns:
        Figure: The trajectory figure.
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

    if RUN == "single_ball":
        single_ball_demo()
    elif RUN == "multi_ball":
        multi_ball_animation()
    elif RUN == "histograms":
        plot_distance_histograms()
    elif RUN == "conservation":
        plot_conservation_checks()
    elif RUN == "ideal_gas":
        plot_ideal_gas_comparisons()
    elif RUN == "temperature_ratio":
        plot_temperature_ratio()
    elif RUN == "speeds":
        plot_speed_distributions()
    elif RUN == "van_der_waals":
        plot_van_der_waals_fit()
    elif RUN == "brownian":
        plot_brownian_trajectory()
    else:
        raise ValueError(f"Unknown RUN option: {RUN}")

    plt.show()