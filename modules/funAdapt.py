import numpy as np
from numba import njit, prange

@njit(nogil=True, parallel=False)
def hist2d_numba_seq(tracks, bins, ranges):
    H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1

    dx = (ranges[0, 1] - ranges[0, 0]) / bins[0]
    dy = (ranges[1, 1] - ranges[1, 0]) / bins[1]

    return H/np.sum(H*dx*dy)

@njit
def numba_log_zero_arr(array):
    N = array.shape[0]
    M = array.shape[1]
    out = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if array[i, j] > 0:
                out[i, j] = np.log(array[i, j])

    return out

@njit
def numba_log_zero_val(val):
    if val > 0:
        return np.log(val)
    else:
        return 0

@njit
def find_bin(val, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= val < bins[i + 1]:
            return i

    return -1

@njit
def simulate_xyeta(Nsteps, dt, sigma, a, theta_eta, tau_x = 1, theta_y = 1, x0 = 0, y0 = 0, eta0 = 0):
    """
    Simulates the Ornstein-Uhlenbeck process for x, y and eta.

    Parameters
    ----------
    Nsteps : int
        Number of steps to simulate.
    dt : float
        Time step.
    sigma : float
        Coupling between x -> eta.
    a : float
        Coupling between y -> x.
    theta_eta : float
        Time scale of eta.
    tau_x : float
        Time scale of x.
    theta_y : float
        Time scale of y.
    x0 : float
        Initial value of x.
    y0 : float
        Initial value of y.
    eta0 : float
        Initial value of eta.

    Returns
    -------
    x : numpy.ndarray
        Array of x values.
    y : numpy.ndarray
        Array of y values.
    eta : numpy.ndarray
        Array of eta values.
    """
    x = np.zeros(Nsteps, dtype=np.float64)
    y = np.zeros(Nsteps, dtype=np.float64)
    eta = np.zeros(Nsteps, dtype=np.float64)

    tau_y = theta_y * tau_x
    tau_eta = theta_eta * tau_x
    sqtau_x = np.sqrt(2*dt/tau_x)
    sqtau_y = np.sqrt(2*dt/tau_y)
    sqtau_eta = np.sqrt(2*dt/tau_eta)

    x[0] = x0
    y[0] = y0
    eta[0] = eta0


    for t in range(Nsteps-1):
        y[t + 1] = y[t] + dt * (-y[t] + a * x[t])/tau_y + np.random.randn() * sqtau_y
        x[t + 1] = x[t] + dt * (-x[t] + sigma * eta[t])/tau_x + np.random.randn() * sqtau_x
        eta[t + 1] = eta[t] + dt * (-eta[t])/tau_eta + np.random.randn() * sqtau_eta
        
    return x, y, eta



@njit
def cov_matrix(sigma, theta_eta, a):
    """
    Covariance matrix of the joint distribution of x and y.

    Parameters
    ----------
    sigma : float
        Coupling between x -> eta.
    theta_eta : float
        Time scale of eta.
    a : float
        Coupling between y -> x.

    Returns
    -------
    cov : numpy.ndarray
        Covariance matrix.
    """
    theta = 1 + theta_eta
    offdiag = 1/2*a*(1 + theta_eta*sigma**2*(1+2*theta_eta)/theta**2)
    return np.array([[1 + theta_eta*sigma**2/theta,
                      offdiag],
                     [offdiag,
                      1 + a*offdiag]])

@njit
def probability_xy(x, y, det, cov_inv):
    """
    Probability density function of the joint distribution of x and y.

    Parameters
    ----------
    x : float or array_like
        Value of x.
    y : float or array_like
        Value of y.
    det : float
        Determinant of the covariance matrix.
    cov_inv : numpy.ndarray
        Inverse of the covariance matrix.

    Returns
    -------
    pjoint : float or array_like
        Probability density.
    """
    return np.exp(-0.5 * (x**2 * cov_inv[0,0] + y**2 * cov_inv[1,1] + 2*x*y*cov_inv[0,1])) / (2*np.pi*np.sqrt(det))

@njit
def probability_x(x, cov):
    """
    Marginal probability density function of x.

    Parameters
    ----------
    x : float or array_like
        Value of x.
    cov : numpy.ndarray
        Covariance matrix.

    Returns
    -------
    px : float or array_like
        Probability density.
    """
    return np.exp(-0.5 * (x**2 / cov[0,0])) / (np.sqrt(2*np.pi*cov[0,0]))

@njit
def probability_y(y, cov):
    """
    Marginal probability density function of y.

    Parameters
    ----------
    y : float or array_like
        Value of y.
    cov : numpy.ndarray
        Covariance matrix.
    
    Returns
    -------
    py : float or array_like
        Probability density.
    """
    return np.exp(-0.5 * (y**2 / cov[1,1])) / (np.sqrt(2*np.pi*cov[1,1]))

@njit
def find_mutual_traj(Nsteps, dt, sigma, a, theta_eta, tau_x = 1, theta_y = 1, Nburn = 1000000):
    """
    Finds the mutual information for the marginalized xy process from a simulated
    trajectory of x and y.

    Parameters
    ----------
    Nsteps : int
        Number of steps to simulate.
    dt : float
        Time step.
    sigma : float
        Coupling between x -> eta.
    a : float
        Coupling between y -> x.
    theta_eta : float
        Time scale of eta.
    tau_x : float
        Time scale of x.
    theta_y : float
        Time scale of y.
    Nburn : int
        Number of discarded steps to reach stationarity.

    Returns
    -------
    mutual_trajectory : numpy.ndarray
        Array of mutual information values at each timepoint.
    """
    x0, y0, eta0 = simulate_xyeta(Nburn, dt, sigma, a, theta_eta, tau_x, theta_y)
    x, y, _ = simulate_xyeta(Nsteps, dt, sigma, a, theta_eta, tau_x, theta_y, x0[-1], y0[-1], eta0[-1])
    x0 = None
    y0 = None

    cov = cov_matrix(sigma, theta_eta, a)
    det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)


    mutual_trajectory = np.zeros(Nsteps, dtype=np.float64)

    for t in range(Nsteps):
        pjoint_curr = probability_xy(x[t], y[t], det, cov_inv)
        px_curr = probability_x(x[t], cov)
        py_curr = probability_y(y[t], cov)
        mutual_trajectory[t] = np.log(pjoint_curr / (px_curr * py_curr))

    return mutual_trajectory, x, y

@njit
def find_Axy(sigma, theta_eta, a):
    """
    Finds the interation matrix for the marginalized xy process.

    Parameters
    ----------
    sigma : float
        Coupling between x -> eta.
    theta_eta : float
        Time scale of eta.
    a : float
        Coupling between y -> x.

    Returns
    -------
    Amat : numpy.ndarray
        Interaction matrix.
    """
    theta = 1 + theta_eta

    den = (4 + a**2)*theta**4 + 2*(2+a**2)*theta_eta*theta**3*sigma**2
    den += a**2*theta_eta**2*(theta + theta_eta)*sigma**4

    num1 = 4*theta**4 + a**2*(theta**4 + 2*(theta_eta + theta_eta**2)**2*sigma**2 - theta_eta**2*(theta + theta_eta)*sigma**4)
    num2 = 2*a*theta_eta*theta*sigma**2*(-1+theta_eta**2-theta_eta*sigma**2)

    return np.array([[num1/den, - num2/den],
                     [- a, 1]])

@njit
def find_Sxy_traj(Nsteps, dt, sigma, a, theta_eta, tau_x = 1, theta_y = 1, Nburn = 1000000):
    """
    Finds the dissipation rate of the marginalized xy process from a simulated trajectory.

    Parameters
    ----------
    Nsteps : int
        Number of steps to simulate.
    dt : float
        Time step.
    sigma : float
        Coupling between x -> eta.
    a : float
        Coupling between y -> x.
    theta_eta : float
        Time scale of eta.
    tau_x : float
        Time scale of x.
    theta_y : float
        Time scale of y.
    Nburn : int
        Number of discarded steps to reach stationarity.

    Returns
    -------
    Sx_traj : numpy.ndarray
        Array of dissipation rate values for x at each timepoint.
    Sy_traj : numpy.ndarray
        Array of dissipation rate values for y at each timepoint.
    """
    x0, y0, eta0 = simulate_xyeta(Nburn, dt, sigma, a, theta_eta, tau_x, theta_y)
    x, y, _ = simulate_xyeta(Nsteps, dt, sigma, a, theta_eta, tau_x, theta_y, x0[-1], y0[-1], eta0[-1])
    x0 = None
    y0 = None

    Amat = find_Axy(sigma, theta_eta, a)

    Sx_traj = np.zeros(Nsteps - 1, dtype = np.float64)
    Sy_traj = np.zeros(Nsteps - 1, dtype = np.float64)

    for t in range(Nsteps-1):
        Deltax = x[t+1] - x[t]
        Deltay = y[t+1] - y[t]
        Stratx = 1/2*(x[t+1] + x[t])
        Straty = 1/2*(y[t+1] + y[t])
        Sx_traj[t] = (-Amat[0,0]*Stratx - Amat[0,1]*Straty)*Deltax/dt
        Sy_traj[t] = (-Amat[1,0]*Stratx - Amat[1,1]*Straty)*Deltay/dt

    return Sx_traj, Sy_traj

@njit
def find_functional(Nsteps, dt, sigma, a, theta_eta, Lambda, x0, y0, eta0, tau_x = 1, theta_y = 1):
    """
    Finds the Pareto functional from a simualted trajectory of x and y.

    Parameters
    ----------
    Nsteps : int
        Number of steps to simulate.
    dt : float
        Time step.
    sigma : float
        Coupling between x -> eta.
    a : float
        Coupling between y -> x.
    theta_eta : float
        Time scale of eta.
    Lambda : float
        Tradeoff parameter.
    x0 : float
        Initial value of x.
    y0 : float
        Initial value of y.
    eta0 : float
        Initial value of eta.
    tau_x : float
        Time scale of x.
    theta_y : float
        Time scale of y.

    Returns
    -------
    functional : float
        Pareto functional.
    x0 : float
        Final value of x.
    y0 : float
        Final value of y.
    eta0 : float
        Final value of eta.
    Sxy : float
        Dissipation rate of the marginalized xy process.
    Ixy : float
        Mutual information of the marginalized xy process.
    """
    x, y, eta = simulate_xyeta(Nsteps, dt, sigma, a, theta_eta, tau_x, theta_y, x0, y0, eta0)

    Amat = find_Axy(sigma, theta_eta, a)
    cov = cov_matrix(sigma, theta_eta, a)
    det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    Sxy = 0.
    Ixy = 0.

    for t in range(Nsteps-1):
        Delta_x = x[t+1] - x[t]
        Delta_y = y[t+1] - y[t]
        Strat_x = 1/2*(x[t+1] + x[t])
        Strat_y = 1/2*(y[t+1] + y[t])

        Sxy += -(Amat[0,0]*Strat_x + Amat[0,1]*Strat_y)*Delta_x/dt
        Sxy += -(Amat[1,0]*Strat_x + Amat[1,1]*Strat_y)*Delta_y/dt

        pjoint_curr = probability_xy(x[t], y[t], det, cov_inv)
        px_curr = probability_x(x[t], cov)
        py_curr = probability_y(y[t], cov)
        Ixy += np.log(pjoint_curr / (px_curr * py_curr))

    return -(1 - Lambda)*Sxy/(Nsteps - 1) + Lambda*Ixy/(Nsteps - 1), x[-1], y[-1], eta[-1], Sxy/(Nsteps - 1), Ixy/(Nsteps - 1)

@njit
def find_functional_empirical_cov(Nsteps, dt, sigma, a, theta_eta,
                                  Lambda, x0, y0, eta0, tau_x = 1, theta_y = 1):
    """
    Finds the Pareto functional from a simualted trajectory of x and y.
    Instead of the theoretical distribution, it uses the empirical distribution
    estimated form the numerical covariance matrix.

    Parameters
    ----------
    Nsteps : int
        Number of steps to simulate.
    dt : float
        Time step.
    sigma : float
        Coupling between x -> eta.
    a : float
        Coupling between y -> x.
    theta_eta : float
        Time scale of eta.
    Lambda : float
        Tradeoff parameter.
    x0 : float
        Initial value of x.
    y0 : float
        Initial value of y.
    eta0 : float
        Initial value of eta.
    tau_x : float
        Time scale of x.
    theta_y : float
        Time scale of y.

    Returns
    -------
    functional : float
        Pareto functional.
    x0 : float
        Final value of x.
    y0 : float
        Final value of y.
    eta0 : float
        Final value of eta.
    Sxy : float
        Dissipation rate of the marginalized xy process.
    Ixy : float
        Mutual information of the marginalized xy process.
    """
    x, y, eta = simulate_xyeta(Nsteps, dt, sigma, a, theta_eta, tau_x, theta_y, x0, y0, eta0)

    Amat = find_Axy(sigma, theta_eta, a)
    cov = np.cov(x, y)
    det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    Sxy = 0.
    Ixy = 0.

    for t in range(Nsteps-1):
        Delta_x = x[t+1] - x[t]
        Delta_y = y[t+1] - y[t]
        Strat_x = 1/2*(x[t+1] + x[t])
        Strat_y = 1/2*(y[t+1] + y[t])

        Sxy += -(Amat[0,0]*Strat_x + Amat[0,1]*Strat_y)*Delta_x/dt
        Sxy += -(Amat[1,0]*Strat_x + Amat[1,1]*Strat_y)*Delta_y/dt

        pjoint_curr = probability_xy(x[t], y[t], det, cov_inv)
        px_curr = probability_x(x[t], cov)
        py_curr = probability_y(y[t], cov)
        Ixy += np.log(pjoint_curr / (px_curr * py_curr))

    return -(1 - Lambda)*Sxy/(Nsteps - 1) + Lambda*Ixy/(Nsteps - 1), x[-1], y[-1], eta[-1], Sxy/(Nsteps - 1), Ixy/(Nsteps - 1)

@njit
def find_functional_empirical(Nsteps, dt, sigma, a, theta_eta,
                              Lambda, x0, y0, eta0, tau_x = 1, theta_y = 1,
                              bins_num = 100):
    """
    Finds the Pareto functional from a simualted trajectory of x and y.
    Instead of the theoretical distribution, it uses the empirical distribution
    estimated form the trajectories.

    Parameters
    ----------
    Nsteps : int
        Number of steps to simulate.
    dt : float
        Time step.
    sigma : float
        Coupling between x -> eta.
    a : float
        Coupling between y -> x.
    theta_eta : float
        Time scale of eta.
    Lambda : float
        Tradeoff parameter.
    x0 : float
        Initial value of x.
    y0 : float
        Initial value of y.
    eta0 : float
        Initial value of eta.
    tau_x : float
        Time scale of x.
    theta_y : float
        Time scale of y.

    Returns
    -------
    functional : float
        Pareto functional.
    x0 : float
        Final value of x.
    y0 : float
        Final value of y.
    eta0 : float
        Final value of eta.
    Sxy : float
        Dissipation rate of the marginalized xy process.
    Ixy : float
        Mutual information of the marginalized xy process.
    """
    x, y, eta = simulate_xyeta(Nsteps, dt, sigma, a, theta_eta, tau_x, theta_y, x0, y0, eta0)

    Amat = find_Axy(sigma, theta_eta, a)

    bins = np.array((bins_num, bins_num))
    ranges = np.array(((np.min(x), np.max(x)), (np.min(y), np.max(y))))
    xbins_space = np.linspace(ranges[0, 0], ranges[0, 1], bins[0])
    ybins_space = np.linspace(ranges[1, 0], ranges[1, 1], bins[1])

    dx = (ranges[0, 1] - ranges[0, 0]) / bins[0]
    dy = (ranges[1, 1] - ranges[1, 0]) / bins[1]

    pxy_emp = hist2d_numba_seq(np.vstack((x,y)), bins=bins, ranges=ranges)

    py_emp = np.sum(pxy_emp*dx, axis = 0)
    px_emp = np.sum(pxy_emp*dy, axis = 1)

    Sxy = 0.
    Ixy = 0.

    for t in range(Nsteps-1):
        Delta_x = x[t+1] - x[t]
        Delta_y = y[t+1] - y[t]
        Strat_x = 1/2*(x[t+1] + x[t])
        Strat_y = 1/2*(y[t+1] + y[t])

        Sxy += -(Amat[0,0]*Strat_x + Amat[0,1]*Strat_y)*Delta_x/dt
        Sxy += -(Amat[1,0]*Strat_x + Amat[1,1]*Strat_y)*Delta_y/dt

        xbin = find_bin(x[t], xbins_space)
        ybin = find_bin(y[t], ybins_space)
        pjoint_curr = pxy_emp[xbin, ybin]
        px_curr = px_emp[xbin]
        py_curr = py_emp[ybin]

        if px_curr * py_curr > 0 and pjoint_curr > 0:
            Ixy += numba_log_zero_val(pjoint_curr) - numba_log_zero_val(px_curr * py_curr)

    return -(1 - Lambda)*Sxy/(Nsteps - 1) + Lambda*Ixy/(Nsteps - 1), x[-1], y[-1], eta[-1], Sxy/(Nsteps - 1), Ixy/(Nsteps - 1)

@njit
def adaptive_dynamics(Nsteps, dt, sigma, theta_eta, Lambda, delta_a,
                      Ncheck = 2000, Nadapt_min = 5000, Nadapt_max = 10000,
                      a_init = 0., tau_x = 1, theta_y = 1, Nburn = 100000,
                      empirical_cov = False, empirical = False):
    
    x0_burn, y0_burn, eta0_burn = simulate_xyeta(Nburn, dt, sigma, a_init, theta_eta, tau_x, theta_y)
    x0 = x0_burn[-1]
    y0 = y0_burn[-1]
    eta0 = eta0_burn[-1]

    x0_burn = None
    y0_burn = None
    eta0_burn = None

    L_adapt = np.zeros(Nadapt_max, dtype = np.float64)
    a_adapt = np.zeros(Nadapt_max, dtype = np.float64)
    Ixy_adapt = np.zeros(Nadapt_max, dtype = np.float64)
    Sxy_adapt = np.zeros(Nadapt_max, dtype = np.float64)
    L_adapt[0] = 0.
    a_adapt[0] = a_init

    stop_counter = 0
    stop_time = Nadapt_max

    for idx_adapt in range(1, Nadapt_max):
        a_bar = a_adapt[idx_adapt - 1] + delta_a * np.random.randn()
        if empirical_cov:
            L_bar, x0, y0, eta0, Sxy, Ixy = find_functional_empirical_cov(Nsteps, dt, sigma, a_bar, theta_eta, Lambda, x0, y0, eta0, tau_x, theta_y)
        elif empirical:
            L_bar, x0, y0, eta0, Sxy, Ixy = find_functional_empirical(Nsteps, dt, sigma, a_bar, theta_eta, Lambda, x0, y0, eta0, tau_x, theta_y)
        else:
            L_bar, x0, y0, eta0, Sxy, Ixy = find_functional(Nsteps, dt, sigma, a_bar, theta_eta, Lambda, x0, y0, eta0, tau_x, theta_y)

        Sxy_adapt[idx_adapt] = Sxy
        Ixy_adapt[idx_adapt] = Ixy        

        if L_bar > L_adapt[idx_adapt - 1]:
            a_adapt[idx_adapt] = a_bar
            L_adapt[idx_adapt] = L_bar

            stop_counter = 0
        else:
            a_adapt[idx_adapt] = a_adapt[idx_adapt - 1]
            L_adapt[idx_adapt] = L_adapt[idx_adapt - 1]

            stop_counter += 1

            if idx_adapt > Nadapt_min and stop_counter > Ncheck:
                stop_time = idx_adapt
                break

    return a_adapt, L_adapt, Ixy_adapt, Sxy_adapt, stop_time

@njit(parallel = True)
def repeat_adaptive_dynamics(Nrepeat, Nsteps, dt, sigma, theta_eta, Lambda, delta_a,
                             Ncheck = 2000, Nadapt_min = 5000, Nadapt_max = 10000,
                             a_init = 0., tau_x = 1, theta_y = 1, Nburn = 100000,
                             empirical_cov = False, empirical = False):
    a_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    L_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    Ixy_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    Sxy_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    stop_time_adapt = np.zeros(Nrepeat, dtype = np.int64)

    for idx_repeat in prange(Nrepeat):
        res = adaptive_dynamics(Nsteps, dt, sigma, theta_eta, Lambda, delta_a,
                                Ncheck, Nadapt_min, Nadapt_max,
                                a_init, tau_x, theta_y, Nburn,
                                empirical_cov = empirical_cov,
                                empirical = empirical)
        a_adapt[idx_repeat], L_adapt[idx_repeat], Ixy_adapt[idx_repeat], Sxy_adapt[idx_repeat], stop_time_adapt[idx_repeat] = res

    return a_adapt, L_adapt, Ixy_adapt, Sxy_adapt, stop_time_adapt


@njit
def functional_exact(sigma, a, theta_eta, Lambda):
    theta = 1 + theta_eta

    den = (4 + a**2)*theta**4 + 2*(2+a**2)*theta_eta*theta**3*sigma**2
    den += a**2*theta_eta**2*(theta + theta_eta)*sigma**4

    S = (4 + a**2)*theta**2 + theta_eta*(2 + a**2 + 2*(1 + a**2)*theta_eta)*sigma**2
    S *= (a + a*theta_eta*(2 + theta_eta + sigma**2))**2
    S /= 2*den*theta**2

    I = 1/2*np.log(2*theta*(theta + theta_eta*sigma**2)*((2 + a**2)*theta**2+a**2*theta_eta*(1 + 2*theta_eta)*sigma**2)/den)

    return -(1 - Lambda)*S + Lambda*I

@njit
def Sxyeta_exact(sigma, a, theta_eta):
    theta = 1 + theta_eta
    return a**2/2 + (2 + theta_eta*(2+a**2))/theta**2*sigma**2/2

@njit
def Sdota_exact(sigma, a, theta_eta):
    theta = 1 + theta_eta
    return a**2/2*(1 + theta_eta*(2 + sigma**2 + theta_eta))/theta**2

@njit
def Sxy_exact(sigma, a, theta_eta):
    theta = 1 + theta_eta

    den = (4 + a**2)*theta**4 + 2*(2+a**2)*theta_eta*theta**3*sigma**2
    den += a**2*theta_eta**2*(theta + theta_eta)*sigma**4

    S = (4 + a**2)*theta**2 + theta_eta*(2 + a**2 + 2*(1 + a**2)*theta_eta)*sigma**2
    S *= (a + a*theta_eta*(2 + theta_eta + sigma**2))**2
    S /= 2*den*theta**2

    return S


@njit 
def Ixy_exact(sigma, a, theta_eta):
    theta = 1 + theta_eta

    den = (4 + a**2)*theta**4 + 2*(2+a**2)*theta_eta*theta**3*sigma**2
    den += a**2*theta_eta**2*(theta + theta_eta)*sigma**4

    return 1/2*np.log(2*theta*(theta + theta_eta*sigma**2)*((2 + a**2)*theta**2+a**2*theta_eta*(1 + 2*theta_eta)*sigma**2)/den)

@njit
def Iyeta_exact(sigma, a, theta_eta):
    theta = 1 + theta_eta

    den = (2 + a**2)*theta**4 + a**2*theta_eta*(1 + 4*theta_eta + 5*theta_eta**2)*sigma**2
    num = (2 + a**2)*theta**4 + a**2*theta_eta*theta**2*(1 + 2*theta_eta)*sigma**2

    return 1/2*np.log(num/den)






#########################################

@njit
def find_functional_afluc(Nsteps, dt, sigma, amean, tau_a, sigma_a, theta_eta, Lambda, x0, y0, eta0, tau_x = 1, theta_y = 1):
    x, y, eta, a_fluc = simulate_xyeta_afluc(Nsteps, dt, sigma, amean, tau_a, sigma_a, theta_eta, tau_x, theta_y, x0, y0, eta0)

    amean_traj = np.mean(a_fluc)

    Amat = find_Axy(sigma, theta_eta, amean_traj)
    cov = cov_matrix(sigma, theta_eta, amean_traj)
    det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    Sxy = 0.
    Ixy = 0.

    for t in range(Nsteps-1):
        Delta_x = x[t+1] - x[t]
        Delta_y = y[t+1] - y[t]
        Strat_x = 1/2*(x[t+1] + x[t])
        Strat_y = 1/2*(y[t+1] + y[t])

        Sxy += -2*(Amat[0,0]*Strat_x + Amat[0,1]*Strat_y)*Delta_x/dt
        Sxy += -2*(Amat[1,0]*Strat_x + Amat[1,1]*Strat_y)*Delta_y/dt

        pjoint_curr = probability_xy(x[t], y[t], det, cov_inv)
        px_curr = probability_x(x[t], cov)
        py_curr = probability_y(y[t], cov)
        Ixy += np.log(pjoint_curr / (px_curr * py_curr))

    return -(1 - Lambda)*Sxy/(Nsteps - 1) + Lambda*Ixy/(Nsteps - 1), x[-1], y[-1], eta[-1], Sxy, Ixy, a_fluc


@njit
def simulate_xyeta_afluc(Nsteps, dt, sigma, amean, tau_a, sigma_a, theta_eta, tau_x = 1, theta_y = 1, x0 = 0, y0 = 0, eta0 = 0):
    x = np.zeros(Nsteps, dtype=np.float64)
    y = np.zeros(Nsteps, dtype=np.float64)
    eta = np.zeros(Nsteps, dtype=np.float64)
    a = np.zeros(Nsteps, dtype=np.float64)

    tau_y = theta_y * tau_x
    tau_eta = theta_eta * tau_x
    sqtau_x = np.sqrt(dt/tau_x)
    sqtau_y = np.sqrt(dt/tau_y)
    sqtau_eta = np.sqrt(dt/tau_eta)
    sqtau_a = np.sqrt(dt/tau_a)

    x[0] = x0
    y[0] = y0
    eta[0] = eta0
    a[0] = amean


    for t in range(Nsteps-1):
        y[t + 1] = y[t] + dt * (-y[t] + a[t] * x[t])/tau_y + np.random.randn() * sqtau_y
        x[t + 1] = x[t] + dt * (-x[t] + sigma * eta[t])/tau_x + np.random.randn() * sqtau_x
        eta[t + 1] = eta[t] + dt * (-eta[t])/tau_eta + np.random.randn() * sqtau_eta
        a[t + 1] = a[t] + dt * (-a[t] + amean)/tau_a + np.random.randn() * sqtau_a * sigma_a
        
    return x, y, eta, a
