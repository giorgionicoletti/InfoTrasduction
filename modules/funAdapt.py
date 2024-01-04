import numpy as np
from numba import njit, prange


@njit
def simulate_xyeta(Nsteps, dt, sigma, a, theta_eta, tau_x = 1, theta_y = 1, x0 = 0, y0 = 0, eta0 = 0):
    x = np.zeros(Nsteps, dtype=np.float64)
    y = np.zeros(Nsteps, dtype=np.float64)
    eta = np.zeros(Nsteps, dtype=np.float64)

    tau_y = theta_y * tau_x
    tau_eta = theta_eta * tau_x
    sqtau_x = np.sqrt(dt/tau_x)
    sqtau_y = np.sqrt(dt/tau_y)
    sqtau_eta = np.sqrt(dt/tau_eta)

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
    theta = 1 + theta_eta
    offdiag = 1/2*a*(1 + theta_eta*sigma**2*(1+2*theta_eta)/theta**2)
    return np.array([[1 + theta_eta*sigma**2/theta,
                      offdiag],
                     [offdiag,
                      1 + a*offdiag]])

@njit
def probability_xy(x, y, det, cov_inv):
    return np.exp(-0.5 * (x**2 * cov_inv[0,0] + y**2 * cov_inv[1,1] + 2*x*y*cov_inv[0,1])) / (2*np.pi*np.sqrt(det))

@njit
def probability_x(x, cov):
    return np.exp(-0.5 * (x**2 / cov[0,0])) / (np.sqrt(2*np.pi*cov[0,0]))

@njit
def probability_y(y, cov):
    return np.exp(-0.5 * (y**2 / cov[1,1])) / (np.sqrt(2*np.pi*cov[1,1]))

@njit
def find_mutual_traj(Nsteps, dt, sigma, a, theta_eta, tau_x = 1, theta_y = 1, Nburn = 1000000):
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
    theta = 1 + theta_eta

    den = (4 + a**2)*theta**4 + 2*(2+a**2)*theta_eta*theta**3*sigma**2
    den += a**2*theta_eta**2*(theta + theta_eta)*sigma**4

    num1 = 4*theta**4 + a**2*(theta**4 + 2*(theta_eta + theta_eta**2)**2*sigma**2 - theta_eta**2*(theta + theta_eta)*sigma**4)
    num2 = 2*a*theta_eta*theta*sigma**2*(-1+theta_eta**2-theta_eta*sigma**2)

    return np.array([[num1/den, - num2/den],
                     [- a, 1]])

@njit
def find_Sxy_traj(Nsteps, dt, sigma, a, theta_eta, tau_x = 1, theta_y = 1, Nburn = 1000000):
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

        Sxy += -2*(Amat[0,0]*Strat_x + Amat[0,1]*Strat_y)*Delta_x/dt
        Sxy += -2*(Amat[1,0]*Strat_x + Amat[1,1]*Strat_y)*Delta_y/dt

        pjoint_curr = probability_xy(x[t], y[t], det, cov_inv)
        px_curr = probability_x(x[t], cov)
        py_curr = probability_y(y[t], cov)
        Ixy += np.log(pjoint_curr / (px_curr * py_curr))

    return -(1 - Lambda)*Sxy/(Nsteps - 1) + Lambda*Ixy/(Nsteps - 1), x[-1], y[-1], eta[-1], Sxy, Ixy


@njit
def adaptive_dynamics(Nsteps, dt, sigma, theta_eta, Lambda, delta_a,
                      Ncheck = 2000, Nadapt_min = 5000, Nadapt_max = 10000,
                      a_init = 0., tau_x = 1, theta_y = 1, Nburn = 100000):
    
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
                             a_init = 0., tau_x = 1, theta_y = 1, Nburn = 100000):
    a_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    L_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    Ixy_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    Sxy_adapt = np.zeros((Nrepeat, Nadapt_max), dtype = np.float64)
    stop_time_adapt = np.zeros(Nrepeat, dtype = np.int64)

    for idx_repeat in prange(Nrepeat):
        res = adaptive_dynamics(Nsteps, dt, sigma, theta_eta, Lambda, delta_a,
                                Ncheck, Nadapt_min, Nadapt_max,
                                a_init, tau_x, theta_y, Nburn)
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
def Sxy_exact(sigma, a, theta_eta):
    theta = 1 + theta_eta
    return a**2/2 + (2 + theta_eta*(2+a**2))/theta**2*sigma**2/2
    

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
