import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from tqdm import tqdm


def obs_Overlap(i, j, Masking) :
    """
    Overlapped column indices both with value 1 
    """
    T = Masking.shape[1]
    overlap = []
    for t in range(T) : 
        if Masking[i, t] == 1 and Masking[j, t] == 1:
            overlap.append(t)
            
    return np.array((overlap))

def sqmmd_est2(dat1, dat2, kernel) :
    """
    Computes U-statistics estimate of squared MMD_k, when number of samples from each distribution are different

    Input
        dat 1 : (m * d) data matrix coming from first d dim distribution
        dat 2 : (n * d) data matrix coming from second d dim distribution
        kernel : k(x, y) that defines MMD_k^2
    
    Output
        MMD_k^2 estimator
    """
    m = dat1.shape[0]
    n = dat2.shape[0]

    if dat1.shape[1] != dat2.shape[1] : 
        print("Data dimension do not match!")
        return
    
    d = dat1.shape[1]

    XX = np.matmul(dat1, np.transpose(dat1)) # m by m matrix with x_i^Tx_j
    YY = np.matmul(dat2, np.transpose(dat2)) # n by n matrix with y_i^Ty_j
    XY = np.matmul(dat1, np.transpose(dat2)) # m by n matrix with x_i^Ty_j  

    if kernel == "linear" :
        kXX, kYY, kXY = XX, YY, XY
    if kernel == "square" :
        kXX, kYY, kXY = (XX + np.ones( (m, m) ))**2, (YY + np.ones( (n, n) ))**2, (XY + np.ones( (m, n) ))**2
    if kernel == "exponential" :
        dXX_mm = np.vstack((np.diag(XX), )*m)
        dYY_nn = np.vstack((np.diag(YY), )*n)
        dXX_mn = np.hstack((np.diag(XX), )*n)
        dYY_mn = np.vstack((np.diag(YY), )*m)

        kXX = math.exp( -0.5*( dXX_mm + np.transpose(dXX_mm) - 2*XX ) ) 
        kYY = math.exp( -0.5*( dYY_nn + np.transpose(dYY_nn) - 2*YY ) )
        kXY = math.exp( -0.5*( dXX_mn + dYY_mn - 2*XY ) )
        
    val = (kXX.sum() - np.diag(kXX).sum())/(m*(m - 1)) + (kYY.sum() - np.diag(kYY).sum())/(n*(n - 1)) - 2*kXY.sum()/(n*m)
    if val < 0 : 
        val = 0

    return(val)

def row_Metric(i, j, t, Data, Masking, kernel, exc_opt) : 
    """
    Computes MMD_k^2 based metric between rows of Distributional Matrix~(DM) with missingness

    Input
        i, j : two row indices under comparison
        t : column of interest - so need to exclude this column
        kernel : baseline kernel
        Data : (N * T * n * d) array 
        Masking : (N * T) sized matrix indicating which entries are observed(val = 1)
        exc_opt : if True, omit t th column when constructing row metric, if False, include t th column
    Output
        Metric between row i, j when the target parameter is on the t th column
    """

    N = Masking.shape[0]
    T = Masking.shape[1]
    
    overlap = obs_Overlap(i, j, Masking)

    if exc_opt == True : 
        if sum(np.isin(overlap, t)) == 1 : 
            overlap = np.delete(overlap, np.where(overlap == t)[0])
    if len(overlap) == 0 : 
        val = 10**5
        return(val)

    Data_i = Data[i, :, :, :]
    Data_j = Data[j, :, :, :]

    pre_val = np.zeros(len(overlap))
    for tau in range(len(overlap)) :
        tau_ind = overlap[tau]
        pre_val[tau] = sqmmd_est2(Data_i[tau_ind, :, :], Data_j[tau_ind, :, :], kernel)
    
    val = sum(pre_val)/len(overlap)

    return(val)

def row_mmDNN(i, t, Data, row_Dissim_vec, Masking, eta) : 
    """
    Implements DNN with MMD_k^2 to impute (i, t) entry using eta radius

    t th column of Data is used for averaging
    t th column of Masking is used to pick the ones observed ... (1)
    row_Dissim_vec, eta are used to pick the ones within neighborhood ... (2)
    when intersecting (1) and (2), make sure to exclude i th row and then take the barycenter
    
    note kernel information is already used when constructing row_Dissim_vec from another function 

    Input
        i, t : index of target distribution - mu_{i, t}
        row_Dissim_vec : row-wise metric (rho_{i, j}) for j in [N], excluding t th column for construction
        Data : (N * T * n * d) array
        Masking : (N * T) array
        kernel : baseline kernel
        eta : radius
    Output
        (|neighbor| * n * d) sized array of all the neighboring measurements
    """
    N, T, n, d = Data.shape[0], Data.shape[1], Data.shape[2], Data.shape[3]

    neighbor_candidate = Data[:, t, :, :]

    neighbor_ind = np.where( (row_Dissim_vec < eta)*(Masking[:, t]) == 1 )[0]

    if sum(np.isin(neighbor_ind, i)) == 1 : # Pretending as IF (i, t) entry is missing
        neighbor_ind = np.delete(neighbor_ind, np.where(neighbor_ind == i)[0])

    if len(neighbor_ind) == 0 :
        neighbor = np.zeros( (n, d) )
        return(neighbor)

    neighbor = neighbor_candidate[neighbor_ind, :, :]

    neighbor = neighbor.reshape(-1, neighbor.shape[-1]) # ((|neighbor_ind| x n) * d) array

    return(neighbor)

    # if len(neighbor_ind) == 1 :
    #     return(neighbor) # (n * d) array
    # if len(neighbor_ind) > 1 :
    #     neighbor = neighbor.reshape(-1, neighbor.shape[-1]) # ((|neighbor_ind| x n) * d) array
    #     return(neighbor)

def mmDNN_cv(Data, Masking, kernel, eta_cand) : 
    """
    Among eta_cand, chooses the optimal radius eta that minimizes 2-fold CV error

    Input 
        Data : Full data that is split into two - train & test
        Masking : Full Masking matrix that is split into two - train & test
        eta_cand : Candidate of radius that is explored

    Output
        Optimal radius eta^star
    """
    N, T, n, d = Data.shape[0], Data.shape[1], Data.shape[2], Data.shape[3]

    if (T % 2) == 0 : 
        T_1 = int(T/2)
    if (T % 2) == 1 :
        T_1 = int((T + 1)/2)

    Data1 = Data[:, np.arange(T_1), :, :]
    Masking1 = Masking[:, np.arange(T_1)]

    Data2 = Data[:, np.arange(T_1, T), :, :]
    Masking2 = Masking[:, np.arange(T_1, T)]


    # Construct rho_{i, j} for rows using Data 1 ... (1)
    row_Dissim_mat = np.zeros( (N, N) )

    t_0 = 1 # Arbitrary column index when constructing rho_{i, j} under CV process
    for i in range(N - 1) :
        for j in range((i + 1), N) :
            row_Dissim_mat[i, j] = row_Metric(i, j, t_0, Data1, Masking1, kernel, exc_opt = False)
    
    row_Dissim_mat = row_Dissim_mat + np.transpose(row_Dissim_mat)
    
    # Estimate `observed` entries in Data 2 using rho_{i, j} in (1) ... (2)
    T_2 = T - T_1 # Length of second partition of Data

    perf = np.zeros(len(eta_cand))
    for eta_ind in range( len(eta_cand) ) :
        eta = eta_cand[eta_ind]
        mmd_error = np.zeros( (N, T_2) )
        for i in range(N) : 
            for t in range(T_2) :
                if Masking2[i, t] == 1 :
                    hat_mu_it = row_mmDNN(i, t, Data2, row_Dissim_mat[i, :], Masking2, eta)
                    mmd_error_it = sqmmd_est2(hat_mu_it, Data2[i, t, :, :], kernel)
                    mmd_error[i, t] = mmd_error_it
        perf[eta_ind] = mmd_error.sum()/(Masking2.sum())

    eta_star = eta_cand[np.argmin(perf)]
    
    return(eta_star)

expit = lambda x : np.exp(x)/(1 + np.exp(x))

def row_snn(i, t, Data, row_Dissim_vec, Masking, eta) : 
    N, T, d = Data.shape[0], Data.shape[1], Data.shape[2]

    neighbor_candidate = Data[:, t, : ]

    neighbor_ind = np.where( (row_Dissim_vec < eta)*(Masking[:, t]) == 1 )[0]

    if sum(np.isin(neighbor_ind, i)) == 1 : # Pretending as IF (i, t) entry is missing
        neighbor_ind = np.delete(neighbor_ind, np.where(neighbor_ind == i)[0])

    if len(neighbor_ind) == 0 :
        neighbor = np.zeros( (d) )
        return(neighbor)

    neighbor = neighbor_candidate[neighbor_ind, :]

    neighbor = neighbor.reshape(-1, neighbor.shape[-1]) # ((|neighbor_ind| x n) * d) array

    return(neighbor)

def snn_row_metric(i, j, t, Data, Masking, exc_opt):
    N = Masking.shape[0]
    T = Masking.shape[1]
    
    overlap = obs_Overlap(i, j, Masking)

    if exc_opt == True : 
        if sum(np.isin(overlap, t)) == 1 : 
            overlap = np.delete(overlap, np.where(overlap == t)[0])
    if len(overlap) == 0 : 
        val = 10**5
        return(val)

    Data_i = Data[i, :, :]
    Data_j = Data[j, :, :]

    pre_val = np.zeros(len(overlap))
    for tau in range(len(overlap)) :
        tau_ind = overlap[tau]
        pre_val[tau] = np.linalg.norm((Data_i[tau_ind] - Data_j[tau_ind])**2)
    
    val = sum(pre_val)/len(overlap)

    return(val)

def snn_CV(Data, Masking, eta_cand):
    N, T, d = Data.shape[0], Data.shape[1], Data.shape[2]

    if (T % 2) == 0 : 
        T_1 = int(T/2)
    if (T % 2) == 1 :
        T_1 = int((T + 1)/2)

    Data1 = Data[:, np.arange(T_1), :]
    Masking1 = Masking[:, np.arange(T_1)]

    Data2 = Data[:, np.arange(T_1, T), :]
    Masking2 = Masking[:, np.arange(T_1, T)]


    # Construct rho_{i, j} for rows using Data 1 ... (1)
    row_Dissim_mat = np.zeros( (N, N) )

    t_0 = 1 # Arbitrary column index when constructing rho_{i, j} under CV process
    for i in range(N - 1) :
        for j in range((i + 1), N) :
            row_Dissim_mat[i, j] = snn_row_metric(i, j, t_0, Data1, Masking1, exc_opt = False)
    
    row_Dissim_mat = row_Dissim_mat + np.transpose(row_Dissim_mat)
    
    # Estimate `observed` entries in Data 2 using rho_{i, j} in (1) ... (2)
    T_2 = T - T_1 # Length of second partition of Data

    perf = np.zeros(len(eta_cand))
    for eta_ind in range( len(eta_cand) ) :
        eta = eta_cand[eta_ind]
        snn_error = np.zeros( (N, T_2) )
        for i in range(N) : 
            for t in range(T_2) :
                if Masking2[i, t] == 1 :
                    hat_mu_it = row_snn(i, t, Data2, row_Dissim_mat[i, :], Masking2, eta)
                    snn_error_it = np.linalg.norm((hat_mu_it - Data2[i, t, :])**2)
                    snn_error[i, t] = snn_error_it
        perf[eta_ind] = snn_error.sum()/(Masking2.sum())

    eta_star = eta_cand[np.argmin(perf)]
    
    return(eta_star)


def gendata_s_adopt(N, T, n, d, beta, seed) : 
    """ 
    Generates Gaussian data, with latent dimension r = 2

    beta : vector of two fractions between (0, 1)

    required : N, d are both EVEN positive integers
    """

    np.random.seed(seed = seed)

    Data = np.zeros( (N, T, n, d) )
    true_Mean = np.zeros( (N, T, d) )
    true_Cov = np.zeros( (N, T, d, d) )

    u_1 = np.random.uniform(-1, 1, N)
    u_2 = np.random.uniform(0.2, 1, N)

    v_1 = np.random.uniform(-2, 2, T)
    v_2 = np.random.uniform(0.5, 2, T)

    even_ones = np.repeat([0, 1], d/2)
    odd_ones = np.repeat([1, 0], d/2)

    for i in range(N) : 
        for t in range(T) : 
            m_it = u_1[i]*v_1[t]*(even_ones - odd_ones)
            c_it = np.diag(u_2[i]*v_2[t]*(0.5*even_ones + odd_ones))
            true_Mean[i, t, :] = m_it
            true_Cov[i, t, :, :] = c_it
            dat_mat = np.random.multivariate_normal(m_it, c_it, size = n)
            Data[i, t, :, :] = dat_mat

    Masking = np.zeros( (N, T) )
    pre_Masking = np.zeros( (N, T) )

    g1_inds = np.arange(0, N // 2)
    g2_inds = np.arange(N // 2, 3 * N // 4)
    g3_inds = np.arange(3 * N // 4, N)

    gamma_1 = [2, 0.7, 1, 0.7]
    gamma_2 = [2.5, 0.2, 1, 0.2]

    T1_lower = math.floor(T**beta[0])
    T2_lower = math.floor(T**beta[1])

    for i in range(N) : 
        if i in g1_inds :
            pre_Masking[i, :] = np.concatenate((np.ones(T1_lower), np.zeros(T - T1_lower)))
            for t in range(T - T1_lower) :
                pre_Masking[i, (t + T1_lower)] = np.random.binomial(1, expit(gamma_1[0] + ( 0.99**t )*gamma_1[1]*u_1[i-1] + gamma_1[2]*u_1[i] + ( 0.99**t )*gamma_1[3]*u_1[i+1]), 1)
            pre_A = pre_Masking[i, :]
            if len([i for i in range(len(pre_A)) if pre_A[i] == 0]) == 0:
                Masking[i, :] = pre_A
            elif len([i for i in range(len(pre_A)) if pre_A[i] == 0]) > 0:
                adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0]) 
                Masking[i, :] = np.concatenate((np.ones(adopt_time), np.zeros(T - adopt_time)))
        elif i in g2_inds :
            pre_Masking[i, :] = np.concatenate((np.ones(T2_lower), np.zeros(T - T2_lower)))
            for t in range(T - T2_lower) :
                pre_Masking[i, (t + T2_lower)] = np.random.binomial(1, expit(gamma_2[0] + ( 1.01**t )*gamma_2[1]*u_1[i-1] + gamma_2[2]*u_1[i] + ( 1.01**t )*gamma_2[3]*u_1[i+1]), 1)
            pre_A = pre_Masking[i, :]
            if len([i for i in range(len(pre_A)) if pre_A[i] == 0]) == 0:
                Masking[i, :] = pre_A
            elif len([i for i in range(len(pre_A)) if pre_A[i] == 0]) > 0:
                adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0]) 
                Masking[i, :] = np.concatenate((np.ones(adopt_time), np.zeros(T - adopt_time)))
            # adopt_time = min([i for i in range(len(pre_A)) if pre_A[i] == 0]) 
            # Masking[i, :] = np.concatenate((np.ones(adopt_time), np.zeros(T - adopt_time)))
        elif i in g3_inds : 
            Masking[i, :] = np.ones( T )

    return(Data, Masking, pre_Masking, true_Mean, true_Cov)



def main():
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 20
        mpl.rcParams['xtick.labelsize'] = label_size 
        mpl.rcParams['ytick.labelsize'] = label_size 
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 2.5
        mpl.rcParams['legend.fontsize'] = label_size
        plt.rcParams['xtick.major.pad']=5
        plt.rcParams['ytick.major.pad']=5

        lss = ['--',  ':', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':', '-']
        mss = ['>', 'o',  's', 'D', '>', 's', 'o', 'D', '>', 's', 'o', 'D']
        ms_size = [25, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        colors = ['#e41a1c', '#0000cd', '#4daf4a',  'black' , 'magenta']
    else:
        pass
    #####################################
    # Simulation for Staggered Adoption #
    #####################################

    T, n, d = 80, 30, 2
    beta = [3.9/6, 3.9/5]
    kernel = "square"
    #eta_pool = np.arange(1, 30, 0.5)/3 # used for MNAR + non-positive
    eta_pool = np.concatenate( ( np.arange(1, 25, 0.5)/6, np.arange(22, 50, 1.5)/5 ) ) # used for MCAR
    eta_pool_snn = np.arange(1, 50, 1)
    i, t = 0, T - 1

    nsim = 30

    pools = []
    pools_eta = []
    pools_eta_snn = []
    pools_snn_ests = []
    pools_mmd_mean_ests = []
    pools_realmean_ests = []
    # print(f"{0}-th iteration")
    for i, row_exp in enumerate(np.arange(5, 9)): 
        print(f"{i}-th iteration")
        N = 2**(row_exp)
        perf_pool, eta_star_pool = np.zeros( nsim ), np.zeros( nsim )
        perf_snn_pool, eta_star_snn_pool = np.zeros( nsim ), np.zeros( nsim )
        snn_ests = np.zeros([nsim, d])
        mmd_mean_ests = np.zeros([nsim, d])
        real_mean = np.zeros([nsim, d])

        for sim in tqdm(range(nsim)) :
            Data, Masking, pre_Masking, true_Mean, true_Cov = gendata_s_adopt(N, T, n, d, beta, seed = sim)
            true_Mean_it = true_Mean[i, t, :]
            true_Cov_it = true_Cov[i, t, :, :]
            Data_mean = np.mean(Data, axis = 2)
            row_Dissim_vec = np.zeros(N)
            snn_row_Dissim_vec = np.zeros(N)

            for j in range(N) :
                row_Dissim_vec[j] = row_Metric(i, j, t, Data, Masking, kernel, exc_opt = True)
                snn_row_Dissim_vec[j] = snn_row_metric(i, j, t, Data_mean, Masking, exc_opt=True)
            eta_star_snn = snn_CV(Data_mean, Masking, eta_cand = eta_pool_snn)
            eta_star = mmDNN_cv(Data, Masking, kernel, eta_cand = eta_pool)
            eta_star_pool[sim] = eta_star
            eta_star_snn_pool[sim] = eta_star_snn

            hat_mu_it = row_mmDNN(i, t, Data, row_Dissim_vec, Masking, eta = eta_star)
            neighbors_it = row_snn(i, t, Data_mean, snn_row_Dissim_vec, Masking, eta = eta_star_snn)
            mean_it = np.mean(neighbors_it, axis = 0)
            mmd_mean_ests[sim] = np.mean(hat_mu_it, axis = 0)
            snn_ests[sim] = mean_it
            real_mean[sim] = true_Mean_it
            samples_from_truth = np.random.multivariate_normal( true_Mean_it, true_Cov_it, size = (N*n) )
            perf_pool[sim] = sqmmd_est2( hat_mu_it, samples_from_truth, kernel )
        pools_eta.append(eta_star_pool)
        pools.append(perf_pool)
        pools_snn_ests.append(snn_ests)
        pools_mmd_mean_ests.append(mmd_mean_ests)
        pools_realmean_ests.append(real_mean)
        pools_eta_snn.append(eta_star_snn_pool)

    np.save("stag80_30_2.npy", np.array(pools, dtype = object), allow_pickle=True)
    np.save("stag80_30_2_eta.npy", np.array(pools_eta, dtype = object), allow_pickle = True)
    np.save("stag80_30_2_snn_ests.npy", np.array(pools_snn_ests, dtype = object), allow_pickle = True)
    np.save("stag80_30_2_mmd_mean_ests.npy", np.array(pools_mmd_mean_ests, dtype = object), allow_pickle = True)
    np.save("stag80_30_2_realmean_ests.npy", np.array(pools_realmean_ests, dtype = object), allow_pickle = True)
    np.save("stag80_30_2_eta_snn.npy", np.array(pools_eta_snn, dtype = object), allow_pickle = True)

    #with open("results/stag40_20_4.pkl", "rb") as f:
    #    stag40_20_4 = pickle.load(f)

    #with open("results/stag40_20_4_eta.pkl", "rb") as f:
    #    stag40_20_4_eta = pickle.load(f)
    stag40_20_4 = pools
    stag40_20_4_eta = pools_eta
    center_mcar = np.zeros(4)
    sd_mcar = np.zeros(4)

    for i in range(4) : 
        center_mcar[i] = np.mean( stag40_20_4[i] )
        sd_mcar[i] = np.std( stag40_20_4[i] )

    ci_length = 1.96 * sd_mcar / np.sqrt(10)
    ci_upper = center_mcar + ci_length
    ci_lower = center_mcar - ci_length

    center_eta_mcar = np.zeros(4)
    sd_eta_mcar = np.zeros(4)

    for i in range(4) : 
        center_eta_mcar[i] = np.mean( stag40_20_4_eta[i] )
        sd_eta_mcar[i] = np.std( stag40_20_4_eta[i] )

    ci_eta_length = 1.96 * sd_eta_mcar / np.sqrt(10)
    ci_eta_upper = center_eta_mcar + ci_length
    ci_eta_lower = center_eta_mcar - ci_length


    print(center_mcar)
    log_N = np.arange(5, 9)

    df_eta_mcar = np.column_stack( (log_N, center_eta_mcar, ci_eta_upper, ci_eta_lower) )
    df_mcar = np.column_stack( (log_N, center_mcar, ci_upper, ci_lower) )

    df_mcar = pd.DataFrame(df_mcar, columns= ["Log row size", "mean", "upper 95", "lower 95"] )

    fig, ax = plt.subplots()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.4)

    x = df_mcar["Log row size"]
    l1, = ax.plot(x, df_mcar["mean"], marker=mss[3], color = "red", linestyle='None')
    ax.fill_between(x, df_mcar["lower 95"], df_mcar["upper 95"], color='b', alpha=.15)

    print(center_eta_mcar)
    plt.savefig("stag_80_30_2_plot.pdf", bbox_inches = "tight")

    # design = np.column_stack((np.ones(4), log_N))

    # model = sm.OLS(center_mcar, design).fit()
    # print(model.predict())
    # print(model.params[1])

    # l2, = ax.plot(x, (model.predict()), linestyle=lss[1], linewidth=4, color = "red", alpha=.5)
    # ls = []
    # labs = []

    # ls.append(l1, l2)

    # ax.set_ylim(ymin=0)
    # ax.set_title()
    # fig.autofmt_xdate(rotation=45)   

if __name__ == "__main__":
    main()
