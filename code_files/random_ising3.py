import numpy as np
import tqdm
import matplotlib.pyplot as plt

def gibbs_transition(samples, temp, edge_weights):
    batch_size = samples.shape[0]
    grid_size = list(samples.shape[1:])
    assert grid_size[0] % 2 == 0 and grid_size[1] % 2 == 0
    grid_x = np.arange(grid_size[0])[:,None]
    grid_y = np.arange(grid_size[1])
    mask = (grid_x+grid_y) % 2 == 0

    coeffs = edge_weights[0]*np.roll(samples, -1, axis=1) + np.roll(edge_weights[0]*samples, 1, axis=1) + edge_weights[1]*np.roll(samples, -1, axis=2) + np.roll(edge_weights[1]*samples, 1, axis=2)
    probs = 1/(1+np.e**(-2*temp*coeffs))
    samples = np.where(mask, (np.random.rand(*probs.shape) < probs).astype(np.float32) * 2 - 1, samples)

    coeffs = edge_weights[0]*np.roll(samples, -1, axis=1) + np.roll(edge_weights[0]*samples, 1, axis=1) + edge_weights[1]*np.roll(samples, -1, axis=2) + np.roll(edge_weights[1]*samples, 1, axis=2)
    probs = 1/(1+np.e**(-2*temp*coeffs))
    samples = np.where(np.logical_not(mask), (np.random.rand(*probs.shape) < probs).astype(np.float32) * 2 - 1, samples)

    return samples

def annealed_importance_sampling(edge_weights, T):
    assert edge_weights[0].shape == edge_weights[1].shape
    batch_size = 10
    grid_size = edge_weights[0].shape
    logprob_diffs = []
    samples = np.random.randint(0, 2, (batch_size, grid_size[0], grid_size[1]))*2-1
    n_iters = 3000
    for i in range(n_iters):
        log_prob = np.sum(samples*np.roll(samples, -1, axis=1)*edge_weights[0] + samples*np.roll(samples, -1, axis=2)*edge_weights[1], axis=(1, 2))
        logprob_diff = T*log_prob / n_iters
        logprob_diffs.append(logprob_diff)
        samples = gibbs_transition(samples, T*(i+1)/n_iters, edge_weights)

    base_log_Z = grid_size[0]*grid_size[1]*np.log(2)
    logprob_diffs = np.stack(logprob_diffs, axis=0)  # temp, batch
    prob_diffs = np.e**logprob_diffs
    prob_diffs = np.mean(prob_diffs, axis=1)  # temp
    logprob_diffs = np.log(prob_diffs)
    logprob_diffs = base_log_Z + np.sum(logprob_diffs)  # dimensionless
    return logprob_diffs, samples

def tempering_sample(grid_size, edge_weights, T, n_samples=1, verbose=True):
    assert edge_weights[0].shape == edge_weights[1].shape
    grid_size = list(edge_weights[0].shape)
    temperatures = (np.linspace(0, 1.2, 12*4+1)*T).tolist()
    grid = np.ones(grid_size)
    log_Zs = []
    for t in tqdm.tqdm(temperatures, desc="Measuring Z", disable = not verbose):
        log_Zs.append(annealed_importance_sampling(edge_weights, t)[0])
    temp_num = 0
    batch_size = 1
    samples = np.random.randint(0, 2, size=([batch_size] + grid_size))*2-1
    temp_history = []
    mean_history = []
    select_samples = []
    n_iters = 1000000
    progress_bar = tqdm.tqdm(total=n_samples+2, desc="Sampling", position=0, disable = not verbose)
    thermalize_steps = 4
    for i in range(n_iters):
#    while True:
        if np.random.rand() < 0.5:  # gibbs
            samples = gibbs_transition(samples, temperatures[temp_num], edge_weights)
        else:
            # MH
            if temp_num == 0:
                new_temp_num = 1
            elif temp_num == len(temperatures)-1:
                new_temp_num = len(temperatures)-2
            else:
                new_temp_num = int(temp_num + np.random.binomial(1, 1/2)*2-1)
            log_prob = np.sum(samples*np.roll(samples, -1, axis=1)*edge_weights[0] + samples*np.roll(samples, -1, axis=2)*edge_weights[1], axis=(1, 2))
            log_prob_diff = (temperatures[new_temp_num]-temperatures[temp_num])*log_prob
            Z_diff = log_Zs[new_temp_num] - log_Zs[temp_num]
            tot_diff = log_prob_diff - Z_diff
            probability = 1 if tot_diff > 0 else np.e**tot_diff
#            if np.random.binomial(1, probability) == 1:
            if np.random.rand() < probability:
                temp_num = new_temp_num
        temp_history.append(temperatures[temp_num])
        mean_history.append(np.mean(samples[0]))
        if np.abs(temperatures[temp_num] - T) < 1e-4:
            if not select_samples or np.sum(samples[0]*select_samples[-1]) < 0:
                select_samples.append(np.copy(samples[0,:,:]))
                progress_bar.update(1)
                if len(select_samples) >= thermalize_steps+n_samples:
                    samples = np.stack(select_samples[thermalize_steps:], axis=0)
                    for i in range(1000):
                        samples = gibbs_transition(samples, temperatures[temp_num], edge_weights)
                    return samples, temp_history, mean_history, True

    return samples, temp_history, mean_history, False

def mutual_information(grid_size, u, v, T, p):
    n_models = 10
    n_samples = 100
#    n_models = 1
#    n_samples = 10
    mutual_informations = []
    mutual_information_stds = []
    for i in range(n_models):
        print("Ising Model #"+str(i)+"/"+str(n_models))
        node_weights = np.random.randint(0, 2, size=grid_size)*2-1
        edge_weights = [node_weights*np.roll(node_weights, -1, axis=0), node_weights*np.roll(node_weights, -1, axis=1)]
        mutators = ((np.random.rand(*grid_size) < p).astype(np.float32)*2-1), ((np.random.rand(*grid_size) < p).astype(np.float32)*2-1)
        edge_weights = [mutators[0]*edge_weights[0], mutators[1]*edge_weights[1]]
        samples, _, _, correct = tempering_sample(grid_size, edge_weights, T, n_samples=n_samples, verbose=True)
        x_u = samples[:,u[0], u[1]]
        x_v = samples[:,v[0], v[1]]
        same = (x_u==x_v).astype(np.float32)
        same_rate = np.mean(same)
        print(same_rate)
        same_rate_std = np.sqrt(same_rate*(1-same_rate)/n_samples)
        if same_rate == 1 or same_rate == 0:
            mutual_information = np.log(2)
            mutual_information_std = 0  # This is the limit as same_rate -> 0 or 1
        else:
            mutual_information = np.log(2) - (-(same_rate*np.log(same_rate) + (1-same_rate)*np.log(1-same_rate)))
            mutual_information_std = np.abs(np.log(1/same_rate-1))*same_rate_std
        mutual_informations.append(mutual_information)
        mutual_information_stds.append(mutual_information_std)
    mutual_information = np.mean(mutual_informations)
    mutual_information_std = np.sqrt(np.sum(mutual_information_std**2) + np.std(mutual_informations)**2*n_models)/n_models
    return mutual_information, mutual_information_std


p = 0.1
p = 1/2 + 1/np.sqrt(8)
p = 0.2
for p in reversed([0.1, 1/2 - 1/np.sqrt(8), 0.2]):
    T = np.log(p/(1-p))/2
    grid_size = [100, 100]
    #node_weights = np.random.randint(0, 2, size=grid_size)*2-1
    node_weights = np.ones(grid_size)
    edge_weights = [node_weights*np.roll(node_weights, -1, axis=0), node_weights*np.roll(node_weights, -1, axis=1)]
    mutators = ((np.random.rand(*grid_size) < p).astype(np.float32)*2-1), ((np.random.rand(*grid_size) < p).astype(np.float32)*2-1)
    edge_weights = [mutators[0]*edge_weights[0], mutators[1]*edge_weights[1]]

    samples, temp_history, mean_history, success = tempering_sample(grid_size, edge_weights, T, 1, verbose=True)
    sample = samples[0]
    print(success)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow((sample+1)/2, cmap="Greys")
    plt.savefig('tempering_sample_'+str(p)+'.pdf', bbox_inches='tight')
    plt.close()
