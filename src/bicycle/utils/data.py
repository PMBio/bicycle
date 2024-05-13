import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from bicycle.utils.training import lyapunov_direct
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def create_data(
    n_genes,
    n_samples_control,
    n_samples_per_perturbation,
    make_counts=False,
    train_gene_ko=None,
    test_gene_ko=None,
    graph_type="erdos-renyi",
    edge_assignment="random-uniform",
    sem="linear-ou",
    make_contractive=True,
    verbose=False,
    device="cpu",
    intervention_type = "dCas9",
    T = 1.0,
    library_size_range = [5000, 25000],
    **graph_kwargs,
):
    N = n_genes

    # We start counting gene ids from 0, 1, ..., N-1
    # We start counting regimes/contexts from 0, 1, ...

    if verbose:
        print("Training/Validation Target Genes:")
        print(train_gene_ko)

        print("Test Target Genes:")
        print(test_gene_ko)

    # No intervention must be in train and test sim regimes:
    if len([x for x in train_gene_ko if x in test_gene_ko]) > 0:
        raise ValueError("train and test gene knock-outs must be disjoint")

    sim_regime_ctrl = torch.zeros((n_samples_control,), device=device)
    n_contexts_list = train_gene_ko + test_gene_ko
    n_contexts_list_w_ctrl = [""] + n_contexts_list

    sim_regime_pert = torch.arange(1, 1 + len(n_contexts_list), device=device).repeat_interleave(
        n_samples_per_perturbation
    )

    # Shuffle all tensor elements randomly
    sim_regime_pert = sim_regime_pert[torch.randperm(len(sim_regime_pert))]

    sim_regime = torch.cat((sim_regime_ctrl, sim_regime_pert)).long()

    # Genes x regimes matrices, which indexes which genes are intervened in
    # which context
    gt_interv = torch.zeros((N, len(n_contexts_list) + 1), device=device)

    # Gene i is intervened in regime i
    for idx, p in enumerate(n_contexts_list_w_ctrl):
        # Empy string represents control
        if p == "":
            continue
        elif "," in p:
            # Comma separated values represent multiple interventions
            for pp in p.split(","):
                gt_interv[int(pp), idx] = 1
        else:
            gt_interv[int(p), idx] = 1

    intervened_variables = gt_interv[:, sim_regime].transpose(0, 1)

    unsuccessful_sampling = True
    while unsuccessful_sampling:
        beta = generate_weighted_graph(
            graph_type=graph_type,
            nodes=N,
            edge_assignment="random-uniform",
            make_contractive=True,
            device=device,
            **graph_kwargs,
        )
        beta = torch.tensor(beta, device=device).float()
        # Compute eigenvalues of beta
        B = torch.eye(N) - beta
        eigvals_B = np.real(np.linalg.eigvals(B))
        # Check if all eigvals are positive
        if np.all(eigvals_B > 0):
            unsuccessful_sampling = False
        else:
            print("*" * 100)
            print("Unsuccessful sampling. Re-sampling...")
            print("*" * 100)

    # Print sparsity level of adjacency matrix
    if verbose:
        print(f"Sparsity level of adjacency matrix: {(100*(beta != 0).sum() / (N * N)):.2f}%")

        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
        sns.heatmap(
            gt_dyn,
            annot=True,
            center=0,
            cmap="vlag",
            square=True,
            annot_kws={"fontsize": 7},
            ax=ax[0],
        )

        adjacency_matrix = (beta != 0).detach().cpu().numpy().astype(int)
        mylabels = {i: k for i, k in enumerate([f"{i}" for i in range(N)])}
        # Check if adjacency matrix contains cycles
        if not nx.is_directed_acyclic_graph(nx.from_numpy_array(adjacency_matrix)):
            print("Adjacency matrix contains cycles")
        rows, cols = np.where(adjacency_matrix != 0)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.DiGraph(directed=True)
        all_rows = range(0, adjacency_matrix.shape[0])
        for n in all_rows:
            G.add_node(n)
        G.add_edges_from(edges)
        nx.draw(
            G,
            node_size=250,
            labels=mylabels,
            with_labels=True,
            ax=ax[1],
            pos=nx.circular_layout(G),
        )
        plt.show()

    if sem == "linear-ou":
        gt_dyn = (
            ((1.0 - torch.eye(N, device=device)) * beta - torch.eye(N, device=device)).detach().cpu().numpy()
        )

        # Knockdown-Interventions:
        alpha_p = 0.1 * torch.ones(N, device=device)
        # prev: 1.0 * torch.ones(N, device=device)
        alpha = 1.0 * torch.ones(
            N, device=device
        )  # torch.distributions.Gamma(0.5, 1).sample((N,)).to(device)
        sigma_p = 0.001 * torch.ones(N, device=device)
        # prev: sigma = 0.1 * torch.ones(N, device=device)
        sigma = 0.1 * torch.ones(
            N, device=device
        )  # 0.001 * torch.distributions.Gamma(0.5, 1).sample((N,)).to(device)
        beta_p = torch.zeros((N, N), device=device)

        iv_a = (1 - gt_interv).T
        
        if intervention_type == "dCas9":
            
            print('Simulating data of intervention type dCas9')
            
            betas = iv_a[:, None, :] * beta + (1 - iv_a)[:, None, :] * beta_p
            alphas = iv_a * alpha[None, :] + (1 - iv_a) * alpha_p[None, :]
            sigmas = iv_a[:, None, :] * torch.diag(sigma) + (1 - iv_a)[:, None, :] * torch.diag(sigma_p)
            
            print('Shapes dCas9:',betas.shape, alphas.shape, sigmas.shape)
            
        elif intervention_type == "Cas9":
            
            print('Simulating data of intervention type Cas9')
            
            betas = iv_a[:, :, None] * beta + (1 - iv_a)[:, :, None] * beta_p
            alphas = alpha[None, :].expand(iv_a.shape[0],alpha.shape[0])
            sigmas = torch.diag(sigma)[None, :, :].expand(iv_a.shape[0],sigma.shape[0],sigma.shape[0])
            
            print('Shapes Cas9:',betas.shape, alphas.shape, sigmas.shape)
        else:
            raise NotImplementedError("Currently only Cas9 and dCas9 are supported as intervention_type.")

        B = torch.eye(N, device=betas.device)[None, :, :] - (1.0 - torch.eye(N, device=betas.device))[
            None, :, :
        ] * betas.transpose(1, 2)

        omegas = lyapunov_direct(
            B.double(),
            torch.bmm(sigmas, sigmas.transpose(1, 2)).double(),
        ).float()

        # Broadcast arrays back to batch_shape
        B_broadcasted = B[sim_regime]
        alphas_broadcasted = alphas[sim_regime]
        omegas_broadcasted = omegas[sim_regime]

        # for k in range(omegas.shape[0]):
        # print(np.linalg.cholesky(omegas[k]))
        # print(sorted([np.round(x, 4) for x in np.linalg.eigvals(omegas[k])]))

        x_bar = torch.bmm(torch.linalg.inv(B), alphas[:, :, None]).squeeze()

        x_bar_broadcasted = x_bar[sim_regime]
        
        samples = (
            torch.distributions.MultivariateNormal(x_bar_broadcasted, covariance_matrix=omegas_broadcasted)
            .sample()
            .detach()
        )

        if verbose:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            plt.scatter(
                samples[:, 0][sim_regime == 0].cpu().numpy(),
                samples[:, 1][sim_regime == 0].cpu().numpy(),
                # c=[int(x) for x in sim_regime.cpu().numpy()],
                s=2,
            )
            ax.grid(True)
            plt.colorbar()
            plt.show()

            plot_indices = np.where(torch.abs(beta * (1 - torch.eye(N))).cpu().numpy() > 0.5)

            i_plot = plot_indices[0][0]
            j_plot = plot_indices[1][0]

            plt.figure()
            plt.title("Weight: %f" % beta[i_plot, j_plot].item())
            plt.scatter(
                samples[:, i_plot].cpu().numpy(),
                samples[:, j_plot].cpu().numpy(),
                c=sim_regime.cpu().numpy(),
            )
            plt.xlabel("gene %d" % i_plot)
            plt.ylabel("gene %d" % j_plot)
            plt.colorbar()

        if make_counts:
            # ps = torch.nn.Softmax(dim=-1)(samples)
            ps = torch.softmax(samples / T, dim=-1)

            # Cell-specific library size
            library_size = torch.randint(
                low=library_size_range[0],
                high=library_size_range[1],
                size=(samples.shape[0], 1),
            )
            
            for i in tqdm(range(samples.shape[0])):
                P = torch.distributions.multinomial.Multinomial(total_count=library_size[i].item(), probs=ps[i]) 
                samples[i] = P.sample()

        if verbose:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            if make_counts:
                plt.scatter(
                    samples[:, 0].cpu().numpy() / library_size.cpu().numpy().squeeze(),
                    samples[:, 4].cpu().numpy() / library_size.cpu().numpy().squeeze(),
                    c=[int(x) for x in sim_regime.cpu().numpy()],
                    s=2,
                )
            else:
                plt.scatter(
                    samples[:, 0].cpu().numpy(),
                    samples[:, 4].cpu().numpy(),
                    c=[int(x) for x in sim_regime.cpu().numpy()],
                    s=2,
                )
            ax.grid(True)
            plt.colorbar()
            plt.show()

        return (gt_dyn, intervened_variables, samples, gt_interv, sim_regime, beta, x_bar, omegas)

    elif sem == "linear":
        beta = np.array(beta)

        def generate_data(n_samples, intervention_set=[None]):
            noise_scale = graph_kwargs.get("noise_scale", 0.5)
            intervention_scale = graph_kwargs.get("intervention_scale", 0.1)

            # set intervention_set = [None] for purely observational data.
            E = noise_scale * np.random.randn(N, n_samples)

            observed_set = np.setdiff1d(np.arange(N), intervention_set)
            U = np.zeros((N, N))
            U[observed_set, observed_set] = 1

            C = np.zeros((N, n_samples))
            if intervention_set[0] is not None:
                # Knock-Down
                C[intervention_set, :] = intervention_scale * np.random.randn(
                    len(intervention_set), n_samples
                )

            I = np.eye(N)
            X = np.linalg.inv(I - U @ beta.T) @ (U @ E + C)

            # check: Compute eigenvalues for U@beta.T and assert if their abs is > 1
            eigvals = np.linalg.eigvals(U @ beta)
            if not np.all(np.abs(eigvals) <= 1):
                raise ValueError("Eigenvalues of U @ beta must be <= 1")

            return X.T

        samples = []
        int_contexts = []
        for context in np.unique(sim_regime):
            # Convert context i to intervention set
            i = np.array(torch.where(gt_interv[:, context] == 1)[0])
            # # Subtract 1 from all elements in i
            # i = i - 1
            if len(i) == 0:
                X = generate_data(n_samples=torch.sum(sim_regime == context), intervention_set=[None])
            else:
                X = generate_data(n_samples=torch.sum(sim_regime == context), intervention_set=i)
            samples.append(X)

            int_contexts.append(np.ones(len(X)) * context)

        samples = np.concatenate(samples, axis=0)
        int_contexts = np.concatenate(int_contexts, axis=0)
        # Convert to tensor
        samples = torch.tensor(samples, device=device).float()
        int_contexts = torch.tensor(int_contexts, device=device).long()

        beta = torch.tensor(beta, device=device).float()

    else:
        raise ValueError(f"SEM {sem} not implemented")

    return (None, None, samples, gt_interv, int_contexts, beta)


def create_loaders(
    samples,
    sim_regime,
    validation_size,
    batch_size,
    SEED,
    train_gene_ko=None,
    test_gene_ko=None,
    num_workers=1,
    persistent_workers=False,
    covariates=None,
    **kwargs,
):
    samples_interventions = sim_regime.long().to(torch.int64)

    if len(test_gene_ko) > 0:
        # No intervention must be in train and test sim regimes:
        if len([x for x in train_gene_ko if x in test_gene_ko]) > 0:
            raise ValueError("train and test gene knock-outs must be disjoint")

        n_contexts_list = train_gene_ko + test_gene_ko
        n_contexts_list_w_ctrl = [""] + n_contexts_list

        test_regimes = list(range(len(n_contexts_list_w_ctrl)))[-len(test_gene_ko) :]

        idx_test_samples = torch.tensor(
            [False if x not in test_regimes else True for x in samples_interventions], dtype=torch.bool
        )

        sample_test = samples[idx_test_samples]
        samples_interventions_test = samples_interventions[idx_test_samples]
        
        samples = samples[~idx_test_samples]
        samples_interventions = samples_interventions[~idx_test_samples]
        
        if covariates is not None:            
            print('covariates device before:', covariates.device)
            covariates_test = covariates[idx_test_samples]            
            covariates = covariates[~idx_test_samples]
            print('covariates device after:', covariates.device)

        if validation_size > 0:
            # Stratify train and validation sets across conditions
            train_idx, validation_idx = train_test_split(
                np.arange(len(samples)),
                test_size=validation_size,
                random_state=SEED,
                shuffle=True,
                stratify=samples_interventions,
            )

            train_dataset = torch.utils.data.TensorDataset(
                torch.cat((samples[train_idx], samples[validation_idx], sample_test), dim=0),
                torch.cat(
                    (
                        samples_interventions[train_idx],
                        samples_interventions[validation_idx],
                        samples_interventions_test,
                    ),
                    dim=0,
                ),
                torch.arange(len(train_idx) + len(validation_idx) + len(sample_test)),
                torch.cat(
                    (
                        torch.zeros(samples[train_idx].shape[0]),
                        torch.ones(samples[validation_idx].shape[0]),
                        2 * torch.ones(sample_test.shape[0]),
                    ),
                    dim=0,
                ),
            )
            validation_dataset = torch.utils.data.TensorDataset(
                samples[validation_idx],
                samples_interventions[validation_idx],
                torch.arange(len(train_idx), len(train_idx) + len(validation_idx)),
                torch.ones(samples[validation_idx].shape[0]),
            )
            test_dataset = torch.utils.data.TensorDataset(
                sample_test,
                samples_interventions_test,
                torch.arange(
                    len(train_idx) + len(validation_idx),
                    len(train_idx) + len(validation_idx) + len(sample_test),
                ),
                2 * torch.ones(sample_test.shape[0]),
            )

            assert len(train_dataset) == (len(train_idx) + len(validation_idx) + len(sample_test))
            
            if covariates is not None:
                covariates_reordered = torch.concatenate(
                                           ( 
                                               covariates[train_idx],
                                               covariates[validation_idx],
                                               covariates_test 
                                           ),
                                           axis = 0                    
                                       )

        else:
            train_dataset = torch.utils.data.TensorDataset(
                torch.cat((samples, sample_test), dim=0),
                torch.cat(
                    (
                        samples_interventions,
                        samples_interventions_test,
                    ),
                    dim=0,
                ),
                torch.arange(len(samples) + len(sample_test)),
                torch.cat(
                    (
                        torch.zeros(samples.shape[0]),
                        2 * torch.ones(sample_test.shape[0]),
                    ),
                    dim=0,
                ),
            )
            test_dataset = torch.utils.data.TensorDataset(
                sample_test,
                samples_interventions_test,
                torch.arange(len(samples), len(samples) + len(sample_test)),
                2 * torch.ones(sample_test.shape[0]),
            )
            
            if covariates is not None:
                covariates_reordered = torch.concatenate(
                                           ( 
                                               covariates,
                                               covariates_test 
                                           ),
                                           axis = 0                    
                                       )

    else:
        # Leave one group out for testing
        if validation_size > 0:
            train_idx, validation_idx = train_test_split(
                np.arange(len(samples)),
                test_size=validation_size,
                random_state=SEED,
                shuffle=True,
                stratify=samples_interventions,
            )
            train_dataset = torch.utils.data.TensorDataset(
                torch.cat((samples[train_idx], samples[validation_idx]), dim=0),
                torch.cat(
                    (
                        samples_interventions[train_idx],
                        samples_interventions[validation_idx],
                    ),
                    dim=0,
                ),
                torch.arange(len(train_idx) + len(validation_idx)),
                torch.cat(
                    (
                        torch.zeros(samples[train_idx].shape[0]),
                        torch.ones(samples[validation_idx].shape[0]),
                    ),
                    dim=0,
                ),
            )
            validation_dataset = torch.utils.data.TensorDataset(
                samples[validation_idx],
                samples_interventions[validation_idx],
                torch.arange(len(train_idx), len(train_idx) + len(validation_idx)),
                torch.ones(samples[validation_idx].shape[0]),
            )
            
            if covariates is not None:
                covariates_reordered = torch.concatenate(
                                           ( 
                                               covariates[train_idx],
                                               covariates[validation_idx]
                                           ),
                                           axis = 0                    
                                       )
            
        else:
            train_dataset = torch.utils.data.TensorDataset(
                samples,
                samples_interventions,
                torch.arange(len(samples)),
                torch.zeros(samples.shape[0]),
            )
            
            if covariates is not None:
                covariates_reordered = covariates

    # Dataloader for train and val
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        **kwargs,
    )
    if validation_size > 0:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )
    else:
        validation_loader = None
    if len(test_gene_ko) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )
    else:
        test_loader = None

    if covariates is not None:
        return train_loader, validation_loader, test_loader, covariates_reordered
    else:
        return train_loader, validation_loader, test_loader


def get_diagonal_mask(n_genes, device):
    # Place 1 everywhere except for on the diagonal
    mask = torch.ones((n_genes, n_genes), device=device)
    mask -= torch.eye(n_genes, device=device)
    # sns.heatmap(mask.cpu().detach().numpy())
    return mask


def compute_inits(init_data, rank_w_cov_factor, n_contexts, normalized=False):
    samples, sim_regime, sample_idx, data_category = init_data[:]
    print("Initializing parameters from data")

    nc = samples.shape[0]

    count_per_gene = samples.sum(axis=0)
    
    print(count_per_gene)

    if not normalized:
        alpha = torch.log(count_per_gene / count_per_gene.sum())
        
        print('alpha',alpha)

        library_size = samples.sum(axis=1).reshape((-1, 1))
        
        print('library_size',library_size.min(),library_size.median(),library_size.max())

        normalized_counts = torch.log(samples / library_size + 1e-6)
        standardized_counts = normalized_counts - normalized_counts.mean(axis=0)
        
        print(standardized_counts[:10,:10])
        
    else:
        alpha = count_per_gene / count_per_gene.sum()
        standardized_counts = samples

    pca = torch.pca_lowrank(standardized_counts, q=rank_w_cov_factor, center=False, niter=50)
    low_rank_init = (pca[1] * pca[2]) / torch.sqrt(torch.tensor(nc))

    w_cov_factor = low_rank_init.unsqueeze(0).repeat(n_contexts, 1, 1)

    diag_init = standardized_counts - torch.mm(pca[0], (pca[1] * pca[2]).T)
    diag_init = torch.sqrt((diag_init**2).mean(axis=0))

    w_cov_diag = diag_init.unsqueeze(0).repeat(n_contexts, 1)

    return {
        "alpha": alpha,
        "w_cov_factor": w_cov_factor,
        "w_cov_diag": w_cov_diag,
    }


def generate_weighted_graph(
    graph_type, nodes, edge_assignment, make_contractive=True, raise_if_not_cycle=True, device="cpu", **kwargs
):
    def scale(g, edge_weights, p_edge_weights):
        if edge_weights is not None:
            if isinstance(edge_weights, (float, int)):
                edge_weights = [edge_weights]

            if p_edge_weights is None:
                p_edge_weights = np.ones(len(edge_weights)) / len(edge_weights)
            g[g != 0] = np.random.choice(edge_weights, size=g[g != 0].shape, p=p_edge_weights)

        return g

    def check_cycle(g):
        if isinstance(g, np.ndarray):
            g = nx.from_numpy_array(g, create_using=nx.DiGraph)
        if not list(nx.simple_cycles(g)):
            if raise_if_not_cycle:
                raise ValueError("Graph does not contain cycles")
            else:
                print("Graph does not contain cycles")

    def contraction(weights):
        s = np.linalg.svd(weights, compute_uv=False)
        scale = 1.1
        if s[0] >= 1.0:
            scale = 1.1 * s[0]

        return weights / scale

    # Create directed edges
    if graph_type == "erdos-renyi":
        expected_density = kwargs.get("expected_density", 2)
        adjacency_matrix = np.zeros((nodes, nodes))
        p_node = expected_density / nodes

        vertices = np.arange(nodes)
        is_dag = True
        max_guesses_allowed = 100
        count = 0

        while is_dag & (count < max_guesses_allowed):
            for i in range(nodes):
                possible_parents = np.setdiff1d(vertices, i)
                num_parents = np.random.binomial(n=len(possible_parents), p=p_node)
                parents = np.random.choice(possible_parents, size=num_parents, replace=False)

                # In networkx, the adjacency matrix is such that
                # the rows denote the parents and the columns denote the children.
                # That is, W_ij = 1 ==> i -> j exists in the graph.
                adjacency_matrix[parents, i] = 1
            graph = nx.DiGraph(adjacency_matrix)
            # check_cycle(graph)
            is_dag = nx.is_directed_acyclic_graph(graph)
            count += 1

        if is_dag:
            raise ValueError("Graph is not cyclic")
    else:
        raise NotImplementedError(f"Graph {graph} not implemented")

    # Assign weights to edges
    if edge_assignment == "random-uniform":
        abs_weight_low = kwargs.get("abs_weight_low", 0.25)
        abs_weight_high = kwargs.get("abs_weight_high", 0.95)
        p_success = kwargs.get("p_success", 0.5)
        weights = torch.zeros((nodes, nodes), device=device)

        weights = np.random.uniform(abs_weight_low, abs_weight_high, size=(nodes, nodes))
        weights *= 2 * np.random.binomial(1, p_success, size=weights.shape) - 1
        weights *= nx.to_numpy_array(graph)
    else:
        raise NotImplementedError(f"Edge assignment {edge_assignment} not implemented")

    if make_contractive:
        weights = contraction(weights)

    return weights


def process_data_for_llc(loader, gt_interv, ko_genes):
    # Datasets is a list of tensors (samples times genes) for each context
    # Loop over all loaders and extract samples and intervention idx
    dataset_samples = list()
    dataset_interventions = list()
    dataset_type = list()
    for batch in loader:
        dataset_samples.append(batch[0])
        dataset_interventions.append(batch[1])
        dataset_type.append(batch[3])
    dataset_samples = torch.cat(dataset_samples, dim=0)
    dataset_interventions = torch.cat(dataset_interventions, dim=0)
    dataset_type = torch.cat(dataset_type, dim=0)

    dataset = []
    dataset_targets = []
    for context in sorted(dataset_interventions.unique().tolist()):
        # Check if context occurs in training or testing
        # Convert context to string
        int_genes = torch.where(gt_interv[:, context] == 1)[0].tolist()
        # Join all list elements to comma separated string
        int_genes = ",".join([str(x) for x in int_genes])

        i_gene = np.array(torch.where(gt_interv[:, context] == 1)[0])
        if int_genes in ko_genes:
            dataset.append(np.array(dataset_samples[dataset_interventions == context]))
            dataset_targets.append(i_gene)

    return dataset, dataset_targets


def process_data_for_nodags(loader, gt_interv, ko_genes, n_samples_control):
    # Datasets is a list of tensors (samples times genes) for each context
    # Loop over all loaders and extract samples and intervention idx
    dataset_samples = list()
    dataset_interventions = list()
    dataset_type = list()
    for batch in loader:
        dataset_samples.append(batch[0])
        dataset_interventions.append(batch[1])
        dataset_type.append(batch[3])
    dataset_samples = torch.cat(dataset_samples, dim=0)
    dataset_interventions = torch.cat(dataset_interventions, dim=0)
    dataset_type = torch.cat(dataset_type, dim=0)

    dataset = []
    dataset_targets = []
    for context in sorted(dataset_interventions.unique().tolist()):
        # Check if context occurs in training or testing
        # Convert context to string
        int_genes = torch.where(gt_interv[:, context] == 1)[0].tolist()
        # Join all list elements to comma separated string
        int_genes = ",".join([str(x) for x in int_genes])

        i_gene = np.array(torch.where(gt_interv[:, context] == 1)[0])
        if int_genes in ko_genes:
            dataset.append(np.array(dataset_samples[dataset_interventions == context]))
            dataset_targets.append(i_gene)
        if (int_genes == "") & (n_samples_control > 0):
            dataset.append(np.array(dataset_samples[dataset_interventions == context]))
            dataset_targets.append([None])

    return dataset, dataset_targets


def create_loaders_norman(
    samples,
    regimes,
    validation_size,
    batch_size,
    SEED,
    train_regimes=None,
    test_regimes=None,
    num_workers=1,
    persistent_workers=False,
    covariates=None,
    **kwargs,
):

    if len(test_regimes) > 0:
        # No intervention must be in train and test sim regimes:
        if len([x for x in train_regimes if x in test_regimes]) > 0:
            raise ValueError("train and test regimes must be disjoint")

        idx_test_samples = torch.tensor(
            [False if x not in test_regimes else True for x in regimes], dtype=torch.bool
        )

        samples_test = samples[idx_test_samples]
        regimes_test = regimes[idx_test_samples]
        
        samples = samples[~idx_test_samples]
        regimes = regimes[~idx_test_samples]
        
        if covariates is not None:            
            print('covariates device before:', covariates.device)
            covariates_test = covariates[idx_test_samples]            
            covariates = covariates[~idx_test_samples]
            print('covariates device after:', covariates.device)

        if validation_size > 0:
            # Stratify train and validation sets across conditions
            train_idx, validation_idx = train_test_split(
                np.arange(len(samples)),
                test_size=validation_size,
                random_state=SEED,
                shuffle=True,
                stratify=regimes,
            )

            train_dataset = torch.utils.data.TensorDataset(
                torch.cat((samples[train_idx], samples[validation_idx], samples_test), dim=0),
                torch.cat(
                    (
                        regimes[train_idx],
                        regimes[validation_idx],
                        regimes_test,
                    ),
                    dim=0,
                ),
                torch.arange(len(train_idx) + len(validation_idx) + len(samples_test)),
                torch.cat(
                    (
                        torch.zeros(samples[train_idx].shape[0]),
                        torch.ones(samples[validation_idx].shape[0]),
                        2 * torch.ones(samples_test.shape[0]),
                    ),
                    dim=0,
                ),
            )
            validation_dataset = torch.utils.data.TensorDataset(
                samples[validation_idx],
                regimes[validation_idx],
                torch.arange(len(train_idx), len(train_idx) + len(validation_idx)),
                torch.ones(samples[validation_idx].shape[0]),
            )
            test_dataset = torch.utils.data.TensorDataset(
                samples_test,
                regimes_test,
                torch.arange(
                    len(train_idx) + len(validation_idx),
                    len(train_idx) + len(validation_idx) + len(samples_test),
                ),
                2 * torch.ones(samples_test.shape[0]),
            )

            assert len(train_dataset) == (len(train_idx) + len(validation_idx) + len(samples_test))
            
            if covariates is not None:
                covariates_reordered = torch.concatenate(
                                           ( 
                                               covariates[train_idx],
                                               covariates[validation_idx],
                                               covariates_test 
                                           ),
                                           axis = 0                    
                                       )

        else:
            train_dataset = torch.utils.data.TensorDataset(
                torch.cat((samples, samples_test), dim=0),
                torch.cat(
                    (
                        regimes,
                        regimes_test,
                    ),
                    dim=0,
                ),
                torch.arange(len(samples) + len(samples_test)),
                torch.cat(
                    (
                        torch.zeros(samples.shape[0]),
                        2 * torch.ones(samples_test.shape[0]),
                    ),
                    dim=0,
                ),
            )
            test_dataset = torch.utils.data.TensorDataset(
                samples_test,
                regimes_test,
                torch.arange(len(samples), len(samples) + len(samples_test)),
                2 * torch.ones(samples_test.shape[0]),
            )
            
            if covariates is not None:
                covariates_reordered = torch.concatenate(
                                           ( 
                                               covariates,
                                               covariates_test 
                                           ),
                                           axis = 0                    
                                       )

    else:
        # Leave one group out for testing
        if validation_size > 0:
            train_idx, validation_idx = train_test_split(
                np.arange(len(samples)),
                test_size=validation_size,
                random_state=SEED,
                shuffle=True,
                stratify=samples_interventions,
            )
            train_dataset = torch.utils.data.TensorDataset(
                torch.cat((samples[train_idx], samples[validation_idx]), dim=0),
                torch.cat(
                    (
                        samples_interventions[train_idx],
                        samples_interventions[validation_idx],
                    ),
                    dim=0,
                ),
                torch.arange(len(train_idx) + len(validation_idx)),
                torch.cat(
                    (
                        torch.zeros(samples[train_idx].shape[0]),
                        torch.ones(samples[validation_idx].shape[0]),
                    ),
                    dim=0,
                ),
            )
            validation_dataset = torch.utils.data.TensorDataset(
                samples[validation_idx],
                samples_interventions[validation_idx],
                torch.arange(len(train_idx), len(train_idx) + len(validation_idx)),
                torch.ones(samples[validation_idx].shape[0]),
            )
            
            if covariates is not None:
                covariates_reordered = torch.concatenate(
                                           ( 
                                               covariates[train_idx],
                                               covariates[validation_idx]
                                           ),
                                           axis = 0                    
                                       )
            
        else:
            train_dataset = torch.utils.data.TensorDataset(
                samples,
                samples_interventions,
                torch.arange(len(samples)),
                torch.zeros(samples.shape[0]),
            )
            
            if covariates is not None:
                covariates_reordered = covariates

    # Dataloader for train and val
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        **kwargs,
    )
    if validation_size > 0:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )
    else:
        validation_loader = None
    if len(test_regimes) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )
    else:
        test_loader = None

    if covariates is not None:
        return train_loader, validation_loader, test_loader, covariates_reordered
    else:
        return train_loader, validation_loader, test_loader
