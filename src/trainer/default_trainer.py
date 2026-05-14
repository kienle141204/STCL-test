import os
import torch
import random
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch

from src.model.ewc import EWC
from src.model.gdap import compute_node_stats, compute_plasticity_weights, compute_gdap_fisher, gdap_ewc_loss
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np
from utils.common_tools import mkdirs, load_best_model


def train(inputs, args):
    path = osp.join(args.path, str(args.year))  # Define the current year model save path
    if osp.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pkl"):
                os.remove(osp.join(path, f))
        args.logger.warning("[*] Cleared existing checkpoints in {}".format(path))
    mkdirs(path)
    
    # Setting the loss function
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss
    
    # Dataset definition
    # Subsample training data for subsequent years if incremental_train_ratio is set
    incremental_train_ratio = getattr(args, 'incremental_train_ratio', 1.0)
    train_x = inputs["train_x"]
    train_y = inputs["train_y"]
    if args.year > args.begin_year and incremental_train_ratio < 1.0:
        n_total = train_x.shape[0]
        n_sample = max(1, int(n_total * incremental_train_ratio))
        train_x = train_x[:n_sample]
        train_y = train_y[:n_sample]
        args.logger.info(f"[*] Using first {n_sample}/{n_total} samples ({incremental_train_ratio*100:.0f}%)")
    
    if args.strategy == 'incremental' and args.year > args.begin_year:
        # Incremental Policy Data Loader
        train_loader = DataLoader(SpatioTemporalDataset("", "", x=train_x[:, :, args.subgraph.numpy()], y=train_y[:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        # Construct the adjacency matrix of the subgraph
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)  # Convert to adjacency matrix
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)  # Normalized adjacency matrix
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        # Common Data Loader
        train_loader = DataLoader(SpatioTemporalDataset("", "", x=train_x, y=train_y, edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        vars(args)["sub_adj"] = vars(args)["adj"]  # Use the adjacency matrix of the entire graph
    
    # Test Data Loader
    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
    
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args)  # If it is not the first year, load the optimal model
        if args.ewc:  # If you use the ewc strategy, use the ewc model
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))  # Record EWC related parameters
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)  # Initialize the EWC model
            ewc_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            model.register_ewc_params(ewc_loader, lossfunc, args.device)  # Register EWC parameters
        else:
            model = gnn_model  # Otherwise, use the best model loaded
        
        if args.method == 'EAC' or args.method == 'KPrompt' or args.method == 'LSPCL':
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn" in name or "gcn2" in name or "fc" in name or "gcn" in name:
                    param.requires_grad = False

        if args.method == 'EAC':
            model.expand_adaptive_params(args.graph_size)

        if args.method == 'LSPCL':
            model.on_new_year()

        if args.method == 'SCAA':
            model.freeze_backbone()

        if args.method == 'AdaRev':
            model.expand_adaptive_params(args.graph_size)
            model.freeze_backbone()

        if args.method == 'Universal' and args.use_eac == True:
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False
        
        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
        
    else:
        gnn_model = args.methods[args.method](args).to(args.device)  # If it is the first year, use the base model
        model = gnn_model
        if args.method == 'EAC':
            model.expand_adaptive_params(args.graph_size)

        if args.method == 'GAPT':
            model.expand_adaptive_params(args.graph_size)

        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
    
    #if args.logname != 'trafficstream':
    #    model.count_parameters()
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name} | Requires Grad: {param.requires_grad}")
    
    
    # GDAP pre-training: always compute per-node stats; set plasticity weights for year > begin_year
    gdap_curr_mean = gdap_curr_std = None
    if args.method == 'GDAP':
        gdap_curr_mean, gdap_curr_std = compute_node_stats(train_x)
        gdap_buffer = getattr(args, 'gdap_buffer', None)
        if args.year > args.begin_year and gdap_buffer is not None:
            adj_curr_np = args.adj.cpu().numpy()
            adj_prev_np = gdap_buffer['adj']
            w = compute_plasticity_weights(
                adj_curr_np, adj_prev_np,
                gdap_curr_mean, gdap_buffer['node_mean'], gdap_buffer['node_std'],
                N_prev=adj_prev_np.shape[0],
                beta=getattr(args, 'gdap_beta', 1.0),
                gamma=getattr(args, 'gdap_gamma', 0.5),
                device=args.device,
            )
            model.plasticity_weights = w
            model.fisher_diag_dict   = gdap_buffer.get('fisher', {})
            model.prev_params        = gdap_buffer.get('prev_params', {})
            args.logger.info(
                "[GDAP] year {} plasticity: mean={:.4f} min={:.4f} max={:.4f}".format(
                    args.year, w.mean().item(), w.min().item(), w.max().item()
                )
            )

    # Model Optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []
    
    for epoch in range(args.epoch):
        
        start_time = datetime.now()
        
        # Training the model
        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)
            
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)  # to_dense_batch is used to convert a batch of sparse adjacency matrices into a batch of dense adjacency matrices
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]  # Slice according to the mapping to obtain the prediction and true value of the change node
                data.y = data.y[:, args.mapping, :]
            
            if args.method == 'SCAA' and getattr(model, 'node_weights', None) is not None:
                # Improvement D — SIS-weighted loss: focus training on structurally
                # changed nodes while still learning on stable ones (min weight 0.3).
                N  = args.sub_adj.shape[0]
                bs = pred.shape[0] // N
                w  = model.node_weights.repeat(bs).unsqueeze(-1)   # [bs*N, 1]
                loss = (lossfunc(data.y, pred, reduction='none') * w).mean()
            else:
                loss = lossfunc(data.y, pred, reduction="mean")

            if args.method in ('KPrompt', 'LSPCL', 'SCAA') and getattr(model, 'aux_loss', None) is not None:
                loss = loss + model.aux_loss

            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()  # Calculate and add ewc loss if necessary

            if (args.method == 'GDAP' and args.year > args.begin_year
                    and model.fisher_diag_dict and model.prev_params):
                ewc_pen = gdap_ewc_loss(model, model.fisher_diag_dict, model.prev_params)
                w_mean  = (model.plasticity_weights.mean()
                           if model.plasticity_weights is not None
                           else torch.tensor(0.5, device=args.device))
                loss = loss + getattr(args, 'gdap_lambda_ewc', 0.1) * (1.0 - w_mean) * ewc_pen

            training_loss += float(loss)
            cn += 1
            
            loss.backward()
            optimizer.step()
        
        
        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn 
        
        # Validate the model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        # Early Stopping Strategy
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            if args.ewc:
                torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
            else:
                torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
        
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
        
    if args.method == 'TrafficStream':
        best_model = args.methods[args.method](args)
    
    else:
        best_model = model
    
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)

    # Save GAPT prompts after training this year
    if args.method == 'GAPT' and hasattr(best_model, 'save_prompts'):
        best_model.save_prompts(year=args.year, n_nodes=args.graph_size)

    # GDAP post-training: compute Fisher + save buffer for next phase
    if args.method == 'GDAP':
        n_fisher = getattr(args, 'gdap_n_fisher_samples', 200)
        fisher = compute_gdap_fisher(
            best_model, train_x, train_y, args.adj,
            n_samples=n_fisher, batch_size=args.batch_size, device=args.device,
        )
        prev_params = {
            n: p.data.clone()
            for n, p in best_model.named_parameters()
            if p.requires_grad
        }
        node_mean = gdap_curr_mean if gdap_curr_mean is not None else np.zeros(args.graph_size)
        node_std  = gdap_curr_std  if gdap_curr_std  is not None else np.ones(args.graph_size)
        vars(args)['gdap_buffer'] = {
            'node_mean':   node_mean,
            'node_std':    node_std,
            'adj':         args.adj.cpu().numpy(),
            'fisher':      fisher,
            'prev_params': prev_params,
        }
        args.logger.info("[GDAP] buffer saved for year {} (N={})".format(args.year, args.graph_size))

    # Test the Model
    test_model(best_model, args, test_loader, True)
    args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred = model(data, args.adj)
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args)