import os, re, json, csv, torch
import os.path as osp
import numpy as np
from datetime import datetime
from Bio.Cluster import kcluster


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)


def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year +  1) or args.train == 0:  # Determine whether to load the first year's model
        load_path = args.first_year_model_path  # Set the loading path to the first year model path
        loss = load_path.split("/")[-1].replace(".pkl", "")  # Get the model file name and remove the extension
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year-1))):  # Traverse the files under the model path of the previous year and get all loss values
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year-1), loss[0]+".pkl")  # Set the loading path to the model file corresponding to the minimum loss value
        
    args.logger.info("[*] load from {}".format(load_path))  # Recording Load Path
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]  # Loading the model state dictionary
    
    model = args.methods[args.method](args)  # Initialize the model
    
    if args.method == 'EAC':
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    if args.method == 'Universal' and args.use_eac == True:
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        import logging
        logging.getLogger(__name__).warning(
            f"load_state_dict: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(args.device)  # Move the model to the specified device
    return model, loss[0]  # Returns the model and the minimum loss value


def long_term_pattern(args, long_pattern):
    attention, _, _ = kcluster(long_pattern, nclusters=args.cluster, dist='u')  # [number of nodes, average number of days per day] -> [number of nodes] ranges from 0 to k-1
    np_attention = np.zeros((len(attention), args.cluster))  # [number of nodes, clusters]
    for i in attention:
        np_attention[i][attention[i]] = 1.0
    return np_attention.astype(np.float32)


def get_max_columns(matrix):
    tensor = torch.tensor(matrix)
    max_columns, _ = torch.max(tensor, dim=1)
    return max_columns


def save_results_csv(args, total_time, csv_path="results.csv"):
    years = list(range(args.begin_year, args.end_year + 1))

    def mean_metric(horizon_key, metric_key):
        bucket = args.result.get(horizon_key, {}).get(metric_key, {})
        vals = [bucket[y] for y in years if y in bucket]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    dataset = osp.basename(args.model_path.rstrip("/\\"))

    row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "logname":      args.logname,
        "method":       args.method,
        "backbone":     getattr(args, "backbone_type", "stgnn"),
        "dataset":      dataset,
        "seed":         args.seed,
        "begin_year":   args.begin_year,
        "end_year":     args.end_year,
        "strategy":     getattr(args, "strategy", ""),
        "lr":           args.lr,
        "batch_size":   args.batch_size,
        "epoch":        args.epoch,
        "hidden_ch":    args.gcn["hidden_channel"],
        "dropout":      args.dropout,
        "total_time_s": round(total_time, 2),
        # avg over all 12 horizons, then averaged over all years
        "avg_MAE":      mean_metric("Avg", " MAE"),
        "avg_RMSE":     mean_metric("Avg", "RMSE"),
        "avg_MAPE":     mean_metric("Avg", "MAPE"),
    }

    write_header = not osp.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    args.logger.info(f"[*] Results appended to {csv_path}")