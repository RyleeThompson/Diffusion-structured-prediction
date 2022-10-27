import json
import pickle
import turibolt as bolt
import wandb
import os
import pandas as pd
import shutil
import time

config_file = 'config.json'

def sync_run(task, args):
    """Syncs the given bolt task to weights & biases. Currently just very simple
    functionality. Uploads the logged bolt metrics and log files.

    Parameters
    ----------
    task : bolt.Task
        The bolt task to sync to wandb
    args : argparse.Namespace
        The command line arguments containing extra info
    """
    cfg = get_config(task)
    cfg['bolt_task_id'] = args.task_id
    metrics = get_metrics(task)

    wandb_run = wandb.init(
        project=args.wandb_project,
        resume='allow',
        config=cfg,
        reinit=True
    )

    with wandb_run:
        upload_metrics(metrics, wandb_run)
        upload_log_files(task, wandb_run)
    wandb.join()

def get_metrics(task):
    """Get the logged metrics from the bolt task. Probably breaks/has undefined
    behaviour if metrics are not all logged at the same interval.

    Parameters
    ----------
    task : bolt.Task
        The bolt task to get metrics for.

    Returns
    -------
    List of dicts
        Contains the metrics logged at each interval (epoch).
    """
    metrics = task.get_metrics() # Returns a dict with a single key which is the run id
    # Extract the actual metrics
    metrics_dct = metrics[list(metrics.keys())[0]] # This returns a dict with the metric names as keys and values as pd.DataFrames
    metrics_df = pd.concat([metric['metric_value'] for metric in metrics_dct.values()], axis=1) # Concatenate dataframes
    metrics_df.columns = list(metrics_dct.keys()) # Rename columns to be each metric (they're all given the name 'metric_value' originally)
    metrics_dct = metrics_df.to_dict(orient='records') # Convert df to list of dicts
    return metrics_dct

def get_config(task):
    """Gets the model configuration file for the given bolt job (NOT the bolt configuration used).
    File name is set at the top of the file.

    Parameters
    ----------
    task : bolt.Task
        The bolt task to download the model configuration for.

    Returns
    -------
    dict
        Dictionary containing the model configuration
    """
    task.artifacts.download_file(src=config_file, dest=f'./_temp.json') # Download config file
    cfg = json.load(open('./_temp.json', 'r'))
    os.remove('./_temp.json')
    return cfg

def upload_metrics(metrics, wandb_run):
    """Upload logged bolt metrics to wandb.

    Parameters
    ----------
    metrics : list of dicts
        List of dictionaries containing the metrics to upload to wandb.
    wandb_run : wandb.Run
        The wandb run to upload to.
    """
    for result in metrics:
        wandb.log(result)

def upload_log_files(task, wandb_run):
    # Download the bolt log files to the wandb run directory so they sync to wandb
    task.logs.download_dir(dest=wandb_run.dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', help='The task ID you want to sync to W&B.')
    parser.add_argument('--wandb_project', help='The wandb project name to sync to.', default='test')
    args = parser.parse_args()

    parent = bolt.get_task(args.task_id)
    children = list(parent.children)
    if len(children) > 0:
        for child in children:
            sync_run(child, args)
    else:
        sync_run(parent, args)
