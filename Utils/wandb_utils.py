import wandb

def log_performance_with_wandb(eval_metric_df, is_save_wandb, **config_kwargs):
    if is_save_wandb:
        print('save fram_performance to wandb')
        wandb.log(eval_metric_df.to_dict())
    else:
        print('frame_perforamnce is not ssaved to wandb')
