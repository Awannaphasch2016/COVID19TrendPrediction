import wandb
import math
import random
import wandb
import os

# run = wandb.init(project='test',job_type="dataset-creation",)
# # Start a new run, tracking hyperparameters in config
# run = wandb.init(project="test", 
#     # group="OneStep/PredictNextN/WindowLengthN/state",
#     name='Dataset',
#     save_code=True,
#     job_type='train',
#     # tags=['Outputs', 'WindowLengthN','OneStep','PredictNextN', 'state'],
#     tags=['Data', 'Raw', 'COVID19Cases', 'StateLevels', 'us-states.csv'],

#     # config={
#     #     "name": "Anak",
#     #     "learning_rate": 0.01,
#     #     "dropout": 0.2,
#     #     "architecture": "CNN",
#     #     "dataset": "CIFAR-100",
#     # }

# )

# # artifact = wandb.Artifact('Raw_COVID19Cases_StateLevels_us-states.csv', type='dataset')
# # artifact.add_file('Data/Raw/COVID19Cases/StateLevels/us-states.csv')
# # run.log_artifact(artifact)
# # wandb.save('README.md')
# wandb.finish()

# FRAME_PERFORMANCE_PATH = f"/Outputs/Models/Performances/Baselines/{}/PredictNext{}/WindowLength{}/{}/{}_{}_performance.csv"

performance_path = 'Outputs/Models/Performances/Baselines/OneStep/PredictNext7/WindowLength1/Alaska/Alaska_mlp_performance.csv'
from pandas import read_csv

performance_dict = read_csv(performance_path).to_dict()
cols = list(performance_dict.keys())
del performance_dict[cols[0]]
performance_dict = { i:list(j.values())[0] for i, j in performance_dict.items()}

# Start a new run, tracking hyperparameters in config
wandb.init(project="test", 
    # group="OneStep/PredictNextN/WindowLengthN/state",
    name='df',
    save_code=True,
    job_type='train',
    tags=['WindowLengthN','OneStep','PredictNextN', 'state'],
    config={
        "name": "Anak",
        "learning_rate": 0.01,
        "dropout": 0.2,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
    }
)

config = wandb.config
wandb.run.name = '_'.join([wandb.run.name, wandb.run.id])
print(wandb.run.id)
print(wandb.run.name)

# Simulating a training or evaluation loop
for x in range(50):
  acc = math.log(1 + x + random.random()*config.learning_rate) + random.random() + config.dropout
  loss = 10 - math.log(1 + x + random.random() + config.learning_rate*x) + random.random() + config.dropout
  # Log metrics from your script to W&B
  wandb.log({"acc":acc, "loss":loss})
