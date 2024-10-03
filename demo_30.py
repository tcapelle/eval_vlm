import wandb

import numpy as np
import random


# project name
wandb.init(
    project="demo_30", 
    config={"epochs": 10, "batch_size": 32, "learning_rate": 0.001},)

config = wandb.config

# artifact_ds = wandb.use_artifact('capecape/demo_30/dataset:v0', type='dataset')
# dataset = artifact_ds.download()
table_data = []
table = wandb.Table(columns=["epoch", "image", "loss", "accuracy"])

for epoch in range(config.epochs):
    accuracy = random.random()
    loss = random.random()*0.2

    # log accuracy and loss
    wandb.log({"accuracy": accuracy, "loss": loss}, step=epoch)


    table.add_data(epoch, wandb.Image(np.random.rand(100, 100, 3)), loss, accuracy)

wandb.log({"fancy_table": table})
# artifact = wandb.Artifact("my_model", type="model")
# artifact.add_file("model.pth")
# wandb.log_artifact(artifact)