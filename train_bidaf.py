import config
import wandb
from utils.utils import *
from models.BIDAF import *

setup_seed(168)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize wandb logging to your project
wandb.init(
    job_type='Question_Answering_Squad2.0',
    project="BiDAF_model",
    dir='/root/autotmp-dl/????',
    entity=config.TEAM_NAME,
    config=config.wandb_config,
    # sync_tensorboard=True,
    name='?????????',
    notes='min_lr=0.00001',
    ####
)

# To log wandb
for epoch in range(10):
    wandb.log({'epoch': epoch, 'loss': loss})

    # by default, this will save to a new subfolder for files associated
    # with your run, created in wandb.run.dir (which is ./wandb by default)
    wandb.save("mymodel.h5")

config = wandb.config
batch_size = config.batch_size
freeze_epochs = config.freeze_epochs
# finetune_epochs = config.finetune_epochs
lr = config.learning_rate
weight_decay = config.weight_decay
# early_stopping = config.early_stopping
activation = config.activation
# augment = config.augment

model = BiDAF(CHAR_VOCAB_DIM,
              EMB_DIM,
              CHAR_EMB_DIM,
              NUM_OUTPUT_CHANNELS,
              KERNEL_SIZE,
              HIDDEN_DIM,
              device).to(device)

