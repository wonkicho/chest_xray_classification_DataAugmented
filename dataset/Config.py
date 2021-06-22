import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 64
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
ACCUM_ITER = 2
VERBOSE_STEP = 1
