class config_v1:
    MODEL_PATH = 'trained-model/v1/v1.pth.tar'
    LOG_PATH = 'trained-model/v1/train-v1.log'
    BATCH_SIZE = 100
    EPOCHS = 100
    ETA = 1e-2
    TINY = False
    NORMALIZE = True

class config_v2:
    MODEL_PATH = 'trained-model/v2/v2.pth.tar'
    LOG_PATH = 'trained-model/v2/train-v2.log'
    BATCH_SIZE = 100
    EPOCHS = 300
    ETA = 1e-2
    TINY = False
    NORMALIZE = True

class config_v3:
    MODEL_PATH = 'trained-model/v3/v3.pth.tar'
    LOG_PATH = 'trained-model/v3/train-v3.log'
    BATCH_SIZE = 100
    EPOCHS = 300
    ETA = (1e-1, 1e-2, 1e-3)
    TINY = False
    NORMALIZE = True