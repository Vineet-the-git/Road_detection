class Config():
    def __init__(self):
        self.path_dataset = "/home/vineet/Desktop/Road_detection/dataset"
        self.prediction = "/home/vineet/Desktop/Road_detection/predictions"
        self.checkpoint = "/home/vineet/Desktop/Road_detection/experiments"
        self.mode = "train"


        self.max_iter = 5000
        self.learning_rate = 0.01
        self.lr_decay = 0.1
        self.lr_decay_iter = 1000
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.batch_size = 1

        self.check_interval = 100
        self.pred_interval = 10