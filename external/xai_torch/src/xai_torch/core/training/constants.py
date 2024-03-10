"""Holds the constants related to training."""


class TrainingStage:
    train = "train"
    val = "val"
    test = "test"
    predict = "predict"

class GANStage:
    disc_real = 'disc_real'
    disc_fake = 'disc_fake'
    gen = 'gen'
