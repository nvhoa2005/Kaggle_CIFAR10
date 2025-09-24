import os

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EPOCHS = 5
LEARNING_RATE = 0.001