from Mnist.CNN.DataPrep import create_dataset
from Mnist.CNN.VisualizeData import draw_samples


def run() -> None:
    digits = create_dataset()
    draw_samples(dataset=digits, samples_number=8)


