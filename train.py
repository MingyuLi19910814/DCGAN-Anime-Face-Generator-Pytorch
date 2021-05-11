from DCGAN import Model
from dataLoader import createDataLoader

if __name__ == "__main__":
    dataloader = createDataLoader()
    model = Model()
    model.train(dataloader)
