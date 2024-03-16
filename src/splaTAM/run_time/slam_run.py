from .base import Evaluator, ModelTrainer, DatasetLoader


class SLAMSystem:
    def __init__(self, config):
        self.config = config
        self.dataset_loader = DatasetLoader(config)
        self.model_trainer = ModelTrainer(config)
        self.evaluator = Evaluator(config)

    def run(self):
        data = self.dataset_loader.load_data()
        preprocessed_data = self.dataset_loader.preprocess_data(data)
        self.model_trainer.train()
        self.evaluator.evaluate()



