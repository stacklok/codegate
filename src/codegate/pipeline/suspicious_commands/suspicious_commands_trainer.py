"""
A module for spotting suspicious commands using the embeddings
from our local LLM and a futher ANN categorisier.
"""

import os

import torch
from torch import nn

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine
from codegate.pipeline.suspicious_commands.suspicious_commands import SuspiciousCommands


class SimpleNN(nn.Module):
    """
    A simple neural network with one hidden layer.

    Attributes:
        network (nn.Sequential): The neural network layers.
    """

    def __init__(self, input_dim=1, hidden_dim=128, num_classes=2):
        """
        Initialize the SimpleNN model. The default args should be ok,
        but the input_dim must match the incoming training data.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            num_classes (int): Number of output classes.
        """
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.network(x)


class SuspiciousCommandsTrainer(SuspiciousCommands):
    """
    Class to handle suspicious command detection using a neural network.

    Attributes:
        model_path (str): Path to the model.
        inference_engine (LlamaCppInferenceEngine): Inference engine for embedding.
        simple_nn (SimpleNN): Neural network model.
    """

    _instance = None

    @staticmethod
    def get_instance(model_file=None):
        """
        Get the singleton instance of SuspiciousCommands. Initialize and load
        from file on the first call if it has not been done.

        Args:
            model_file (str, optional): The file name to load the model from.

        Returns:
            SuspiciousCommands: The singleton instance.
        """
        if SuspiciousCommands._instance is None:
            SuspiciousCommands._instance = SuspiciousCommands()
            if model_file is None:
                current_file_path = os.path.dirname(os.path.abspath(__file__))
                model_file = os.path.join(current_file_path, "simple_nn_model.onnx")
            SuspiciousCommands._instance.load_trained_model(model_file)
        return SuspiciousCommands._instance

    def __init__(self):
        """
        Initialize the SuspiciousCommands class.
        """
        conf = Config.get_config()
        if conf and conf.model_base_path and conf.embedding_model:
            self.model_path = f"{conf.model_base_path}/{conf.embedding_model}"
        else:
            self.model_path = ""
        self.inference_engine = LlamaCppInferenceEngine()
        self.simple_nn = None  # Initialize to None, will be created in train

    async def train(self, phrases, labels):
        """
        Train the neural network with given phrases and labels.

        Args:
            phrases (list of str): List of phrases to train on.
            labels (list of int): Corresponding labels for the phrases.
        """
        embeds = await self.inference_engine.embed(self.model_path, phrases)
        if isinstance(embeds[0], list):
            embedding_dim = len(embeds[0])
        else:
            raise ValueError("Embeddings should be a list of lists of floats")

        self.simple_nn = SimpleNN(input_dim=embedding_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.simple_nn.parameters(), lr=0.001)

        # Training loop
        for _ in range(100):
            for data, label in zip(embeds, labels):
                data = torch.FloatTensor(data)  # convert to tensor
                label = torch.LongTensor([label])  # convert to tensor

                optimizer.zero_grad()
                outputs = self.simple_nn(data)
                loss = criterion(outputs.unsqueeze(0), label)
                loss.backward()
                optimizer.step()

    def save_model(self, file_name):
        """
        Save the trained model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        if self.simple_nn is not None:
            # Create a dummy input with the correct embedding dimension
            dummy_input = torch.randn(1, self.simple_nn.network[0].in_features)
            torch.onnx.export(
                self.simple_nn,
                dummy_input,
                file_name,
                input_names=["input"],
                output_names=["output"],
            )
