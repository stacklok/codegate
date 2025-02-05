"""
A module for spotting suspicious commands using the embeddings
from our local LLM and a futher ANN categorisier.
"""

import os

import torch
from torch import nn

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine


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


class SuspiciousCommands:
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
                model_file = os.path.join(current_file_path, "simple_nn_model.pt")
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
        self.simple_nn = SimpleNN()

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
            for data, label in zip(torch.FloatTensor(embeds), torch.LongTensor(labels)):
                optimizer.zero_grad()
                outputs = self.simple_nn(data)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

    def save_model(self, file_name):
        """
        Save the trained model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        if self.simple_nn is not None:
            torch.save(  # nosec
                {
                    "model_state_dict": self.simple_nn.state_dict(),
                    "input_dim": self.simple_nn.network[0].in_features,
                },
                file_name,
                pickle_protocol=4,  # Use a safer pickle protocol
            )

    def load_trained_model(self, file_name, weights_only=True):
        """
        Load a trained model from a file.

        Args:
            file_name (str): The file name to load the model from.
            weights_only (bool): Whether to load only the weights.
        """
        # Ensure the file being loaded is trusted
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Model file {file_name} does not exist.")

        checkpoint = torch.load(  # nosec
            file_name, map_location=torch.device("cpu"), weights_only=weights_only
        )
        input_dim = checkpoint["input_dim"]
        self.simple_nn = SimpleNN(input_dim=input_dim)
        self.simple_nn.load_state_dict(checkpoint["model_state_dict"])

    async def compute_embeddings(self, phrases):
        """
        Compute embeddings for a list of phrases.

        Args:
            phrases (list of str): List of phrases to compute embeddings for.

        Returns:
            torch.Tensor: Tensor of embeddings.
        """
        embeddings = []
        embeddings = await self.inference_engine.embed(self.model_path, phrases)
        return torch.tensor(embeddings)

    async def classify_phrase(self, phrase, embeddings=None):
        """
        Classify a single phrase as suspicious or not.

        Args:
            phrase (str): The phrase to classify.
            embeddings (torch.Tensor, optional): Precomputed embeddings for
            the phrase.

        Returns:
            tuple: The predicted class (0 or 1) and its probability.
        """
        if embeddings is None:
            embeddings = await self.compute_embeddings([phrase])
        with torch.no_grad():
            outputs = self.simple_nn(embeddings)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prob, predicted = torch.max(probabilities, 1)
        return predicted.item(), prob.item()
