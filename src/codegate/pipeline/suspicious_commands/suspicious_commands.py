"""
A module for spotting suspicious commands using the embeddings
from our local LLM and a futher ANN categorisier.
"""

import os

import numpy as np  # Add this import
import onnxruntime as ort

from codegate.config import Config
from codegate.inference.inference_engine import LlamaCppInferenceEngine


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

    def load_trained_model(self, file_name):
        """
        Load a trained model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        self.inference_session = ort.InferenceSession(file_name)

    async def compute_embeddings(self, phrases):
        """
        Compute embeddings for a list of phrases.

        Args:
            phrases (list of str): List of phrases to compute embeddings for.

        Returns:
            torch.Tensor: Tensor of embeddings.
        """
        embeddings = await self.inference_engine.embed(self.model_path, phrases)
        return embeddings

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

        input_name = self.inference_session.get_inputs()[0].name
        ort_inputs = {input_name: embeddings}

        # Run the inference session
        ort_outs = self.inference_session.run(None, ort_inputs)

        # Process the output
        prediction = np.argmax(ort_outs[0])
        probability = np.max(ort_outs[0])
        return prediction, probability
