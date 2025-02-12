# pylint: disable=global-statement, redefined-outer-name, unused-argument
"""
Testing the suspicious commands
"""
import csv
import os

import pytest

from codegate.pipeline.suspicious_commands.suspicious_commands import (
    SuspiciousCommands,
)

try:
    from codegate.pipeline.suspicious_commands.suspicious_commands_trainer import (
        SuspiciousCommandsTrainer,
    )
except ImportError:
    print("Torch not installed")

# Global variables for test data
benign_test_cmds = []
malicious_test_cmds = []
unsafe_commands = []
safe_commands = []
train_data = []

MODEL_FILE = "src/codegate/pipeline/suspicious_commands/simple_nn_model.onnx"
TD_PATH = "tests/data/suspicious_commands"


def read_csv(file_path):
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def setup_module(module):
    """
    Setup function to initialize test data before running tests.
    """
    global benign_test_cmds, malicious_test_cmds, safe_commands
    global unsafe_commands, train_data
    benign_test_cmds = read_csv(f"{TD_PATH}/benign_test_cmds.csv")
    malicious_test_cmds = read_csv(f"{TD_PATH}/malicious_test_cmds.csv")
    unsafe_commands = read_csv(f"{TD_PATH}/unsafe_commands.csv")
    safe_commands = read_csv(f"{TD_PATH}/safe_commands.csv")

    for cmd in benign_test_cmds:
        cmd["label"] = 0
    for cmd in malicious_test_cmds:
        cmd["label"] = 1
    for cmd in safe_commands:
        cmd["label"] = 0
    for cmd in unsafe_commands:
        cmd["label"] = 1

    train_data = safe_commands + unsafe_commands
    import random

    random.shuffle(train_data)


@pytest.fixture
def sc():
    """
    Fixture to initialize the SuspiciousCommands instance and
    load the trained model.

    Returns:
        SuspiciousCommands: Initialized instance with loaded model.
    """
    sc1 = SuspiciousCommands()
    sc1.load_trained_model(MODEL_FILE)
    return sc1


def test_initialization(sc):
    """
    Test the initialization of the SuspiciousCommands instance.
    Args:
        sc (SuspiciousCommands): The instance to test.
    """
    assert sc.inference_session is not None
    assert sc.inference_session is not None


@pytest.mark.asyncio
async def test_train_and_save():
    """
    Test the training process of the SuspiciousCommands instance.
    This test is skipped if the model file is there. Also, the
    training code will need torch installed to run. This is not
    included in the default toml file.
    """
    if os.path.exists(MODEL_FILE):
        return
    sc2 = SuspiciousCommandsTrainer()
    phrases = [cmd["cmd"] for cmd in train_data]
    labels = [cmd["label"] for cmd in train_data]
    await sc2.train(phrases, labels)
    assert sc2.simple_nn is not None
    sc2.save_model(MODEL_FILE)
    assert os.path.exists(MODEL_FILE) is True


@pytest.mark.asyncio
async def test_load_model():
    """
    Test saving and loading the trained model.
    """
    sc2 = SuspiciousCommands()
    sc2.load_trained_model(MODEL_FILE)
    assert sc2.inference_session is not None
    class_, prob = await sc2.classify_phrase("brew list")
    assert 0 == class_
    assert prob > 0.7


def check_results(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    assert precision > 0.8
    assert recall > 0.7
    assert f1_score > 0.8


@pytest.mark.asyncio
async def test_classify_phrase(sc):
    """
    Test the classification of phrases as suspicious or not.

    Args:
        sc (SuspiciousCommands): The instance to test.
    """
    tp = tn = fp = fn = 0
    for command in benign_test_cmds:
        prediction, _ = await sc.classify_phrase(command["cmd"])
        if prediction == 0:
            tn += 1
        else:
            fn += 1

    for command in malicious_test_cmds:
        prediction, _ = await sc.classify_phrase(command["cmd"])
        if prediction == 1:
            tp += 1
        else:
            fp += 1
    check_results(tp, tn, fp, fn)


@pytest.mark.asyncio
async def test_classify_phrase_confident(sc):
    """
    Test the classification of phrases as suspicious or not.
    Add a level of confidence to the results.

    Args:
        sc (SuspiciousCommands): The instance to test.
    """
    confidence = 0.9
    tp = tn = fp = fn = 0
    for command in benign_test_cmds:
        prediction, prob = await sc.classify_phrase(command["cmd"])
        if prob > confidence:
            if prediction == 0:
                tn += 1
            else:
                fn += 1
        else:
            print(f"{command['cmd']} {prob} {prediction} 0")

    for command in malicious_test_cmds:
        prediction, prob = await sc.classify_phrase(command["cmd"])
        if prob > confidence:
            if prediction == 1:
                tp += 1
            else:
                fp += 1
        else:
            print(f"{command['cmd']} {prob} {prediction} 1")
    check_results(tp, tn, fp, fn)
