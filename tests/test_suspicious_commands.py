# pylint: disable=global-statement, redefined-outer-name, unused-argument
"""
Testing the suspicious commands
"""
import os

import pandas as pd
import pytest

from codegate.pipeline.suspicious_commands.suspicious_commands import (
    SuspiciousCommands,
)

# Global variables for test data
benign_test_cmds = malicious_test_cmds = pd.DataFrame()
unsafe_commands = safe_commands = train_data = pd.DataFrame()

MODEL_FILE = "src/codegate/pipeline/suspicious_commands/simple_nn_model.h5"
TD_PATH = "tests/data/suspicious_commands"


def setup_module(module):
    """
    Setup function to initialize test data before running tests.
    """
    global benign_test_cmds, malicious_test_cmds, safe_commands
    global unsafe_commands, train_data, train_data
    benign_test_cmds = pd.read_csv(f"{TD_PATH}/benign_test_cmds.csv")
    malicious_test_cmds = pd.read_csv(f"{TD_PATH}/malicious_test_cmds.csv")
    unsafe_commands = pd.read_csv(f"{TD_PATH}/unsafe_commands.csv")
    safe_commands = pd.read_csv(f"{TD_PATH}/safe_commands.csv")
    benign_test_cmds["label"] = 0
    malicious_test_cmds["label"] = 1
    safe_commands["label"] = 0
    unsafe_commands["label"] = 1
    train_data = pd.concat([safe_commands, unsafe_commands])
    train_data = train_data.sample(frac=1).reset_index(drop=True)


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
    assert sc.inference_engine is not None
    assert sc.simple_nn is not None


@pytest.mark.asyncio
async def test_train():
    """
    Test the training process of the SuspiciousCommands instance.
    """
    if os.path.exists(MODEL_FILE):
        return
    sc2 = SuspiciousCommands()
    phrases = train_data["cmd"].tolist()
    labels = train_data["label"].tolist()
    await sc2.train(phrases, labels)
    assert sc2.simple_nn is not None
    sc2.save_model(MODEL_FILE)
    assert os.path.exists(MODEL_FILE) is True


@pytest.mark.asyncio
async def test_save_and_load_model():
    """
    Test saving and loading the trained model.
    """
    sc2 = SuspiciousCommands()
    sc2.load_trained_model(MODEL_FILE)
    assert sc2.simple_nn is not None
    class_, prob = await sc2.classify_phrase("brew list")
    assert 0 == class_
    assert prob > 0.7
    sc2.save_model(MODEL_FILE)


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
    for command in benign_test_cmds["cmd"]:
        prediction, _ = await sc.classify_phrase(command)
        if prediction == 0:
            tn += 1
        else:
            fn += 1

    for command in malicious_test_cmds["cmd"]:
        prediction, _ = await sc.classify_phrase(command)
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
    for command in benign_test_cmds["cmd"]:
        prediction, prob = await sc.classify_phrase(command)
        if prob > confidence:
            if prediction == 0:
                tn += 1
            else:
                fn += 1
        else:
            print(f"{command} {prob} {prediction} 0")

    for command in malicious_test_cmds["cmd"]:
        prediction, prob = await sc.classify_phrase(command)
        if prob > confidence:
            if prediction == 1:
                tp += 1
            else:
                fp += 1
        else:
            print(f"{command} {prob} {prediction} 1")
    check_results(tp, tn, fp, fn)
