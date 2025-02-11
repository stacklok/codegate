from enum import Enum


class ClientType(Enum):
    """
    Enum of supported client types
    """

    GENERIC = "generic"  # Default client type when no specific client is detected
    CLINE = "cline"  # Cline client
    KODU = "kodu"  # Kodu client
    COPILOT = "copilot"  # Copilot client
    OPEN_INTERPRETER = "open_interpreter"  # Open Interpreter client
    AIDER = "aider"  # Aider client
    CONTINUE = "continue"  # Continue client
