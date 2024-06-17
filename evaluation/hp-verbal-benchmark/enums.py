"""Enums for Chat Templates"""

from enum import Enum

class AI_SWE_GPTSW3(Enum):
    eos_token = "<|endoftext|>"
    bos_token = "<s>"
    mapper = {"system":None, "user":"User:", "assistant":"Bot:"}

class CUSTOM_GPTSW3(Enum):
    eos_token = "<|endoftext|>"
    bos_token = "<s>"
    mapper = {"system":None, "user":"USER:", "assistant":"ASSISTANT:"} 