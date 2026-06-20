"""
Módulo de estrategias de evaluación de audio.
Patrón Strategy para múltiples modelos de IA.
"""

from .AudioEvaluationStrategy import AudioEvaluationStrategy
from .GeminiEvaluationStrategy import GeminiEvaluationStrategy
from .GeminiProEvaluationStrategy import GeminiProEvaluationStrategy
from .OpenAIEvaluationStrategy import OpenAIEvaluationStrategy
from .EvaluationStrategyFactory import EvaluationStrategyFactory, EvaluationModel

__all__ = [
    "AudioEvaluationStrategy",
    "GeminiEvaluationStrategy",
    "GeminiProEvaluationStrategy",
    "OpenAIEvaluationStrategy",
    "EvaluationStrategyFactory",
    "EvaluationModel",
]

