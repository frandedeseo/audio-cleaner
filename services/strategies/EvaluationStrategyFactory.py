from enum import Enum
from .AudioEvaluationStrategy import AudioEvaluationStrategy
from .GeminiEvaluationStrategy import GeminiEvaluationStrategy
from .GeminiProEvaluationStrategy import GeminiProEvaluationStrategy
from .OpenAIEvaluationStrategy import OpenAIEvaluationStrategy


class EvaluationModel(Enum):
    """Modelos disponibles para evaluación."""
    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    OPENAI_AUDIO = "openai-audio"


class EvaluationStrategyFactory:
    """Factory para crear estrategias de evaluación."""
    
    _strategies = {
        EvaluationModel.GEMINI_FLASH: GeminiEvaluationStrategy,
        EvaluationModel.GEMINI_PRO: GeminiProEvaluationStrategy,
        EvaluationModel.OPENAI_AUDIO: OpenAIEvaluationStrategy,
    }
    
    @classmethod
    def create(cls, model: str | EvaluationModel = EvaluationModel.GEMINI_FLASH) -> AudioEvaluationStrategy:
        """Crea instancia de estrategia."""
        if isinstance(model, str):
            try:
                model = EvaluationModel(model.lower())
            except ValueError:
                raise ValueError(
                    f"Modelo '{model}' no soportado. "
                    f"Opciones: {', '.join([m.value for m in EvaluationModel])}"
                )
        
        strategy_class = cls._strategies.get(model)
        if strategy_class is None:
            raise ValueError(f"No hay estrategia para {model}")
        
        return strategy_class()
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        """Retorna lista de modelos disponibles."""
        return [m.value for m in EvaluationModel]

