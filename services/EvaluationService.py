from fastapi import UploadFile
from .TextAudioEquivalentService import TextAudioEquivalentService
from .WpmService import WpmService
from .strategies import EvaluationStrategyFactory
#from .NoiseReduceService import NoiseReduceService
#from .VoiceSeparatorService import VoiceSeparatorService

class EvaluationService:
    def __init__(self):
        self.text_audio = TextAudioEquivalentService()
        self.wpm = WpmService()
        #self.nr = NoiseReduceService()
        #self.vs = VoiceSeparatorService()

    async def handle(self, text: str, audio: UploadFile, model: str = "gemini-flash"):
        """Evalúa lectura con modelo especificado."""
        audio_bytes = await audio.read()

        match_info = await self.text_audio.verify(audio_bytes, text)
        if not match_info['match']:
            return {
                "error": "El texto proporcionado no coincide con el audio.",
                **match_info
            }
        
        wpm_value = self.wpm.calculate(audio_bytes, text)
        
        strategy = EvaluationStrategyFactory.create(model)
        evaluation = await strategy.evaluate(text, wpm_value, audio_bytes)
        
        return {
            **match_info,
            'palabras_por_minuto': round(wpm_value, 2),
            'modelo': model,
            'evaluacion': evaluation
        }
