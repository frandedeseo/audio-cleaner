from fastapi import UploadFile
from .TextAudioEquivalentService import TextAudioEquivalentService
from .WpmService import WpmService
from .NoiseReduceService import NoiseReduceService
from .GptService import GptService
from repository.Load import Load # Corrected "respository" to "repository"
from .VoiceSeparatorService import VoiceSeparatorService

class EvaluationService:
    def __init__(self):
        self.text_audio = TextAudioEquivalentService()
        self.wpm = WpmService()
        self.gpt = GptService()
        self.nr = NoiseReduceService()
        self.vs = VoiceSeparatorService()
        self.repo = Load()

    async def handle(self, text: str, audio: UploadFile):
        audio_bytes = await audio.read()

        audio_bytes = self.vs.separar_voces(audio_bytes)
        audio_bytes = self.nr.reducir_ruido(audio_bytes)

        match_info = await self.text_audio.verify(audio_bytes, text)
        if not match_info['match']:
            return {
                "error": "El texto proporcionado no coincide con el audio.",
                **match_info
            }
        wpm_value = self.wpm.calculate(audio_bytes, text)
        evaluation = await self.gpt.evaluate(text, wpm_value, audio_bytes)
        record = {
            **match_info,
            'palabras_por_minuto': round(wpm_value, 2),
            'evaluacion': evaluation
        }
        saved = self.repo.save(text, audio.filename, audio_bytes, record)
        return evaluation
