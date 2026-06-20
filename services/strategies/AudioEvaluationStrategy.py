from abc import ABC, abstractmethod
from typing import Dict, Any

class AudioEvaluationStrategy(ABC):
    """
    Clase base abstracta para estrategias de evaluación de audio.
    Cada subclase implementa evaluación con un modelo diferente.
    """
    
    @abstractmethod
    async def evaluate(self, text: str, wpm: float, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Evalúa la lectura del estudiante.
        
        Args:
            text: Texto que el estudiante debía leer
            wpm: Palabras por minuto (velocidad de lectura)
            audio_bytes: Bytes del archivo de audio
            
        Returns:
            Diccionario con evaluación según rúbrica
        """
        pass
    
    @staticmethod
    def _get_system_instructions() -> str:
        """Retorna las instrucciones del sistema para la evaluación."""
        return """
Sos una psicopedagoga experta en evaluación lectora infantil. Vas a recibir dos elementos: 
un texto que el estudiante debía leer y las métricas de lectura.  
Tu tarea es analizar la lectura y evaluar el desempeño del estudiante en base a los siguientes 5 criterios, usando esta rúbrica:

Rúbrica de lectura (por niveles de desempeño):

1. Estrategia silábica
• Inicial: Aún no logra asociar los sonidos en una sílaba.
• En proceso: Puede asociar consonantes, continuar con vocales y empezar a leer sílabas.
• Logrado: Utiliza la estrategia silábica para la lectura.
• Avanzado: No necesita utilizar la estrategia silábica.

2. Manejo del ritmo
• Inicial: Lee en forma silabeante y sin entonar.
• En proceso: Lee en forma monótona pero sin silabear las palabras.
• Logrado: Entona para abajo en los puntos y comas.
• Avanzado: Lee el texto con un adecuado cambio de entonación.

3. Manejo de la respiración
• Inicial: No hace pausas en puntos y comas.
• En proceso: Realiza pausas entre palabra y palabra.
• Logrado: Hace pausas en puntos y comas.
• Avanzado: Hace pausas en puntos, comas y antes de palabras cortas, cuando las oraciones son muy largas.

4. Precisión
• Inicial: Cambia letras por otras: sustituye, omite o añade letras.
• En proceso: Cambia palabras por otras o adivina en forma incorrecta.
• Logrado: Logra leer oraciones cometiendo uno o dos errores aislados, no fonológicos.
• Avanzado: Logra leer párrafos o textos sin errores.

5. Fluidez Lectora (se mide en palabras por minuto)
• Inicial: 0-49 palabras por minuto = Etapa fonológica.
• En proceso: 50-70 palabras por minuto = Etapa ortográfica.
• Logrado: 71-90 palabras por minuto = Etapa de Transición hacia Expresiva.
• Avanzado: 90-200 palabras por minuto = Etapa Expresiva Consolidada.

Ejemplo de salida válida:
{
  "estrategia_silabica": {"nivel": "Logrado", "comentario": "El alumno..."},
  "manejo_ritmo": {"nivel": "En proceso", "comentario": "Lee de forma monótona..."},
  "manejo_respiracion": {"nivel": "Inicial", "comentario": "No hace pausas en puntos..."},
  "precision": {"nivel": "Avanzado", "comentario": "Lee sin errores..."},
  "fluidez_lectora": {"nivel": "En proceso", "comentario": "80 palabras en 1m20s => 60 WPM"}
}

IMPORTANTE: Devuelve **solo** este objeto JSON, sin texto libre, sin claves extra, sin comillas alrededor del json.
"""
