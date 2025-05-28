import io
import math
import os
import sys
import soundfile as sf

from whisper_online import ASRBase


class AssemblyAIASR(ASRBase):
    """ASR backend using AssemblyAI's API."""

    def __init__(self, lan=None, api_key=None, model="universal", logfile=sys.stderr):
        self.logfile = logfile
        self.original_language = None if lan == "auto" else lan
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if self.api_key is None:
            raise ValueError("AssemblyAI API key must be provided")
        self.model = model
        self.load_model()

    def load_model(self, *args, **kwargs):
        import assemblyai as aai
        aai.settings.api_key = self.api_key
        self.client = aai.Client()
        self.transcribed_seconds = 0

    def transcribe(self, audio_data, init_prompt=""):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        options = {
            "speech_model": self.model,
            "punctuate": True,
            "format_text": False,
        }
        if self.original_language:
            options["language_code"] = self.original_language
        if init_prompt:
            options["prompt"] = init_prompt

        transcript = self.client.transcribe(buffer, **options)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        return transcript

    def ts_words(self, result):
        return [
            (w.start / 1000.0, w.end / 1000.0, w.text)
            for w in getattr(result, "words", [])
        ]

    def segments_end_ts(self, res):
        return [w.end / 1000.0 for w in getattr(res, "words", [])]

    def use_vad(self):
        pass  # not supported

    def set_translate_task(self):
        raise NotImplementedError("AssemblyAI API does not support translation")
