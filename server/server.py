# server.py - Hugging Face modelleri ile AI altyazı üretimi
# pip install fastapi uvicorn transformers torch torchaudio whisper librosa soundfile pydub ffmpeg-python

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import subprocess
import tempfile
import uuid
from typing import List, Optional
import json
from pathlib import Path
import shutil
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForSpeechSeq2Seq, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from datetime import timedelta
import re

app = FastAPI(title="AI Subtitle Generator with Hugging Face", version="2.0.0")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Geçici dosyalar için klasörler
TEMP_DIR = "temp_files"
OUTPUT_DIR = "output_files"
MODELS_DIR = "models"

# Klasörleri oluştur
for dir_name in [TEMP_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(dir_name, exist_ok=True)

class AISubtitleGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.processors = {}
        self.tokenizers = {}
        self.load_models()
    
    def load_models(self):
        """Hugging Face modellerini yükle"""
        try:
            print("🚀 AI modeller yükleniyor...")
            
            # Whisper modeli - Speech to Text
            print("📢 Whisper modeli yükleniyor...")
            self.models['whisper'] = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3",
                cache_dir=MODELS_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.processors['whisper'] = WhisperProcessor.from_pretrained(
                "openai/whisper-large-v3",
                cache_dir=MODELS_DIR
            )
            
            # Çeviri modeli - Helsinki NLP
            print("🌍 Çeviri modelleri yükleniyor...")
            self.translation_pipelines = {
                'en-tr': pipeline("translation", model="Helsinki-NLP/opus-mt-en-tr", device=0 if self.device == "cuda" else -1),
                'tr-en': pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en", device=0 if self.device == "cuda" else -1),
                'en-es': pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=0 if self.device == "cuda" else -1),
                'en-fr': pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr", device=0 if self.device == "cuda" else -1),
                'en-de': pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device=0 if self.device == "cuda" else -1),
                'multi': pipeline("translation", model="facebook/nllb-200-distilled-600M", device=0 if self.device == "cuda" else -1)
            }
            
            # Wav2Vec2 modeli - Alternatif Speech to Text
            print("🎵 Wav2Vec2 modeli yükleniyor...")
            self.speech_pipelines = {
                'tr': pipeline("automatic-speech-recognition", model="microsoft/wav2vec2-large-xlsr-53-turkish", device=0 if self.device == "cuda" else -1),
                'en': pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h-lv60-self", device=0 if self.device == "cuda" else -1),
                'multilingual': pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-multilingual", device=0 if self.device == "cuda" else -1)
            }
            
            print("✅ Tüm modeller başarıyla yüklendi!")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)}")
            raise e
    
    def extract_audio_from_video(self, video_path: str, output_path: str = None):
        """Videodan ses çıkar"""
        if not output_path:
            output_path = os.path.join(TEMP_DIR, f"audio_{uuid.uuid4().hex}.wav")
        
        try:
            # FFmpeg ile ses çıkarma
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-y", output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Ses çıkarma hatası: {str(e)}")
    
    def split_audio_by_silence(self, audio_path: str, silence_threshold: float = -40.0):
        """Sessizliklere göre ses dosyasını böl"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            chunks = []
            
            # Sessizlikleri tespit et
            silence_chunks = []
            chunk_length = 1000  # 1 saniye
            
            for i in range(0, len(audio), chunk_length):
                chunk = audio[i:i + chunk_length]
                if chunk.dBFS < silence_threshold:
                    silence_chunks.append(i)
            
            # Ses parçalarını oluştur
            start = 0
            for silence_start in silence_chunks:
                if silence_start - start > 2000:  # En az 2 saniye
                    chunk = audio[start:silence_start]
                    chunks.append({
                        'audio': chunk,
                        'start_time': start / 1000.0,
                        'end_time': silence_start / 1000.0
                    })
                    start = silence_start + 1000
            
            # Son parçayı ekle
            if start < len(audio):
                chunk = audio[start:]
                chunks.append({
                    'audio': chunk,
                    'start_time': start / 1000.0,
                    'end_time': len(audio) / 1000.0
                })
            
            return chunks
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ses bölme hatası: {str(e)}")
    
    def transcribe_with_whisper(self, audio_path: str, language: str = "tr"):
        """Whisper ile ses metne çevir"""
        try:
            # Ses dosyasını yükle
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Whisper ile işle
            inputs = self.processors['whisper'](
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            # Dil kodu ayarla
            forced_decoder_ids = self.processors['whisper'].get_decoder_prompt_ids(
                language=language, 
                task="transcribe"
            )
            
            # Transkripsiyon
            with torch.no_grad():
                predicted_ids = self.models['whisper'].generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    do_sample=True,
                    temperature=0.0
                )
            
            # Metni decode et
            transcription = self.processors['whisper'].batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper transkripsiyon hatası: {str(e)}")
    
    def transcribe_with_wav2vec2(self, audio_path: str, language: str = "tr"):
        """Wav2Vec2 ile ses metne çevir"""
        try:
            # Dil koduna göre model seç
            if language in self.speech_pipelines:
                pipe = self.speech_pipelines[language]
            else:
                pipe = self.speech_pipelines['multilingual']
            
            # Transkripsiyon
            result = pipe(audio_path)
            return result['text']
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Wav2Vec2 transkripsiyon hatası: {str(e)}")
    
    def translate_text(self, text: str, source_lang: str, target_lang: str):
        """Metni çevir"""
        try:
            # Çeviri modelini seç
            translation_key = f"{source_lang}-{target_lang}"
            
            if translation_key in self.translation_pipelines:
                pipe = self.translation_pipelines[translation_key]
                result = pipe(text)
                return result[0]['translation_text']
            else:
                # Çok dilli model kullan
                pipe = self.translation_pipelines['multi']
                result = pipe(text, src_lang=source_lang, tgt_lang=target_lang)
                return result[0]['translation_text']
                
        except Exception as e:
            print(f"Çeviri hatası: {str(e)}")
            return text  # Hata durumunda orijinal metni döndür
    
    def generate_subtitles(self, video_path: str, source_lang: str = "tr", 
                          target_lang: str = None, model_type: str = "whisper",
                          translate: bool = False, confidence_threshold: float = 0.7):
        """Video için AI altyazı üret"""
        try:
            # Ses çıkar
            audio_path = self.extract_audio_from_video(video_path)
            
            # Ses parçalarına böl
            audio_chunks = self.split_audio_by_silence(audio_path)
            
            subtitles = []
            
            for i, chunk in enumerate(audio_chunks):
                # Geçici ses dosyası oluştur
                temp_chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}_{uuid.uuid4().hex}.wav")
                chunk['audio'].export(temp_chunk_path, format="wav")
                
                try:
                    # Transkripsiyon
                    if model_type == "whisper":
                        text = self.transcribe_with_whisper(temp_chunk_path, source_lang)
                    else:
                        text = self.transcribe_with_wav2vec2(temp_chunk_path, source_lang)
                    
                    # Boş metinleri atla
                    if not text.strip():
                        continue
                    
                    # Çeviri
                    if translate and target_lang and target_lang != source_lang:
                        translated_text = self.translate_text(text, source_lang, target_lang)
                        text = translated_text
                    
                    # Altyazı ekle
                    subtitles.append({
                        'index': len(subtitles) + 1,
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'text': text
                    })
                    
                finally:
                    # Geçici dosyayı temizle
                    if os.path.exists(temp_chunk_path):
                        os.remove(temp_chunk_path)
            
            # Ses dosyasını temizle
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return subtitles
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Altyazı üretme hatası: {str(e)}")
    
    def format_time(self, seconds: float):
        """Zamanı SRT formatına çevir"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def create_srt_file(self, subtitles: List[dict], output_path: str):
        """SRT dosyası oluştur"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sub in subtitles:
                    f.write(f"{sub['index']}\n")
                    f.write(f"{self.format_time(sub['start_time'])} --> {self.format_time(sub['end_time'])}\n")
                    f.write(f"{sub['text']}\n\n")
            
            return output_path
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SRT dosyası oluşturma hatası: {str(e)}")

# Global AI instance
ai_generator = AISubtitleGenerator()

@app.get("/")
async def root():
    return {
        "message": "AI Subtitle Generator with Hugging Face",
        "version": "2.0.0",
        "device": ai_generator.device,
        "models_loaded": len(ai_generator.models)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": ai_generator.device,
        "models_loaded": list(ai_generator.models.keys()),
        "translation_pipelines": list(ai_generator.translation_pipelines.keys()),
        "speech_pipelines": list(ai_generator.speech_pipelines.keys())
    }

@app.post("/generate-ai-subtitles")
async def generate_ai_subtitles(
    video: UploadFile = File(...),
    source_language: str = Form("tr"),
    target_language: str = Form(None),
    model_type: str = Form("whisper"),
    translate: bool = Form(False),
    confidence_threshold: float = Form(0.7),
    burn_subtitles: bool = Form(True)
):
    """AI ile altyazı üret"""
    temp_files = []
    
    try:
        # Video dosyasını kaydet
        temp_video_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}_{video.filename}")
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        temp_files.append(temp_video_path)
        
        # AI altyazı üret
        subtitles = ai_generator.generate_subtitles(
            video_path=temp_video_path,
            source_lang=source_language,
            target_lang=target_language,
            model_type=model_type,
            translate=translate,
            confidence_threshold=confidence_threshold
        )
        
        # SRT dosyası oluştur
        srt_filename = f"subtitles_{uuid.uuid4().hex}.srt"
        srt_path = os.path.join(OUTPUT_DIR, srt_filename)
        ai_generator.create_srt_file(subtitles, srt_path)
        
        if burn_subtitles:
            # Altyazıları videoya göm
            output_filename = f"video_with_subtitles_{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            cmd = [
                "ffmpeg", "-i", temp_video_path,
                "-vf", f"subtitles={srt_path}",
                "-c:v", "libx264", "-c:a", "aac",
                "-y", output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return FileResponse(
                path=output_path,
                filename=output_filename,
                media_type='video/mp4'
            )
        else:
            # Sadece altyazı dosyasını döndür
            return FileResponse(
                path=srt_path,
                filename=srt_filename,
                media_type='text/plain'
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI altyazı üretme hatası: {str(e)}")
    
    finally:
        # Geçici dosyaları temizle
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

@app.post("/translate-subtitles")
async def translate_subtitles(
    srt_file: UploadFile = File(...),
    source_language: str = Form("tr"),
    target_language: str = Form("en")
):
    """Mevcut altyazıları çevir"""
    try:
        # SRT dosyasını oku
        content = await srt_file.read()
        srt_content = content.decode('utf-8')
        
        # SRT'yi parse et
        subtitles = []
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\d+\n|\n*$)'
        matches = re.findall(pattern, srt_content, re.DOTALL)
        
        for match in matches:
            index, start_time, end_time, text = match
            
            # Metni çevir
            translated_text = ai_generator.translate_text(
                text.strip(), 
                source_language, 
                target_language
            )
            
            subtitles.append({
                'index': int(index),
                'start_time': start_time,
                'end_time': end_time,
                'text': translated_text
            })
        
        # Çevrilmiş SRT dosyası oluştur
        output_filename = f"translated_{uuid.uuid4().hex}.srt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(f"{sub['index']}\n")
                f.write(f"{sub['start_time']} --> {sub['end_time']}\n")
                f.write(f"{sub['text']}\n\n")
        
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type='text/plain'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Altyazı çeviri hatası: {str(e)}")

@app.get("/supported-languages")
async def get_supported_languages():
    return {
        "speech_to_text": ["tr", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "ar", "zh"],
        "translation": {
            "helsinki_nlp": ["en-tr", "tr-en", "en-es", "en-fr", "en-de"],
            "nllb": ["tr", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "ar", "zh"]
        }
    }

@app.delete("/cleanup")
async def cleanup_temp_files():
    """Geçici dosyaları temizle"""
    try:
        for folder in [TEMP_DIR, OUTPUT_DIR]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        return {"message": "Geçici dosyalar temizlendi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temizleme hatası: {str(e)}")

if __name__ == "__main__":
    print("🚀 AI Subtitle Generator başlatılıyor...")
    print(f"📱 Cihaz: {ai_generator.device}")
    print("🤖 Hugging Face modelleri hazır")
    print("\n🌐 Server endpoints:")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)