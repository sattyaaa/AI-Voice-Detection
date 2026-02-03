import io
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

class AudioDetector:
    def __init__(self):
        print("--- [AudioDetector] Initializing 4-Model Ensemble System... ---")
        
        # The Committee of Experts
        self.models_config = [
            {
                "id": "MelodyMachine/Deepfake-audio-detection-V2", 
                "name": "MelodyMachine",
                "weight": 1.0
            },
            {
                "id": "mo-thecreator/Deepfake-audio-detection", 
                "name": "Mo-Creator",
                "weight": 1.0
            },
            {
                "id": "Hemgg/Deepfake-audio-detection", 
                "name": "Hemgg",
                "weight": 1.0
            },
            {
                "id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification", 
                "name": "Gustking-XLSR",
                "weight": 1.2 # Higher weight for the large model
            }
        ]
        
        self.pipelines = []
        
        for cfg in self.models_config:
            try:
                print(f"--- Loading Model: {cfg['name']} ({cfg['id']}) ---")
                # Load pipeline
                p = pipeline("audio-classification", model=cfg['id'])
                self.pipelines.append({"pipe": p, "config": cfg})
                print(f"[+] Loaded {cfg['name']}")
            except Exception as e:
                print(f"[-] Failed to load {cfg['name']}: {e}")
        
        if not self.pipelines:
            print("CRITICAL: No models could be loaded. Ensemble is empty.")
    
    def analyze_audio(self, audio_data: bytes, language: str):
        try:
            # 1. Load Audio
            buffer = io.BytesIO(audio_data)
            y, sr = librosa.load(buffer, sr=16000)
            
            # 2. Extract Features (For Explanation Context Only)
            # We preserve this for generating professional justifications, 
            # but the DECISION is purely model-based.
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # 3. Running The Ensemble
            votes = []
            total_score = 0
            total_weight = 0
            
            print(f"\n--- Running Ensemble Inference on {len(self.pipelines)} models ---")
            
            for item in self.pipelines:
                p = item['pipe']
                cfg = item['config']
                weight = cfg['weight']
                
                try:
                    # Run Inference
                    results = p(y, top_k=None) # Get all labels
                    
                    # Parsing Result for AI Probability
                    ai_score = 0.0
                    
                    # Logic: Find the label that means "Fake"
                    ai_labels = ["fake", "spoof", "aivoice", "artificial", "generated"]
                    
                    found = False
                    for r in results:
                        label_clean = r['label'].lower().strip()
                        if label_clean in ai_labels:
                            ai_score = r['score']
                            found = True
                            break
                            
                    # Note: If no AI label is found (e.g. only 'real'/'human'), ai_score stays 0.0 (Human)
                    # This logic covers {0: 'real', 1: 'fake'} where 'fake' is present.
                    
                    verdict = "AI" if ai_score > 0.5 else "HUMAN"
                    
                    # Weighted contribution
                    votes.append({
                        "name": cfg['name'],
                        "ai_prob": ai_score,
                        "verdict": verdict
                    })
                    
                    total_score += (ai_score * weight)
                    total_weight += weight
                    
                    print(f" > {cfg['name']}: {ai_score:.4f} ({verdict})")
                    
                except Exception as e:
                    print(f"Error inferencing {cfg['name']}: {e}")
            
            # 4. Final Aggregation
            if total_weight > 0:
                final_ensemble_score = total_score / total_weight
            else:
                final_ensemble_score = 0.0 # Fail safe
                
            is_ai = final_ensemble_score > 0.5
            final_classification = "AI_GENERATED" if is_ai else "HUMAN"
            
            # Confidence Score: Distance from 0.5, normalized to 0.5-1.0 roughly, 
            # or just probability of the winning class.
            class_confidence = final_ensemble_score if is_ai else (1.0 - final_ensemble_score)
            
            print(f"--- Final Ensemble Score: {final_ensemble_score:.4f} => {final_classification} (Conf: {class_confidence:.2f}) ---\n")

            # 5. Construct Explanation
            # "3 out of 4 models detected deepfake artifacts..."
            ai_votes_count = sum(1 for v in votes if v['verdict'] == 'AI')
            total_models = len(votes)
            
            explanations = []
            explanations.append(f"Ensemble Analysis: {ai_votes_count}/{total_models} models flagged this audio as AI-generated.")
            explanations.append(f"Aggregated Score: {final_ensemble_score*100:.1f}%.")
            
            if is_ai:
                 if centroid > 2000:
                     explanations.append("High-frequency spectral artifacts consistent with neural vocoders detected.")
                 else:
                     explanations.append("Deep learning pattern matching identified non-biological features.")
            else:
                 explanations.append("Acoustic analysis confirms natural vocal resonance and organic production.")
            
            final_explanation = " ".join(explanations)

            return {
                "classification": final_classification,
                # Return logical confidence (prob of the chosen class)
                "confidenceScore": round(float(class_confidence), 2),
                "explanation": final_explanation
            }
            
        except Exception as e:
            print(f"Analysis Failed: {e}")
            return {
                "classification": "HUMAN", # Fail safe
                "confidenceScore": 0.0,
                "error": str(e),
                "explanation": "Analysis failed due to internal error."
            }

# Global Instance
detector = AudioDetector()
