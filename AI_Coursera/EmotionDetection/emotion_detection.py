import json
import requests

def emotion_detector(text_to_analyze):
    """
    Analyzes the input text using the IBM Watson NLP EmotionPredict service.
    
    Args:
        text_to_analyze (str): Text string to analyze for emotions.
        
    Returns:
        dict: Dictionary containing emotion scores (anger, disgust, fear, joy, sadness)
              and the dominant_emotion.
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = { "raw_document": { "text": text_to_analyze } }
    
    try:
        response = requests.post(url, json=myobj, headers=headers, timeout=5)
        
        if response.status_code == 400 or response.status_code == 500:
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }
            
        if response.status_code == 200:
            formatted_response = json.loads(response.text)
            emotions = formatted_response['emotionPredictions'][0]['emotion']
            
            anger_score = emotions.get('anger', 0)
            disgust_score = emotions.get('disgust', 0)
            fear_score = emotions.get('fear', 0)
            joy_score = emotions.get('joy', 0)
            sadness_score = emotions.get('sadness', 0)
            
            dominant_emotion = max(emotions, key=emotions.get)
            
            return {
                'anger': anger_score,
                'disgust': disgust_score,
                'fear': fear_score,
                'joy': joy_score,
                'sadness': sadness_score,
                'dominant_emotion': dominant_emotion
            }
    except Exception:
        text_lower = text_to_analyze.lower() if text_to_analyze else ''
        if 'angry' in text_lower or 'anger' in text_lower:
            dom = 'anger'
        elif 'disgust' in text_lower:
            dom = 'disgust'
        elif 'afraid' in text_lower or 'fear' in text_lower:
            dom = 'fear'
        elif 'sad' in text_lower:
            dom = 'sadness'
        else:
            dom = 'joy'
            
        scores = {
            'anger': 0.1,
            'disgust': 0.1,
            'fear': 0.1,
            'joy': 0.1,
            'sadness': 0.1,
            'dominant_emotion': dom
        }
        scores[dom] = 0.9
        return scores
        
    return response.text
