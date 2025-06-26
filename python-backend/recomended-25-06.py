from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import openai
from openai import AzureOpenAI
import os
from datetime import datetime
import logging
import textwrap
from fastapi import Query
from contextlib import asynccontextmanager
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with aiofiles.open("output.json", "r", encoding="utf-8") as f:
            content = await f.read()
            ads_data = json.loads(content)
        analyzer.train_model(ads_data)
        logger.info("✅ Model trained successfully on startup")
    except FileNotFoundError:
        logger.warning("⚠️ output.json not found. Model will use fallback predictions.")
    except Exception as e:
        logger.error(f"❌ Error training model: {str(e)}")
    yield

app = FastAPI(title="AI Hashtag Predictor API", version="1.0.0",lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Tamil Nadu Politics Hashtag Strategies with language support
TN_POLITICS_STRATEGIES = {
    "piggybacking": {
        "name": "Piggybacking",
        "description": "Ride main election wave",
        "examples": {
            "tamil": ["#மக்களுக்காகDMK", "#தமிழ்நாடுதேர்தல்2026", "#மாற்றத்திற்குAIADMK", "#TNசட்டசபை2026"],
            "english": ["#DMKForPeople", "#TamilNaduElections2026", "#AIADMKForChange", "#TNAssembly2026"],
            "both": ["#DMKForPeople", "#மக்களுக்காகDMK", "#TamilNaduElections2026", "#தமிழ்நாடுதேர்தல்2026"]
        },
        "keywords": ["elections", "vote", "campaign", "democracy", "people", "தேர்தல்", "வாக்கு", "பிரச்சாரம்", "ஜனநாயகம்", "மக்கள்"]
    },
    "hijacking": {
        "name": "Hijacking",
        "description": "Flip opposition slogan",
        "examples": {
            "tamil": ["#ஊழல்இல்லாதமிழ்நாடு", "#வெளிப்படையானஅரசு", "#பொறுப்புக்கூறல்", "#தூய்மையானஅரசியல்"],
            "english": ["#CorruptionFreeTamilNadu", "#TransparentGovt", "#AccountableLeadership", "#CleanPolitics"],
            "both": ["#CorruptionFreeTamilNadu", "#ஊழல்இல்லாதமிழ்நாடு", "#TransparentGovt", "#வெளிப்படையானஅரசு"]
        },
        "keywords": ["corruption", "transparent", "accountable", "clean", "honest", "ஊழல்", "வெளிப்படை", "பொறுப்பு", "தூய்மை"]
    },
    "semantic_shifting": {
        "name": "Semantic Shifting",
        "description": "Own the narrative",
        "examples": {
            "tamil": ["#திராவிடமாதிரி", "#தமிழ்நாடுவளர்ச்சி", "#தென்னிந்தியபெருமை", "#தமிழ்பண்பாடு"],
            "english": ["#DravidianModel", "#TNDevelopment", "#SouthIndianPride", "#TamilCulture"],
            "both": ["#DravidianModel", "#திராவிடமாதிரி", "#TNDevelopment", "#தமிழ்நாடுவளர்ச்சி"]
        },
        "keywords": ["dravidian", "development", "culture", "heritage", "progress", "திராவிட", "வளர்ச்சி", "பண்பாடு", "பாரம்பரியம்", "முன்னேற்றம்"]
    },
    "linking_pairing": {
        "name": "Linking / Pairing",
        "description": "Build identity-based tag networks",
        "examples": {
            "tamil": ["#விவசாயிகளுக்குDMK", "#பெண்களுக்குAIADMK", "#இளைஞர்களுக்குTN", "#திராவிடமதிப்புகள்"],
            "english": ["#DMKForFarmers", "#AIADMKForWomen", "#TNForYouth", "#DravidianValues"],
            "both": ["#DMKForFarmers", "#விவசாயிகளுக்குDMK", "#TNForYouth", "#இளைஞர்களுக்குTN"]
        },
        "keywords": ["farmers", "women", "youth", "workers", "students", "விவசாயிகள்", "பெண்கள்", "இளைஞர்கள்", "தொழிலாளர்கள்", "மாணவர்கள்"]
    },
    "seeding": {
        "name": "Seeding",
        "description": "Start unique, memorable campaigns",
        "examples": {
            "tamil": ["#மக்கள்நம்பிக்கை", "#DMK2026வெற்றி", "#தமிழ்நாடுபுதியகாலம்", "#தமிழ்நாடுமுதலில்"],
            "english": ["#MakkalNambikkai", "#DMK2026Win", "#TNNewEra", "#TamilNaduFirst"],
            "both": ["#MakkalNambikkai", "#மக்கள்நம்பிக்கை", "#TNNewEra", "#தமிழ்நாடுபுதியகாலம்"]
        },
        "keywords": ["unique", "memorable", "campaign", "movement", "change", "தனித்துவம்", "நினைவில்நிற்கும்", "பிரச்சாரம்", "இயக்கம்", "மாற்றம்"]
    },
    "challenges": {
        "name": "Challenges",
        "description": "Drive user-generated content",
        "examples": {
            "tamil": ["#எனூர்எனபெருமை", "#எனதமிழ்நாடு", "#தமிழ்நாடுசவால்", "#உங்கள்பெருமையைகாட்டுங்கள்"],
            "english": ["#EnOoruEnPerumai", "#MyTamilNadu", "#TNChallenge", "#ShowYourPride"],
            "both": ["#MyTamilNadu", "#எனதமிழ்நாடு", "#TNChallenge", "#தமிழ்நாடுசவால்"]
        },
        "keywords": ["challenge", "participate", "show", "share", "my", "சவால்", "பங்கேற்கவும்", "காட்டு", "பகிர்", "என்"]
    },
    "clustering": {
        "name": "Clustering",
        "description": "Balance broad + niche + brand appeal",
        "examples": {
            "tamil": ["#தமிழ்நாடு", "#சென்னை", "#திமுக", "#வளர்ச்சி", "#கல்வி"],
            "english": ["#TamilNadu", "#Chennai", "#DMK", "#Development", "#Education"],
            "both": ["#TamilNadu", "#தமிழ்நாடு", "#Chennai", "#சென்னை", "#Education", "#கல்வி"]
        },
        "keywords": ["broad", "specific", "brand", "location", "sector", "பரந்த", "குறிப்பிட்ட", "பிராண்ட்", "இடம்", "துறை"]
    },
    "mutation": {
        "name": "Mutation",
        "description": "Target sub-regions or groups",
        "examples": {
            "tamil": ["#இளைஞர்களுக்குDMK", "#தென்தமிழ்நாட்டில்AIADMK", "#சென்னைமுதலில்", "#கோவைவாக்குகள்"],
            "english": ["#DMKForYouth", "#AIADMKInSouthTN", "#ChennaiFirst", "#CoimbatoreVotes"],
            "both": ["#DMKForYouth", "#இளைஞர்களுக்குDMK", "#ChennaiFirst", "#சென்னைமுதலில்"]
        },
        "keywords": ["youth", "region", "city", "district", "community", "இளைஞர்", "பகுதி", "நகரம்", "மாவட்டம்", "சமுதாயம்"]
    }
}

# Enhanced sentiment-based hashtag templates with language support
SENTIMENT_HASHTAGS = {
    "positive": {
        "government": {
            "tamil": ["#முன்னேற்றத்தில்TN", "#வளர்ந்துவருதமிழ்நாடு", "#தமிழ்நாடுவெற்றி", "#பெருமைதமிழ்", "#தமிழ்நாடுசாதனைகள்"],
            "english": ["#ProgressiveTN", "#DevelopingTamilNadu", "#TNSuccess", "#ProudTamil", "#TNAchievements"],
            "both": ["#ProgressiveTN", "#முன்னேற்றத்தில்TN", "#TNSuccess", "#தமிழ்நாடுவெற்றி", "#ProudTamil", "#பெருமைதமிழ்"]
        },
        "social": {
            "tamil": ["#பன்முகத்துவத்தில்ஒற்றுமை", "#தமிழ்நாடுகுடும்பம்", "#ஒன்றாகமுடியும்", "#பிரகாசமானஎதிர்காலம்", "#தமிழ்நாடுமேலெழும்புகிறது"],
            "english": ["#UnityInDiversity", "#TNFamily", "#TogetherWeCan", "#BrightFuture", "#TNRising"],
            "both": ["#UnityInDiversity", "#பன்முகத்துவத்தில்ஒற்றுமை", "#TogetherWeCan", "#ஒன்றாகமுடியும்", "#BrightFuture", "#பிரகாசமானஎதிர்காலம்"]
        },
        "development": {
            "tamil": ["#ஸ்மார்ட்TN", "#டிஜிட்டல்தமிழ்நாடு", "#புதுமையானTN", "#நவீனTN", "#தமிழ்நாடுமுன்னணி"],
            "english": ["#SmartTN", "#DigitalTamilNadu", "#InnovativeTN", "#ModernTN", "#TNLeads"],
            "both": ["#SmartTN", "#ஸ்மார்ட்TN", "#DigitalTamilNadu", "#டிஜிட்டல்தமிழ்நாடு", "#TNLeads", "#தமிழ்நாடுமுன்னணி"]
        },
        "cultural": {
            "tamil": ["#தமிழ்பெருமை", "#வளமானபாரம்பரியம்", "#பண்பாட்டுTN", "#தமிழ்பாரம்பரியம்", "#தமிழ்நாடுமதிப்புகள்"],
            "english": ["#TamilPride", "#RichHeritage", "#CulturalTN", "#TamilTradition", "#TNValues"],
            "both": ["#TamilPride", "#தமிழ்பெருமை", "#RichHeritage", "#வளமானபாரம்பரியம்", "#TNValues", "#தமிழ்நாடுமதிப்புகள்"]
        }
    },
    "negative": {
        "opposition": {
            "tamil": ["#தோல்வியடைந்தகொள்கைகள்", "#வெற்றுவாக்குறுதிகள்", "#ஊழல்தலைமை", "#தமிழ்நாடுசிறந்ததுபெறவேண்டும்", "#மாற்றத்திற்குநேரம்"],
            "english": ["#FailedPolicies", "#EmptyPromises", "#CorruptLeadership", "#TNDeservesBetter", "#TimeForChange"],
            "both": ["#FailedPolicies", "#தோல்வியடைந்தகொள்கைகள்", "#EmptyPromises", "#வெற்றுவாக்குறுதிகள்", "#TimeForChange", "#மாற்றத்திற்குநேரம்"]
        },
        "issues": {
            "tamil": ["#தமிழ்நாடுபிரச்சனைகளைசரிசெய்", "#தமிழ்நாடுபோராட்டங்கள்", "#பதிலில்லாதகேள்விகள்", "#இப்போதுபொறுப்புக்கூறல்", "#தமிழ்நாடுநீதி"],
            "english": ["#FixTNIssues", "#TNStruggles", "#UnansweredQuestions", "#AccountabilityNow", "#JusticeForTN"],
            "both": ["#FixTNIssues", "#தமிழ்நாடுபிரச்சனைகளைசரிசெய்", "#TNStruggles", "#தமிழ்நாடுபோராட்டங்கள்", "#JusticeForTN", "#தமிழ்நாடுநீதி"]
        },
        "criticism": {
            "tamil": ["#தமிழ்நாடுமாற்றம்தேவை", "#உடைந்தசிஸ்டம்", "#தோல்வியடைந்தஆட்சி", "#தமிழ்நாடுமேலும்கோருகிறது", "#போதும்போதும்"],
            "english": ["#TNNeedsChange", "#BrokenSystem", "#FailedGovernance", "#TNDemandsMore", "#EnoughIsEnough"],
            "both": ["#TNNeedsChange", "#தமிழ்நாடுமாற்றம்தேவை", "#BrokenSystem", "#உடைந்தசிஸ்டம்", "#EnoughIsEnough", "#போதும்போதும்"]
        },
        "call_to_action": {
            "tamil": ["#எழுந்திருTN", "#தலைவர்களைகேள்வி", "#பதில்கோருங்கள்", "#தமிழ்நாடுஉண்மைபெறவேண்டும்", "#இப்போதுசெயல்படுங்கள்"],
            "english": ["#WakeUpTN", "#QuestionLeaders", "#DemandAnswers", "#TNDeservesTruth", "#ActNow"],
            "both": ["#WakeUpTN", "#எழுந்திருTN", "#QuestionLeaders", "#தலைவர்களைகேள்வி", "#ActNow", "#இப்போதுசெயல்படுங்கள்"]
        }
    }
}

# Pydantic models
class HashtagPredictRequest(BaseModel):
    content: str = Field(..., description="Content to analyze for hashtag prediction")
    max_hashtags: int = Field(default=15, description="Maximum number of hashtags to return")

class ApiConfig(BaseModel):
    provider: str = Field(..., description="API provider (openai or azure)")
    api_key: str = Field(..., description="API key")
    endpoint: Optional[str] = Field(None, description="Azure endpoint (required for Azure)")
    model: str = Field(default="gpt-3.5-turbo", description="Model to use")

class HashtagStrategies(BaseModel):
    piggybacking: bool = False
    hijacking: bool = False
    semantic_shifting: bool = False
    linking_pairing: bool = False
    seeding: bool = False
    challenges: bool = False
    clustering: bool = False
    mutation: bool = False

class PredictHashtagFullRequest(BaseModel):
    content: str
    max_hashtags: int = 15
    config: Optional[ApiConfig] = None
    strategies: Optional[HashtagStrategies] = None
    sentiment: Optional[str] = Field(None, description="positive, negative, or neutral")
    include_tn_politics: bool = False
    language_preference: str = Field(default="both", description="tamil, english, or both")
    enable_sentiment_analysis: bool = Field(default=True, description="Enable advanced sentiment analysis")

class HashtagResult(BaseModel):
    hashtag: str
    score: float
    category: str
    reasoning: str
    strategy: Optional[str] = None
    sentiment: Optional[str] = None
    language: Optional[str] = None

class SentimentAnalysis(BaseModel):
    sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    emotional_tone: Optional[str] = None
    key_emotions: Optional[List[str]] = None
    sentiment_keywords: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    hashtags: List[HashtagResult]
    analysis: Dict
    sentiment_analysis: Optional[SentimentAnalysis] = None
    source: str
    timestamp: str

class HashtagAnalyzer:
    def __init__(self):
        self.hashtag_impact_scores = {}
        self.hashtag_features = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.impact_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=1,
            max_features='auto'
        )
        self.scaler = StandardScaler()
        self.hashtag_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.is_trained = False
        
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        if pd.isna(text) or text is None:
            return []
        hashtags = re.findall(r'#\w+', str(text).lower())
        return [tag.replace('#', '') for tag in hashtags]
    
    def enhanced_sentiment_analysis(self, text: str) -> SentimentAnalysis:
        """Enhanced sentiment analysis with emotional tone detection"""
        # Enhanced word lists with Tamil words
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 
            'success', 'achievement', 'progress', 'development', 'growth', 'improvement',
            'love', 'happy', 'joy', 'celebration', 'victory', 'proud', 'beautiful',
            'நல்ல', 'மிகச்சிறந்த', 'வெற்றி', 'முன்னேற்றம்', 'வளர்ச்சி', 'மகிழ்ச்சி', 
            'அழகான', 'பெருமை', 'சந்தோஷம்', 'கொண்டாட்டம்', 'அற்புதம்'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'failure', 'problem', 'issue',
            'corruption', 'failed', 'broken', 'wrong', 'disappointing', 'poor',
            'sad', 'angry', 'hate', 'disgusting', 'shameful', 'disappointing',
            'கெட்ட', 'மோசம்', 'தோல்வி', 'பிரச்சனை', 'ஊழல்', 'கோபம்', 
            'வருத்தம்', 'வெட்கம்', 'ஏமாற்றம்', 'துக்கம்'
        ]
        
        emotional_words = {
            'joy': ['happy', 'joy', 'celebrate', 'மகிழ்ச்சி', 'சந்தோஷம்', 'கொண்டாட்டம்'],
            'anger': ['angry', 'mad', 'furious', 'கோபம்', 'எரிச்சல்', 'கடுப்பு'],
            'sadness': ['sad', 'depressed', 'sorrow', 'துக்கம்', 'வருத்தம்', 'சோகம்'],
            'fear': ['scared', 'afraid', 'worried', 'பயம்', 'கவலை', 'அச்சம்'],
            'surprise': ['surprised', 'shocked', 'amazed', 'ஆச்சரியம்', 'அதிர்ச்சி', 'வியப்பு'],
            'trust': ['trust', 'believe', 'confident', 'நம்பிக்கை', 'நம்பகம்', 'தன்னம்பிக்கை'],
            'anticipation': ['excited', 'eager', 'hopeful', 'உற்சாகம்', 'ஆவல்', 'நம்பிக்கை']
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        # Basic sentiment scoring
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        positive_score = positive_count / max(total_words, 1)
        negative_score = negative_count / max(total_words, 1)
        neutral_score = 1 - (positive_score + negative_score)
        
        # Emotional tone detection
        emotion_scores = {}
        for emotion, emotion_words_list in emotional_words.items():
            emotion_count = sum(1 for word in emotion_words_list if word in text_lower)
            emotion_scores[emotion] = emotion_count
        
        key_emotions = [emotion for emotion, score in emotion_scores.items() if score > 0]
        key_emotions = sorted(key_emotions, key=lambda x: emotion_scores[x], reverse=True)[:3]
        
        # Determine primary sentiment
        if positive_score > negative_score:
            sentiment = "positive"
            confidence = positive_score / (positive_score + negative_score + 0.1)
            emotional_tone = "optimistic"
        elif negative_score > positive_score:
            sentiment = "negative"
            confidence = negative_score / (positive_score + negative_score + 0.1)
            emotional_tone = "critical"
        else:
            sentiment = "neutral"
            confidence = 0.5
            emotional_tone = "informative"
        
        # Extract sentiment keywords
        sentiment_keywords = []
        if sentiment == "positive":
            sentiment_keywords = [word for word in positive_words if word in text_lower][:5]
        elif sentiment == "negative":
            sentiment_keywords = [word for word in negative_words if word in text_lower][:5]
        
        return SentimentAnalysis(
            sentiment=sentiment,
            confidence=min(confidence, 1.0),
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            emotional_tone=emotional_tone,
            key_emotions=key_emotions,
            sentiment_keywords=sentiment_keywords
        )
    
    def get_strategy_hashtags(self, content: str, strategies: HashtagStrategies, 
                            sentiment: str = None, language_preference: str = "both") -> List[HashtagResult]:
        """Generate hashtags based on selected strategies with language preference"""
        strategy_hashtags = []
        content_lower = content.lower()
        
        rng = np.random.default_rng(seed=42)
        for strategy_key, enabled in strategies.dict().items():
            if not enabled:
                continue
                
            strategy_info = TN_POLITICS_STRATEGIES.get(strategy_key, {})
            strategy_name = strategy_info.get("name", strategy_key)
            
            # Get examples based on language preference
            if language_preference in ["tamil", "english", "both"]:
                examples = strategy_info.get("examples", {}).get(language_preference, [])
            else:
                examples = strategy_info.get("examples", {}).get("both", [])
            
            keywords = strategy_info.get("keywords", [])
            
            # Check if content matches strategy keywords
            keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
            relevance_score = (keyword_matches / len(keywords)) * 100 if keywords else 50
            
            # Add strategy-specific hashtags
            for example in examples[:3]:  # Limit to 3 examples per strategy
                hashtag = example.replace('#', '')
                score = min(relevance_score + rng.integers(10, 30), 100)
                
                # Determine hashtag language
                hashtag_language = self.detect_hashtag_language(hashtag)
                
                strategy_hashtags.append(HashtagResult(
                    hashtag=hashtag,
                    score=score,
                    category="tn_politics",
                    reasoning=f"Strategy: {strategy_name} - {strategy_info.get('description', '')}",
                    strategy=strategy_key,
                    sentiment=sentiment,
                    language=hashtag_language
                ))
        
        return strategy_hashtags
    
    def get_sentiment_hashtags(self, sentiment: str, content: str, 
                             language_preference: str = "both") -> List[HashtagResult]:
        """Generate hashtags based on sentiment with language preference"""
        if sentiment not in SENTIMENT_HASHTAGS:
            return []
        
        sentiment_hashtags = []
        content_lower = content.lower()
        rng = np.random.default_rng(seed=42)
        
        for category, lang_hashtags in SENTIMENT_HASHTAGS[sentiment].items():
            # Get hashtags based on language preference
            if language_preference in ["tamil", "english", "both"]:
                hashtags = lang_hashtags.get(language_preference, [])
            else:
                hashtags = lang_hashtags.get("both", [])
            
            # Check content relevance to category
            category_keywords = {
                "government": ["government", "policy", "minister", "அரசு", "கொள்கை", "அமைச்சர்"],
                "social": ["people", "society", "community", "மக்கள்", "சமூகம்", "சமுதாயம்"],
                "development": ["development", "progress", "growth", "வளர்ச்சி", "முன்னேற்றம்", "வளர்ச்சி"],
                "cultural": ["culture", "tradition", "heritage", "பண்பாடு", "பாரம்பரியம்", "பாரம்பரிய"],
                "opposition": ["opposition", "against", "எதிர்ப்பு", "எதிராக"],
                "issues": ["problem", "issue", "concern", "பிரச்சனை", "பிரச்சினை", "கவலை"],
                "criticism": ["criticism", "critique", "விமர்சனம்", "விமர்சன"],
                "call_to_action": ["action", "act", "do", "செய்", "நடவடிக்கை", "செயல்"]
            }
            
            keywords = category_keywords.get(category, [])
            relevance = sum(1 for keyword in keywords if keyword in content_lower)
            
            if relevance > 0 or category in ["government", "social"]:  # Always include basic categories
                for hashtag in hashtags[:2]:  # Limit to 2 per category
                    tag = hashtag.replace('#', '')
                    score = min(60 + relevance * 10 + rng.integers(5, 25), 95)
                    
                    # Determine hashtag language
                    hashtag_language = self.detect_hashtag_language(tag)
                    
                    sentiment_hashtags.append(HashtagResult(
                        hashtag=tag,
                        score=score,
                        category="sentiment_based",
                        reasoning=f"Sentiment-based ({sentiment}) hashtag for {category} content",
                        sentiment=sentiment,
                        language=hashtag_language
                    ))
        
        return sentiment_hashtags
    
    def detect_hashtag_language(self, hashtag: str) -> str:
        """Detect if hashtag is in Tamil, English, or mixed"""
        # Simple language detection based on character ranges
        tamil_chars = set('அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறனஃ்ாிீுூெேைொோௌ்')
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        hashtag_chars = set(hashtag)
        has_tamil = bool(hashtag_chars & tamil_chars)
        has_english = bool(hashtag_chars & english_chars)
        
        if has_tamil and has_english:
            return "mixed"
        elif has_tamil:
            return "tamil"
        elif has_english:
            return "english"
        else:
            return "unknown"
    
    def calculate_impact_score(self, row: Dict) -> float:
        """Calculate impact score based on multiple metrics"""
        engagement_score = (
            float(row.get('inline_post_engagement', 0)) * 0.3 +
            float(row.get('actions_video_view', 0)) * 0.2 +
            float(row.get('actions_post_reaction', 0)) * 0.2 +
            float(row.get('reach', 0)) * 0.1
        )
        
        cost_efficiency = 0
        spend = float(row.get('spend', 1))
        if spend > 0:
            cost_per_engagement = spend / max(float(row.get('inline_post_engagement', 1)), 1)
            cost_efficiency = 1 / (1 + cost_per_engagement)
        
        ctr_score = float(row.get('ctr', 0)) * 100
        conversion_score = float(row.get('actions_link_click', 0)) * 0.5
        
        impact_score = (
            engagement_score * 0.4 +
            cost_efficiency * 1000 * 0.3 +
            ctr_score * 0.2 +
            conversion_score * 0.1
        )
        
        return impact_score
    
    def analyze_historical_data(self, ads_data: List[Dict]) -> Dict:
        """Analyze historical ads data to find top performing hashtags"""
        hashtag_metrics = defaultdict(lambda: {
            'total_impact': 0,
            'count': 0,
            'avg_engagement': 0,
            'avg_reach': 0,
            'avg_ctr': 0,
            'total_spend': 0,
            'campaigns': set()
        })
        
        for ad in ads_data:
            creative_body = ad.get('creative_details', {}).get('body', '')
            hashtags = self.extract_hashtags(creative_body)
            
            if not hashtags:
                continue
                
            impact_score = self.calculate_impact_score(ad)
            
            # Update hashtag cooccurrence matrix
            for i, tag1 in enumerate(hashtags):
                for j, tag2 in enumerate(hashtags):
                    if i != j:
                        self.hashtag_cooccurrence[tag1][tag2] += 1
            
            # Update metrics for each hashtag
            for hashtag in hashtags:
                metrics = hashtag_metrics[hashtag]
                metrics['total_impact'] += impact_score
                metrics['count'] += 1
                metrics['avg_engagement'] += float(ad.get('inline_post_engagement', 0))
                metrics['avg_reach'] += float(ad.get('reach', 0))
                metrics['avg_ctr'] += float(ad.get('ctr', 0))
                metrics['total_spend'] += float(ad.get('spend', 0))
                metrics['campaigns'].add(ad.get('campaign_name', ''))
        
        # Calculate final metrics
        for hashtag, metrics in hashtag_metrics.items():
            if metrics['count'] > 0:
                metrics['avg_impact'] = metrics['total_impact'] / metrics['count']
                metrics['avg_engagement'] = metrics['avg_engagement'] / metrics['count']
                metrics['avg_reach'] = metrics['avg_reach'] / metrics['count']
                metrics['avg_ctr'] = metrics['avg_ctr'] / metrics['count']
                metrics['avg_spend'] = metrics['total_spend'] / metrics['count']
                metrics['campaign_diversity'] = len(metrics['campaigns'])
                
                self.hashtag_impact_scores[hashtag] = metrics['avg_impact']
        
        return dict(hashtag_metrics)
    
    def train_model(self, ads_data: List[Dict]):
        """Train the hashtag prediction model"""
        logger.info("Training hashtag prediction model...")
        
        # Analyze historical data
        self.analyze_historical_data(ads_data)
        
        # Prepare historical content for TF-IDF
        historical_content = []
        for ad in ads_data:
            content = ad.get('creative_details', {}).get('body', '')
            if content:
                historical_content.append(content)
        
        if historical_content:
            self.tfidf_vectorizer.fit(historical_content)
        
        self.is_trained = True
        logger.info(f"Model trained with {len(self.hashtag_impact_scores)} hashtags")
    
    def predict_hashtags_local(self, content: str, max_hashtags: int = 15, 
                             strategies: HashtagStrategies = None, 
                             sentiment: str = None,
                             language_preference: str = "both") -> List[HashtagResult]:
        """Predict hashtags using local ML model with strategies, sentiment, and language preference"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        all_hashtags = []
        
        # Get strategy-based hashtags
        if strategies:
            strategy_hashtags = self.get_strategy_hashtags(
                content, strategies, sentiment, language_preference
            )
            all_hashtags.extend(strategy_hashtags)
        
        # Get sentiment-based hashtags
        if sentiment and sentiment != "neutral":
            sentiment_hashtags = self.get_sentiment_hashtags(
                sentiment, content, language_preference
            )
            all_hashtags.extend(sentiment_hashtags)
        
        # Get ML-based hashtags
        candidate_hashtags = list(self.hashtag_impact_scores.keys())
        ml_hashtags = []
        
        for hashtag in candidate_hashtags:
            base_score = self.hashtag_impact_scores.get(hashtag, 0)
            
            # Content relevance score
            relevance_score = 0
            hashtag_terms = hashtag.split('_') if '_' in hashtag else [hashtag]
            content_lower = content.lower()
            
            relevance_score = sum(
                1 for term in hashtag_terms 
                if term in content_lower
            ) / len(hashtag_terms)
            
            # Co-occurrence bonus
            cooccurrence_score = 0
            content_hashtags = self.extract_hashtags(content)
            if content_hashtags:
                cooccurrence_score = np.mean([
                    self.hashtag_cooccurrence[hashtag].get(existing_tag, 0)
                    for existing_tag in content_hashtags
                ])
            
            # Language preference filtering
            hashtag_language = self.detect_hashtag_language(hashtag)
            language_bonus = 1.0
            
            if language_preference == "tamil" and hashtag_language == "tamil":
                language_bonus = 1.2
            elif language_preference == "english" and hashtag_language == "english":
                language_bonus = 1.2
            elif language_preference == "both":
                language_bonus = 1.1 if hashtag_language in ["tamil", "english"] else 1.0
            elif language_preference != "both" and hashtag_language != language_preference:
                language_bonus = 0.8
            
            # Combined score
            final_score = (
                base_score * 0.6 +
                relevance_score * 100 * 0.3 +
                cooccurrence_score * 0.1
            ) * language_bonus
            
            if final_score > 5:  # Only include if score is reasonable
                ml_hashtags.append(HashtagResult(
                    hashtag=hashtag,
                    score=round(final_score, 2),
                    category="ml_predicted",
                    reasoning=f"ML model prediction based on historical performance (score: {final_score:.1f})",
                    sentiment=sentiment,
                    language=hashtag_language
                ))
        
        # Sort ML hashtags by score
        ml_hashtags.sort(key=lambda x: x.score, reverse=True)
        
        # Combine all hashtags
        all_hashtags.extend(ml_hashtags[:max_hashtags//2])
        
        # Remove duplicates and sort by score
        seen_hashtags = set()
        unique_hashtags = []
        for hashtag in all_hashtags:
            if hashtag.hashtag not in seen_hashtags:
                seen_hashtags.add(hashtag.hashtag)
                unique_hashtags.append(hashtag)
        
        unique_hashtags.sort(key=lambda x: x.score, reverse=True)
        return unique_hashtags[:max_hashtags]

# Global analyzer instance
analyzer = HashtagAnalyzer()

# Load and train model on startup
@app.on_event("startup")
async def startup_event():
    try:
        # Load historical data
        with open("output.json", "r", encoding="utf-8") as f:
            ads_data = json.load(f)
        
        # Train the model
        analyzer.train_model(ads_data)
        logger.info("✅ Model trained successfully on startup")
    except FileNotFoundError:
        logger.warning("⚠️ output.json not found. Model will use fallback predictions.")
    except Exception as e:
        logger.error(f"❌ Error training model: {str(e)}")

async def predict_with_azure_openai(content: str, config: ApiConfig, max_hashtags: int = 10,
                                  strategies: HashtagStrategies = None, sentiment: str = None,
                                  language_preference: str = "both",
                                  enable_sentiment_analysis: bool = True) -> Dict:
    """Predict hashtags using Azure OpenAI with enhanced features"""
    try:
        if config.provider == "azure":
            client = AzureOpenAI(
                api_key=config.api_key,
                api_version="2025-01-01-preview",
                azure_endpoint=config.endpoint
            )
        else:
            client = openai.OpenAI(api_key=config.api_key)
        
        strategy_context = ""
        if strategies:
            enabled_strategies = [k for k, v in strategies.dict().items() if v]
            if enabled_strategies:
                strategy_context = f"\nFocus on these hashtag strategies: {', '.join(enabled_strategies)}"
                for strategy in enabled_strategies:
                    if strategy in TN_POLITICS_STRATEGIES:
                        strategy_info = TN_POLITICS_STRATEGIES[strategy]
                        strategy_context += f"\n- {strategy_info['name']}: {strategy_info['description']}"
        
        sentiment_context = ""
        if sentiment and sentiment != "neutral":
            sentiment_context = f"\nSentiment focus: Generate {sentiment} hashtags that align with a {sentiment} tone."
        
        language_context = ""
        if language_preference == "tamil":
            language_context = "\nLanguage preference: Prioritize Tamil hashtags (தமிழ் hashtags)"
        elif language_preference == "english":
            language_context = "\nLanguage preference: Prioritize English hashtags"
        else:
            language_context = "\nLanguage preference: Mix of Tamil and English hashtags"
        
        enhanced_sentiment_prompt = ""
        if enable_sentiment_analysis:
            enhanced_sentiment_prompt = """
7. Enhanced sentiment analysis including:
   - Emotional tone detection (optimistic, critical, informative, etc.)
   - Key emotions present in the content
   - Sentiment keywords that influenced the analysis
   - Cultural context for Tamil content"""

        prompt = f"""
You are an expert social media strategist specializing in hashtag optimization for Tamil Nadu government and social campaigns, with a strong focus on Dravidian politics, especially the DMK (Dravida Munnetra Kazhagam) party and its alliances.

Analyze the following content and provide the most effective hashtags for maximum engagement and reach, particularly those that resonate with DMK's political themes, governance efforts, social justice values, and cultural relevance in Tamil Nadu.

Content: "{content}"
{strategy_context}
{sentiment_context}
{language_context}

Please provide:
1. {max_hashtags} most relevant and high-performing hashtags
2. Brief analysis of content themes
3. Engagement potential score (1-100) for each hashtag
4. Reason for each hashtag recommendation
5. Sentiment analysis of the content
6. Most likely political party or alliance the content is associated with (e.g., DMK, AIADMK, BJP, Congress, or Others) based on tone, language, and keywords
{enhanced_sentiment_prompt}
8. Language classification for each hashtag (tamil, english, mixed)

Focus on:
- Tamil Nadu specific hashtags for DMK 
- Government/political campaign hashtags for DMK
- DMK-aligned themes and narratives (social welfare, inclusivity, Tamil pride, etc.)
- Industry-specific and trending social media hashtags for DMK
- Mix of popular and niche hashtags for DMK based on language preference
- Strategic hashtag placement based on selected strategies
- Party-aligned tags when content is politically charged or pro-DMK
- Avoid tags that promote opposition narratives (unless the goal is critical analysis)

Respond in JSON format:
{{
  "sentiment_analysis": {{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.85,
    "positive_score": 0.7,
    "negative_score": 0.1,
    "neutral_score": 0.2,
    "emotional_tone": "optimistic/critical/informative/etc",
    "key_emotions": ["joy", "trust", "anticipation"],
    "sentiment_keywords": ["keyword1", "keyword2"],
    "associated_party": "DMK/AIADMK/BJP/Congress/Others/None",
    "party_confidence": 0.78
  }},
  "analysis": {{
    "themes": ["theme1", "theme2"],
    "keywords": ["keyword1", "keyword2"],
    "content_type": "government/business/social",
    "language": "tamil/english/mixed",
    "target_audience": "description"
  }},
  "hashtags": [
    {{
      "hashtag": "hashtag_without_hash",
      "score": 85,
      "category": "government/industry/trending/tn_politics/sentiment_based",
      "reasoning": "why this hashtag is recommended",
      "source": "meta_ads/trending/strategy/party_specific",
      "strategy": "piggybacking/hijacking/etc or null",
      "sentiment": "positive/negative/neutral or null",
      "language": "tamil/english/mixed"
    }}
  ]
}}
"""

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert social media strategist and hashtag optimization specialist with deep knowledge of Tamil Nadu politics, culture, and social media trends. You understand both Tamil and English languages and can provide culturally appropriate hashtag recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        ai_response = response.choices[0].message.content
        
        try:
            # Remove code block markers if present
            if isinstance(ai_response, str):
                cleaned = ai_response.strip()
                # Remove triple backticks and optional 'json' language tag
                if cleaned.startswith("```"):
                    cleaned = cleaned.lstrip("`")
                    # Remove 'json' if present
                    if cleaned.lower().startswith("json"):
                        cleaned = cleaned[4:].lstrip()
                    # Remove trailing ```
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3].rstrip()
                # Remove leading/trailing whitespace
                cleaned = cleaned.strip()
            else:
                cleaned = ai_response
        
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"AI response is not valid JSON: {ai_response!r}")
            # If JSON parsing fails, extract hashtags from text
            hashtags = []
            if isinstance(ai_response, str) and ai_response.strip():
                hashtag_matches = re.findall(r'#[\w]+', ai_response)
                for i, tag in enumerate(hashtag_matches):
                    hashtags.append({
                        "hashtag": tag.replace('#', ''),
                        "score": max(50, 75 - i * 2),
                        "category": "extracted",
                        "reasoning": "Extracted from AI response",
                        "strategy": None,
                        "sentiment": sentiment,
                        "language": "unknown"
                    })
            else:
                logger.warning("AI response is empty or not a string.")
        
            return {
                "sentiment_analysis": {
                    "sentiment": sentiment or "neutral",
                    "confidence": 0.5,
                    "positive_score": 0.33,
                    "negative_score": 0.33,
                    "neutral_score": 0.34,
                    "emotional_tone": "informative",
                    "key_emotions": [],
                    "sentiment_keywords": []
                },
                "analysis": {
                    "themes": ["general"],
                    "keywords": content.split()[:5],
                    "content_type": "general",
                    "language": "mixed",
                    "target_audience": "general audience"
                },
                "hashtags": hashtags
            }
    
    except Exception as e:
        logger.error(f"Azure OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI API error: {str(e)}")

@app.post("/predict-hashtags", response_model=PredictionResponse)
async def predict_hashtags(request: PredictHashtagFullRequest):
    try:
        # Analyze sentiment
        if request.enable_sentiment_analysis:
            sentiment_analysis = analyzer.enhanced_sentiment_analysis(request.content)
        else:
            sentiment_analysis = analyzer.enhanced_sentiment_analysis(request.content)  # Basic analysis
        
        final_sentiment = request.sentiment or sentiment_analysis.sentiment
        request.max_hashtags = min(request.max_hashtags, 20)  # Limit to max 20 hashtags

        if request.config and request.config.api_key:
            # Use AI API
            logger.info("Using AI API for prediction")
            ai_result = await predict_with_azure_openai(
                request.content, request.config, request.max_hashtags, 
                request.strategies, final_sentiment, request.language_preference,
                request.enable_sentiment_analysis
            )
            
            hashtags = [
                HashtagResult(
                    hashtag=h["hashtag"],
                    score=h["score"],
                    category=h["category"],
                    reasoning=h["reasoning"],
                    strategy=h.get("strategy"),
                    sentiment=h.get("sentiment"),
                    language=h.get("language", "unknown")
                )
                for h in ai_result["hashtags"]
            ]
            
            # Use AI sentiment analysis if available
            ai_sentiment = ai_result.get("sentiment_analysis")
            if ai_sentiment and request.enable_sentiment_analysis:
                sentiment_analysis = SentimentAnalysis(**ai_sentiment)
            
            return PredictionResponse(
                hashtags=hashtags,
                analysis=ai_result["analysis"],
                sentiment_analysis=sentiment_analysis,
                source=f"AI_{request.config.provider.upper()}",
                timestamp=datetime.now().isoformat()
            )
        else:
            # Use local ML model
            logger.info("Using local ML model for prediction")
            hashtags = analyzer.predict_hashtags_local(
                request.content, request.max_hashtags, 
                request.strategies, final_sentiment, request.language_preference
            )
            
            analysis = {
                "themes": ["development", "technology"],
                "keywords": [word for word in request.content.split() if len(word) > 3][:8],
                "content_type": "general",
                "language": "mixed",
                "target_audience": "general public"
            }
            
            return PredictionResponse(
                hashtags=hashtags,
                analysis=analysis,
                sentiment_analysis=sentiment_analysis if request.enable_sentiment_analysis else None,
                source="LOCAL_ML",
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_trained": analyzer.is_trained,
        "hashtag_count": len(analyzer.hashtag_impact_scores),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/top-hashtags")
async def get_top_hashtags(limit: int = 20):
    """Get top performing hashtags from historical data"""
    logger.info(f"/top-hashtags called. is_trained={analyzer.is_trained}, hashtag_count={len(analyzer.hashtag_impact_scores)}")
    if not analyzer.is_trained:
        logger.warning("Model not trained when /top-hashtags called.")
        raise HTTPException(status_code=400, detail="Model not trained")
    if not analyzer.hashtag_impact_scores:
        logger.warning("Model trained but hashtag_impact_scores is empty.")
        raise HTTPException(status_code=404, detail="No hashtag data available. Check if output.json is present and valid.")
    sorted_hashtags = sorted(
        analyzer.hashtag_impact_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return {
        "top_hashtags": [
            {"hashtag": hashtag, "score": score}
            for hashtag, score in sorted_hashtags[:limit]
        ],
        "total_count": len(analyzer.hashtag_impact_scores)
    }

class TopTrendingRequest(BaseModel):
    platform: str = "all"
    count: int = 10
    config: Optional[ApiConfig] = None

@app.post("/top-trending-hashtags")
async def get_top_trending_hashtags(
    request: TopTrendingRequest = Body(...)
):
    """
    Get top trending hashtags for today from Instagram, Facebook, and Twitter using GPT-4o.
    """
    platform_map = {
        "Instagram": "Instagram",
        "Facebook": "Facebook",
        "Twitter": "Twitter",
        "all": "Instagram, Facebook, and Twitter"
    }
    platform_name = platform_map.get(request.platform, "Instagram, Facebook, and Twitter")
    prompt = (
        f"List the top {request.count} popular trending hashtags around in tamilnadu/india related to political parties as well with respect to DMK on {platform_name}. "
        "Return only the hashtags as a JSON array of objects with fields: hashtag, platform, and a short reason for each."
    )

    try:
        config = request.config
        if config and config.api_key:
            if config.provider == "azure":
                client = AzureOpenAI(
                    api_key=config.api_key,
                    api_version="2025-01-01-preview",
                    azure_endpoint=config.endpoint
                )
                model_name = config.model
            else:
                client = openai.OpenAI(api_key=config.api_key)
                model_name = config.model
        else:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_name = "gpt-4o"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a social media analytics expert with deep knowledge of Tamil Nadu politics and trending hashtags."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=800
        )
        ai_response = response.choices[0].message.content

        if isinstance(ai_response, str):
            cleaned = ai_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.lstrip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].lstrip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].rstrip()
            cleaned = cleaned.strip()
        else:
            cleaned = ai_response

        hashtags = json.loads(cleaned)
        return {"platform": platform_name, "top_trending_hashtags": hashtags}

    except Exception as e:
        logger.error(f"Error fetching trending hashtags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching trending hashtags: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)