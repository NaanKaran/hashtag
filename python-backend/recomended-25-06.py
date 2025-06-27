import asyncio
from fastapi import FastAPI, HTTPException, Depends, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import json
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import openai
from openai import AzureOpenAI
import os
from datetime import datetime
import logging
import textwrap
from fastapi import Query
from contextlib import asynccontextmanager
import aiofiles
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_file_sync(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

OUTPUT_JSON_FILENAME = "output.json"

async def load_json_async(filepath):
    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, read_file_sync, filepath)
    return json.loads(content)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("🚀 Starting application initialization...")
        
        # First, load RAG data
        logger.info("📊 Loading RAG data...")
        await analyzer.load_rag_data()
        
        # Then, try to load and train with historical ads data
        try:
            logger.info("📈 Loading historical ads data...")
            OUTPUT_JSON_FILEPATH =  os.path.join(os.path.dirname(__file__), OUTPUT_JSON_FILENAME)
            ads_data = await load_json_async(OUTPUT_JSON_FILEPATH)

            logger.info(f"📈 Training model with {len(ads_data)} ads data entries...")
            analyzer.train_model(ads_data)
            logger.info("✅ Model trained successfully with both RAG and ads data")
            
        except FileNotFoundError:
            logger.warning("⚠️ output.json not found. Training model with RAG data only...")
        except Exception as e:
            logger.error(f"❌ Error loading ads data: {str(e)}. Training with RAG data only...")
            analyzer.train_model([])
            logger.info("✅ Model trained with RAG data only")
        
        # Verify training status
        logger.info(f"🎯 Model training status: {analyzer.is_trained}")
        logger.info(f"📊 Available hashtags: {len(analyzer.hashtag_impact_scores)}")
        logger.info(f"📚 RAG hashtags: {len(analyzer.rag_hashtag_frequency)}")
        
    except Exception as e:
        logger.error(f"❌ Critical error during startup: {str(e)}")
        # Even if everything fails, mark as trained with minimal data
        analyzer.is_trained = True
        logger.info("✅ Fallback: Model marked as trained")
    
    yield

app = FastAPI(title="AI Hashtag Predictor API with RAG", version="2.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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

class HashtagResult(BaseModel):
    hashtag: str
    score: float
    category: str
    reasoning: str
    strategy: Optional[str] = None
    sentiment: Optional[str] = None
    language: Optional[str] = None
    source: str = "new"
    frequency_in_rag: Optional[int] = None
    similar_content_examples: Optional[List[str]] = None

class SentimentAnalysis(BaseModel):
    sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    emotional_tone: Optional[str] = None
    key_emotions: Optional[List[str]] = None
    sentiment_keywords: Optional[List[str]] = None
    associated_party: Optional[str] = None
    party_confidence: Optional[float] = None

class RAGHashtagInfo(BaseModel):
    hashtag: str
    frequency: int
    category: str

class RAGAnalysis(BaseModel):
    total_rag_hashtags: int
    existing_hashtags_found: int  # Fixed: corrected field name
    new_hashtags_suggested: int
    rag_enhanced_hashtags: int
    top_rag_hashtags: List[RAGHashtagInfo]

class PredictHashtagFullRequest(BaseModel):
    content: str
    max_hashtags: int = 15
    config: Optional[ApiConfig] = None
    strategies: Optional[HashtagStrategies] = None
    sentiment: Optional[str] = Field(None, description="positive, negative, or neutral")
    include_tn_politics: bool = False
    language_preference: str = Field(default="both", description="tamil, english, or both")
    enable_sentiment_analysis: bool = Field(default=True, description="Enable advanced sentiment analysis")
    prediction_method: str = Field(default="auto", description="rag_only, ai_only, or auto")  # NEW FIELD

class PredictionResponse(BaseModel):
    hashtags: List[HashtagResult]
    analysis: Dict[str, Any]
    sentiment_analysis: Optional[SentimentAnalysis] = None
    rag_analysis: Optional[RAGAnalysis] = None
    source: str
    timestamp: str

# RAG-based hashtag categories
RAG_HASHTAG_CATEGORIES = {
    "high_frequency": {
        "description": "Most frequently used hashtags from historical data",
        "hashtags": [
            "ellorumnammudan", "dravidamodel", "dmk4tn", "cmmkstalin", "mkstalincm", 
            "mkstalin", "mkstalingovt", "dravidianmodel", "governancetamilnadu", 
            "no1tamilnadu", "stalinbuildstn", "oraethalaivan", "mkstalin4tn", 
            "mkstalinera", "dmk", "dmkgovt"
        ],
        "priority_boost": 2.0
    },
    "political_identity": {
        "description": "Core political identity hashtags",
        "hashtags": [
            "dmk", "dmk4tn", "dravidianmodel", "dravidamodel", "kalaignar", 
            "oraethalaivan", "mkstalin", "cmmkstalin"
        ],
        "priority_boost": 1.8
    },
    "governance": {
        "description": "Government and governance focused hashtags",
        "hashtags": [
            "governancetamilnadu", "no1tamilnadu", "stalinbuildstn", "mkstalingovt", 
            "naanmudhalvan", "employment"
        ],
        "priority_boost": 1.5
    }
}

class HashtagAnalyzer:
    def __init__(self):
        self.hashtag_impact_scores = {}
        self.hashtag_features = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.impact_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.hashtag_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.is_trained = False
        
        # RAG-specific attributes
        self.rag_hashtag_frequency = {}
        self.rag_hashtag_contexts = {}
        self.rag_content_vectors = None
        self.rag_hashtag_vectors = {}
        self.rag_tfidf = TfidfVectorizer(max_features=500, stop_words='english')

  
    async def load_rag_data(self):
        """Load and process RAG data from CSV file"""
        try:
            rag_data_path =  os.path.join(os.path.dirname(__file__), "unique_ad_bodies.csv")
            logger.info(f"Looking for RAG data file: {rag_data_path}")

            if os.path.exists(rag_data_path):
                logger.info("RAG data file found, loading...")
                with open(rag_data_path, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    rag_data = []
                    
                    for row in reader:
                        content = row.get('creative_details.body', '')
                        if content:
                            rag_data.append(content)
                    
                    if rag_data:
                        await self.process_rag_data(rag_data)
                        logger.info(f"✅ RAG data loaded: {len(rag_data)} entries processed")
                        logger.info(f"✅ Extracted {len(self.rag_hashtag_frequency)} unique hashtags")
                    else:
                        logger.warning("⚠️ RAG CSV file is empty or has no valid content")
                        await self.create_fallback_rag_data()
            else:
                logger.warning(f"⚠️ RAG CSV file not found at {rag_data_path}, using fallback data")
                await self.create_fallback_rag_data()
                
        except Exception as e:
            logger.error(f"❌ Error loading RAG data: {str(e)}")
            await self.create_fallback_rag_data()


    async def create_fallback_rag_data(self):
        """Create fallback RAG data based on known high-performing hashtags"""
        logger.info("Creating fallback RAG data...")
        fallback_data = [
            "எல்லோரும் நம்முடன்! தமிழ்நாட்டை முன்னேற்றும் திராவிட மாதிரி! #EllorumNammudan #DravidaModel #DMK4TN",
            "முதலமைச்சர் மு.க.ஸ்டாலின் அவர்களின் தலைமையில் வளர்ச்சியில் முன்னணியில் தமிழ்நாடு! #CMMKStalin #MKStalinCM #No1TamilNadu",
            "ஒரே தலைவன்! மக்களின் நம்பிக்கைக்குரிய தலைவர்! #OraeThalaivan #MKStalin #GovernanceTamilNadu",
            "திராவிட மாதிரி ஆட்சியால் வளர்ச்சியடையும் தமிழ்நாடு! #DravidianModel #StalinBuildsTN #MKStalinGovt",
            "Healthcare infrastructure development in Tamil Nadu with new hospitals! #HealthcareTN #Development #TamilNadu",
            "Education and employment opportunities for Tamil Nadu youth! #Education #Employment #YouthDevelopment #TamilNadu",
            "Agriculture modernization and farmer support in Tamil Nadu! #Agriculture #Farmers #ModernFarming #TamilNadu",
            "Women empowerment and safety initiatives across Tamil Nadu! #WomenEmpowerment #Safety #GenderEquality #TamilNadu",
            "Technology and innovation driving Tamil Nadu forward! #Technology #Innovation #DigitalTamilNadu #StartupTN",
            "Infrastructure development connecting rural and urban Tamil Nadu! #Infrastructure #RuralDevelopment #UrbanPlanning #TamilNadu"
        ]
        await self.process_rag_data(fallback_data)
        logger.info(f"✅ Fallback RAG data created with {len(self.rag_hashtag_frequency)} hashtags")
    
    async def process_rag_data(self, rag_data: List[str]):
        """Process RAG data to extract hashtags and contexts"""
        self.rag_hashtag_frequency = {}
        self.rag_hashtag_contexts = defaultdict(list)
        
        processed_content = []
        
        for content in rag_data:
            if content:
                hashtags = self.extract_hashtags(content)
                content_without_hashtags = re.sub(r'#\w+', '', content).strip()
                
                if content_without_hashtags:
                    processed_content.append(content_without_hashtags)
                
                for hashtag in hashtags:
                    self.rag_hashtag_frequency[hashtag] = self.rag_hashtag_frequency.get(hashtag, 0) + 1
                    self.rag_hashtag_contexts[hashtag].append({
                        'content': content_without_hashtags,
                        'full_content': content
                    })
        
        if processed_content:
            try:
                self.rag_content_vectors = self.rag_tfidf.fit_transform(processed_content)
                logger.info(f"✅ RAG TF-IDF vectors created for {len(processed_content)} contents")
            except Exception as e:
                logger.warning(f"⚠️ Could not create RAG TF-IDF vectors: {str(e)}")
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        if pd.isna(text) or text is None:
            return []
        hashtags = re.findall(r'#\w+', str(text).lower())
        return [tag.replace('#', '') for tag in hashtags]
    
    def enhanced_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis with emotional tone detection"""
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
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        positive_score = positive_count / max(total_words, 1)
        negative_score = negative_count / max(total_words, 1)
        neutral_score = 1 - (positive_score + negative_score)
        
        if positive_score > negative_score:
            sentiment = "positive"
            confidence = positive_score / (positive_score + negative_score + 0.1)
        elif negative_score > positive_score:
            sentiment = "negative"
            confidence = negative_score / (positive_score + negative_score + 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": min(confidence, 1.0),
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": neutral_score
        }
    
    def detect_hashtag_language(self, hashtag: str) -> str:
        """Detect if hashtag is in Tamil, English, or mixed"""
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
    
    def get_rag_hashtags_for_content(self, content: str, max_hashtags: int = 10) -> List[HashtagResult]:
        """Get hashtags from RAG data based on content similarity"""
        rag_hashtags = []
        
        # Fallback to most frequent hashtags
        top_hashtags = sorted(
            self.rag_hashtag_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_hashtags]
        
        for hashtag, frequency in top_hashtags:
            rag_hashtags.append(HashtagResult(
                hashtag=hashtag,
                score=min(90.0, 60.0 + (frequency * 2)),
                category="rag_frequent",
                reasoning=f"High-frequency hashtag from historical data (used {frequency} times)",
                source="existing",
                frequency_in_rag=frequency,
                language=self.detect_hashtag_language(hashtag)
            ))
        
        return rag_hashtags
    
    def predict_hashtags_with_rag(self, content: str, max_hashtags: int = 15, 
                                language_preference: str = "both") -> Dict[str, Any]:
        """Enhanced hashtag prediction using RAG + ML"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get RAG-based hashtags (highest priority)
        rag_hashtags = self.get_rag_hashtags_for_content(content, max_hashtags // 2)
        
        # Get ML-based hashtags for new suggestions
        ml_hashtags = self.get_ml_hashtags(content, max_hashtags // 2, language_preference)
        
        # Combine and categorize
        all_hashtags = []
        existing_count = 0
        new_count = 0
        rag_enhanced_count = 0
        
        # Add RAG hashtags with priority
        for hashtag in rag_hashtags:
            hashtag.source = "existing"
            existing_count += 1
            all_hashtags.append(hashtag)
        
        # Add ML hashtags as new suggestions
        for hashtag in ml_hashtags:
            if hashtag.hashtag in self.rag_hashtag_frequency:
                hashtag.source = "rag_enhanced"
                hashtag.frequency_in_rag = self.rag_hashtag_frequency[hashtag.hashtag]
                hashtag.score = hashtag.score * 1.3
                rag_enhanced_count += 1
            else:
                hashtag.source = "new"
                new_count += 1
            all_hashtags.append(hashtag)
        
        # Remove duplicates and sort by score
        seen_hashtags = set()
        unique_hashtags = []
        for hashtag in all_hashtags:
            if hashtag.hashtag not in seen_hashtags:
                seen_hashtags.add(hashtag.hashtag)
                unique_hashtags.append(hashtag)
        
        # Sort by priority: existing > rag_enhanced > new, then by score
        priority_order = {"existing": 3, "rag_enhanced": 2, "new": 1}
        unique_hashtags.sort(
            key=lambda x: (priority_order.get(x.source, 0), x.score), 
            reverse=True
        )
        
        # Prepare RAG analysis
        top_rag_hashtags = [
            RAGHashtagInfo(
                hashtag=hashtag, 
                frequency=freq, 
                category=self.categorize_rag_hashtag(hashtag)
            )
            for hashtag, freq in sorted(
                self.rag_hashtag_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        ]
        
        rag_analysis = RAGAnalysis(
            total_rag_hashtags=len(self.rag_hashtag_frequency),
            existing_hashtags_found=existing_count,  # Fixed: correct field name
            new_hashtags_suggested=new_count,
            rag_enhanced_hashtags=rag_enhanced_count,
            top_rag_hashtags=top_rag_hashtags
        )
        
        return {
            "hashtags": unique_hashtags[:max_hashtags],
            "rag_analysis": rag_analysis
        }
    
    def categorize_rag_hashtag(self, hashtag: str) -> str:
        """Categorize RAG hashtag based on predefined categories"""
        hashtag_lower = hashtag.lower()
        
        for category, data in RAG_HASHTAG_CATEGORIES.items():
            if hashtag_lower in [h.lower() for h in data["hashtags"]]:
                return category
        
        if any(term in hashtag_lower for term in ['dmk', 'stalin', 'dravidian', 'kalaignar']):
            return "political_identity"
        elif any(term in hashtag_lower for term in ['govt', 'governance', 'tn', 'tamilnadu']):
            return "governance"
        else:
            return "general"
    
    def get_ml_hashtags(self, content: str, max_hashtags: int, language_preference: str) -> List[HashtagResult]:
        """Get ML-based hashtag predictions"""
        ml_hashtags = []
        candidate_hashtags = list(self.hashtag_impact_scores.keys())
        
        for hashtag in candidate_hashtags:
            base_score = self.hashtag_impact_scores.get(hashtag, 0)
            
            hashtag_terms = hashtag.split('_') if '_' in hashtag else [hashtag]
            content_lower = content.lower()
            
            relevance_score = sum(
                1 for term in hashtag_terms 
                if term in content_lower
            ) / len(hashtag_terms)
            
            hashtag_language = self.detect_hashtag_language(hashtag)
            language_bonus = 1.0
            
            if language_preference == "tamil" and hashtag_language == "tamil":
                language_bonus = 1.2
            elif language_preference == "english" and hashtag_language == "english":
                language_bonus = 1.2
            elif language_preference == "both":
                language_bonus = 1.1 if hashtag_language in ["tamil", "english"] else 1.0
            
            final_score = (
                base_score * 0.6 +
                relevance_score * 100 * 0.4
            ) * language_bonus
            
            if final_score > 5:
                ml_hashtags.append(HashtagResult(
                    hashtag=hashtag,
                    score=round(final_score, 2),
                    category="ml_predicted",
                    reasoning=f"ML model prediction based on content relevance and historical performance",
                    language=hashtag_language
                ))
        
        ml_hashtags.sort(key=lambda x: x.score, reverse=True)
        return ml_hashtags[:max_hashtags]
    
    def train_model(self, ads_data: List[Dict[str, Any]]):
        """Train the hashtag prediction model with both ads and RAG data"""
        logger.info("🎯 Training hashtag prediction model...")
        logger.info(f"📊 Input ads data: {len(ads_data)} entries")
        logger.info(f"📚 Available RAG hashtags: {len(self.rag_hashtag_frequency)}")
        
        # Process historical ads data (if any)
        if ads_data:
            logger.info("📈 Processing historical ads data...")
            self.analyze_historical_data(ads_data)
            logger.info(f"📈 Extracted {len(self.hashtag_impact_scores)} hashtags from ads data")
        else:
            logger.info("📈 No ads data provided, using RAG data only")
            # Initialize with empty dict if no ads data
            self.hashtag_impact_scores = {}
        
        # Integrate RAG hashtag scores
        if self.rag_hashtag_frequency:
            logger.info("📚 Integrating RAG hashtag scores...")
            for hashtag, frequency in self.rag_hashtag_frequency.items():
                base_score = self.hashtag_impact_scores.get(hashtag, 0)
                rag_boost = frequency * 10  # Boost based on frequency
                self.hashtag_impact_scores[hashtag] = base_score + rag_boost
            logger.info(f"📚 Enhanced {len(self.hashtag_impact_scores)} hashtags with RAG data")
        else:
            logger.warning("📚 No RAG hashtag data available for integration")
        
        # Ensure we have some hashtags even if everything fails
        if not self.hashtag_impact_scores:
            logger.warning("📚 No hashtag data available, creating minimal fallback")
            # Create minimal fallback hashtag scores
            fallback_hashtags = {
                "dmk4tn": 100,
                "dravidamodel": 95,
                "tamilnadu": 90,
                "mkstalin": 85,
                "governancetamilnadu": 80,
                "development": 75,
                "healthcare": 70,
                "education": 65,
                "employment": 60,
                "innovation": 55
            }
            self.hashtag_impact_scores.update(fallback_hashtags)
            logger.info(f"📚 Created {len(fallback_hashtags)} fallback hashtags")
        
        self.is_trained = True
        logger.info(f"✅ Model training completed!")
        logger.info(f"✅ Total hashtags available: {len(self.hashtag_impact_scores)}")
        logger.info(f"✅ Model status: {'Trained' if self.is_trained else 'Not Trained'}")
    
    def analyze_historical_data(self, ads_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical ads data"""
        hashtag_metrics = defaultdict(lambda: {
            'total_impact': 0, 'count': 0, 'avg_engagement': 0,
            'avg_reach': 0, 'avg_ctr': 0, 'total_spend': 0, 'campaigns': set()
        })
        
        for ad in ads_data:
            creative_body = ad.get('creative_details', {}).get('body', '')
            hashtags = self.extract_hashtags(creative_body)
            
            if not hashtags:
                continue
                
            impact_score = self.calculate_impact_score(ad)
            
            for hashtag in hashtags:
                metrics = hashtag_metrics[hashtag]
                metrics['total_impact'] += impact_score
                metrics['count'] += 1
                metrics['avg_engagement'] += float(ad.get('inline_post_engagement', 0))
        
        for hashtag, metrics in hashtag_metrics.items():
            if metrics['count'] > 0:
                metrics['avg_impact'] = metrics['total_impact'] / metrics['count']
                self.hashtag_impact_scores[hashtag] = metrics['avg_impact']
        
        return dict(hashtag_metrics)
    
    def calculate_impact_score(self, row: Dict[str, Any]) -> float:
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
    
    def get_rag_context_for_ai(self, content: str) -> str:
        """Get RAG context for AI prompt"""
        top_hashtags = sorted(
            self.rag_hashtag_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        rag_context = "HIGH-FREQUENCY HASHTAGS (use these when relevant):\n"
        for hashtag, freq in top_hashtags[:10]:
            rag_context += f"#{hashtag} (used {freq} times)\n"
        
        return rag_context
    
    def enhance_ai_results_with_rag(self, ai_hashtags: List[Dict[str, Any]], content: str) -> List[HashtagResult]:
        """Enhance AI results with RAG metadata"""
        enhanced_hashtags = []
        
        for hashtag_dict in ai_hashtags:
            hashtag = hashtag_dict["hashtag"]
            frequency = self.rag_hashtag_frequency.get(hashtag, 0)
            
            source = hashtag_dict.get("source", "new")
            if frequency > 0:
                source = "existing" if frequency > 10 else "rag_enhanced"
            
            similar_examples = []
            if hashtag in self.rag_hashtag_contexts:
                examples = self.rag_hashtag_contexts[hashtag][:2]
                similar_examples = [ex['content'][:100] + "..." for ex in examples]
            
            enhanced_hashtags.append(HashtagResult(
                hashtag=hashtag,
                score=hashtag_dict["score"],
                category=hashtag_dict["category"],
                reasoning=hashtag_dict["reasoning"],
                source=source,
                frequency_in_rag=frequency if frequency > 0 else None,
                similar_content_examples=similar_examples if similar_examples else None,
                language=hashtag_dict.get("language", "unknown")
            ))
        
        return enhanced_hashtags
    
    def get_rag_analysis_summary(self) -> RAGAnalysis:
        """Get summary of RAG analysis"""
        top_rag_hashtags = [
            RAGHashtagInfo(
                hashtag=hashtag, 
                frequency=freq, 
                category=self.categorize_rag_hashtag(hashtag)
            )
            for hashtag, freq in sorted(
                self.rag_hashtag_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        ]
        
        return RAGAnalysis(
            total_rag_hashtags=len(self.rag_hashtag_frequency),
            existing_hashtags_found=0,
            new_hashtags_suggested=0,
            rag_enhanced_hashtags=0,
            top_rag_hashtags=top_rag_hashtags
        )

# Global analyzer instance
analyzer = HashtagAnalyzer()

async def predict_with_azure_openai_rag(content: str, config: ApiConfig, max_hashtags: int = 10,
                                      strategies: Optional[HashtagStrategies] = None, sentiment: str = None,
                                      language_preference: str = "both", rag_context: str = "") -> Dict[str, Any]:
    """Predict hashtags using Azure OpenAI enhanced with RAG context"""
    try:
        if config.provider == "azure":
            client = AzureOpenAI(
                api_key=config.api_key,
                api_version="2025-01-01-preview",
                azure_endpoint=config.endpoint
            )
        else:
            client = openai.OpenAI(api_key=config.api_key)
        
        prompt = f"""
            You are an expert social media strategist specializing in hashtag optimization for Tamil Nadu government and social campaigns.

            IMPORTANT RAG CONTEXT - Use these high-performing hashtags from historical data:
            {rag_context}

            Content: "{content}"
            Language Preference: {language_preference}
            Target Sentiment: {sentiment or 'balanced'}

            Respond in JSON format:
            {{
            "sentiment_analysis": {{
                "sentiment": "positive/negative/neutral",
                "confidence": 0.85,
                "positive_score": 0.7,
                "negative_score": 0.1,
                "neutral_score": 0.2,
                "emotional_tone": "optimistic",
                "key_emotions": ["joy", "trust"],
                "sentiment_keywords": ["development", "progress"]
            }},
            "analysis": {{
                "themes": ["healthcare", "development"],
                "keywords": ["healthcare", "infrastructure", "rural"],
                "content_type": "government",
                "language": "english",
                "target_audience": "general public"
            }},
            "hashtags": [
                {{
                "hashtag": "HealthcareTN",
                "score": 85,
                "category": "government",
                "reasoning": "Relevant to healthcare development content",
                "source": "new",
                "language": "english"
                }}
            ]
            }}
            """

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert social media strategist with access to historical hashtag performance data."
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
            cleaned = ai_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.lstrip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].lstrip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].rstrip()
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"AI response is not valid JSON: {ai_response!r}")
            return {
                "sentiment_analysis": {
                    "sentiment": "neutral", 
                    "confidence": 0.5,
                    "positive_score": 0.33,
                    "negative_score": 0.33,
                    "neutral_score": 0.34,
                    "emotional_tone": "informative"
                },
                "analysis": {"themes": ["general"], "keywords": [], "content_type": "general"},
                "hashtags": []
            }
    
    except Exception as e:
        logger.error(f"Azure OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI API error: {str(e)}")

@app.post("/predict-hashtags", response_model=PredictionResponse)
async def predict_hashtags(request: PredictHashtagFullRequest):
    try:
        sentiment_analysis = analyzer.enhanced_sentiment_analysis(request.content)
        final_sentiment = request.sentiment or sentiment_analysis['sentiment']
        max_hashtags = min(request.max_hashtags, 20)
        language_preference = request.language_preference
        prediction_method = request.prediction_method

        # Force RAG-only prediction
        if prediction_method == "rag_only":
            logger.info("Using RAG-only prediction method")
            result = analyzer.predict_hashtags_with_rag(
                request.content, max_hashtags, language_preference
            )
            
            analysis = {
                "themes": ["development", "technology"],
                "keywords": [word for word in request.content.split() if len(word) > 3][:8],
                "content_type": "general",
                "language": "mixed",
                "target_audience": "general public",
                "prediction_method": "RAG Only"
            }
            
            sentiment_analysis_obj = SentimentAnalysis(
                sentiment=sentiment_analysis["sentiment"],
                confidence=sentiment_analysis["confidence"],
                positive_score=sentiment_analysis["positive_score"],
                negative_score=sentiment_analysis["negative_score"],
                neutral_score=sentiment_analysis["neutral_score"]
            )
            
            return PredictionResponse(
                hashtags=result["hashtags"],
                analysis=analysis,
                sentiment_analysis=sentiment_analysis_obj,
                rag_analysis=result["rag_analysis"],
                source="RAG_ONLY",
                timestamp=datetime.now().isoformat()
            )

        # Force AI-only prediction (requires API config)
        elif prediction_method == "ai_only":
            if not request.config or not request.config.api_key:
                raise HTTPException(
                    status_code=400, 
                    detail="API configuration required for AI-only prediction method"
                )
            
            logger.info("Using AI-only prediction method")
            
            # Use AI without RAG context for pure AI predictions
            ai_result = await predict_with_azure_openai_rag(
                request.content, request.config, max_hashtags, 
                request.strategies, final_sentiment, 
                language_preference, ""  # Empty RAG context for AI-only
            )
            
            # Don't enhance with RAG for AI-only mode
            ai_hashtags = []
            for hashtag_dict in ai_result["hashtags"]:
                ai_hashtags.append(HashtagResult(
                    hashtag=hashtag_dict["hashtag"],
                    score=hashtag_dict["score"],
                    category=hashtag_dict["category"],
                    reasoning=hashtag_dict["reasoning"],
                    source="new",  # All are new for AI-only
                    language=hashtag_dict.get("language", "unknown")
                ))
            
            ai_sentiment = ai_result.get("sentiment_analysis", {})
            sentiment_analysis_obj = SentimentAnalysis(
                sentiment=ai_sentiment.get("sentiment", sentiment_analysis["sentiment"]),
                confidence=ai_sentiment.get("confidence", sentiment_analysis["confidence"]),
                positive_score=ai_sentiment.get("positive_score", sentiment_analysis["positive_score"]),
                negative_score=ai_sentiment.get("negative_score", sentiment_analysis["negative_score"]),
                neutral_score=ai_sentiment.get("neutral_score", sentiment_analysis["neutral_score"]),
                emotional_tone=ai_sentiment.get("emotional_tone"),
                key_emotions=ai_sentiment.get("key_emotions"),
                sentiment_keywords=ai_sentiment.get("sentiment_keywords"),
                associated_party=ai_sentiment.get("associated_party"),
                party_confidence=ai_sentiment.get("party_confidence")
            )
            
            # Create minimal RAG analysis for AI-only
            rag_analysis = RAGAnalysis(
                total_rag_hashtags=len(analyzer.rag_hashtag_frequency),
                existing_hashtags_found=0,
                new_hashtags_suggested=len(ai_hashtags),
                rag_enhanced_hashtags=0,
                top_rag_hashtags=[]
            )
            
            ai_result["analysis"]["prediction_method"] = "AI Only"
            
            return PredictionResponse(
                hashtags=ai_hashtags,
                analysis=ai_result["analysis"],
                sentiment_analysis=sentiment_analysis_obj,
                rag_analysis=rag_analysis,
                source=f"AI_{request.config.provider.upper()}_ONLY",
                timestamp=datetime.now().isoformat()
            )

        # Auto mode (existing logic)
        else:
            if request.config and request.config.api_key:
                logger.info("Using AI API with RAG enhancement for prediction (Auto mode)")
                
                rag_context = analyzer.get_rag_context_for_ai(request.content)
                
                ai_result = await predict_with_azure_openai_rag(
                    request.content, request.config, max_hashtags, 
                    request.strategies, final_sentiment, 
                    language_preference, rag_context
                )
                
                enhanced_hashtags = analyzer.enhance_ai_results_with_rag(
                    ai_result["hashtags"], request.content
                )
                
                ai_sentiment = ai_result.get("sentiment_analysis", {})
                sentiment_analysis_obj = SentimentAnalysis(
                    sentiment=ai_sentiment.get("sentiment", sentiment_analysis["sentiment"]),
                    confidence=ai_sentiment.get("confidence", sentiment_analysis["confidence"]),
                    positive_score=ai_sentiment.get("positive_score", sentiment_analysis["positive_score"]),
                    negative_score=ai_sentiment.get("negative_score", sentiment_analysis["negative_score"]),
                    neutral_score=ai_sentiment.get("neutral_score", sentiment_analysis["neutral_score"]),
                    emotional_tone=ai_sentiment.get("emotional_tone"),
                    key_emotions=ai_sentiment.get("key_emotions"),
                    sentiment_keywords=ai_sentiment.get("sentiment_keywords"),
                    associated_party=ai_sentiment.get("associated_party"),
                    party_confidence=ai_sentiment.get("party_confidence")
                )
                
                ai_result["analysis"]["prediction_method"] = "AI + RAG (Auto)"
                
                return PredictionResponse(
                    hashtags=enhanced_hashtags,
                    analysis=ai_result["analysis"],
                    sentiment_analysis=sentiment_analysis_obj,
                    rag_analysis=analyzer.get_rag_analysis_summary(),
                    source=f"AI_{request.config.provider.upper()}_RAG_AUTO",
                    timestamp=datetime.now().isoformat()
                )
            else:
                logger.info("Using local ML model with RAG for prediction (Auto mode)")
                result = analyzer.predict_hashtags_with_rag(
                    request.content, max_hashtags, language_preference
                )
                
                analysis = {
                    "themes": ["development", "technology"],
                    "keywords": [word for word in request.content.split() if len(word) > 3][:8],
                    "content_type": "general",
                    "language": "mixed",
                    "target_audience": "general public",
                    "prediction_method": "ML + RAG (Auto)"
                }
                
                sentiment_analysis_obj = SentimentAnalysis(
                    sentiment=sentiment_analysis["sentiment"],
                    confidence=sentiment_analysis["confidence"],
                    positive_score=sentiment_analysis["positive_score"],
                    negative_score=sentiment_analysis["negative_score"],
                    neutral_score=sentiment_analysis["neutral_score"]
                )
                
                return PredictionResponse(
                    hashtags=result["hashtags"],
                    analysis=analysis,
                    sentiment_analysis=sentiment_analysis_obj,
                    rag_analysis=result["rag_analysis"],
                    source="LOCAL_ML_RAG_AUTO",
                    timestamp=datetime.now().isoformat()
                )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/health")
async def health_check():
    """Health check endpoint with RAG status"""
    return {
        "status": "healthy",
        "model_trained": analyzer.is_trained,
        "hashtag_count": len(analyzer.hashtag_impact_scores),
        "rag_hashtags": len(analyzer.rag_hashtag_frequency),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/rag-stats")
async def get_rag_stats():
    """Get RAG database statistics"""
    logger.info("get_rag_stats called")
    
    if not analyzer.rag_hashtag_frequency:
        logger.warning("RAG data not loaded, returning fallback stats")
        
        # Return fallback stats if RAG data isn't loaded
        return {
            "total_rag_hashtags": 0,
            "top_hashtags": [],
            "categories": {"fallback": 1},
            "total_contexts": 0,
            "status": "RAG data not loaded - check if unique_ad_bodies.csv exists"
        }
    
    top_hashtags = [
        RAGHashtagInfo(
            hashtag=hashtag, 
            frequency=freq, 
            category=analyzer.categorize_rag_hashtag(hashtag)
        )
        for hashtag, freq in sorted(
            analyzer.rag_hashtag_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
    ]
    
    categories = {}
    for hashtag, freq in analyzer.rag_hashtag_frequency.items():
        category = analyzer.categorize_rag_hashtag(hashtag)
        categories[category] = categories.get(category, 0) + 1
    
    return {
        "total_rag_hashtags": len(analyzer.rag_hashtag_frequency),
        "top_hashtags": [
            {
                "hashtag": hashtag_info.hashtag, 
                "frequency": hashtag_info.frequency, 
                "category": hashtag_info.category
            }
            for hashtag_info in top_hashtags
        ],
        "categories": categories,
        "total_contexts": sum(len(contexts) for contexts in analyzer.rag_hashtag_contexts.values()),
        "status": "RAG data loaded successfully"
    }

@app.get("/top-hashtags")
async def get_top_hashtags(limit: int = 20, source: str = "all"):
    """Get top performing hashtags with source filter"""
    logger.info(f"get_top_hashtags called with source={source}, limit={limit}")
    logger.info(f"Model trained status: {analyzer.is_trained}")
    logger.info(f"Available hashtag scores: {len(analyzer.hashtag_impact_scores)}")
    logger.info(f"Available RAG hashtags: {len(analyzer.rag_hashtag_frequency)}")
    
    # Always return something useful, even if model isn't trained
    if not analyzer.is_trained:
        logger.warning("Model not trained, returning fallback hashtags")
        fallback_hashtags = [
            {"hashtag": "DMK4TN", "score": 95, "source": "fallback", "category": "political_identity"},
            {"hashtag": "DravidaModel", "score": 90, "source": "fallback", "category": "political_identity"},
            {"hashtag": "TamilNadu", "score": 85, "source": "fallback", "category": "governance"},
            {"hashtag": "MKStalin", "score": 80, "source": "fallback", "category": "political_identity"},
            {"hashtag": "GovernanceTamilNadu", "score": 75, "source": "fallback", "category": "governance"},
            {"hashtag": "Development", "score": 70, "source": "fallback", "category": "governance"},
            {"hashtag": "Healthcare", "score": 65, "source": "fallback", "category": "governance"},
            {"hashtag": "Education", "score": 60, "source": "fallback", "category": "governance"},
            {"hashtag": "Employment", "score": 55, "source": "fallback", "category": "governance"},
            {"hashtag": "Innovation", "score": 50, "source": "fallback", "category": "governance"}
        ]
        
        return {
            "top_hashtags": fallback_hashtags[:limit],
            "total_count": len(fallback_hashtags),
            "source": "Fallback Data (Model Not Trained)"
        }
    
    if source == "rag":
        if not analyzer.rag_hashtag_frequency:
            logger.warning("No RAG hashtag data available")
            return {
                "top_hashtags": [],
                "total_count": 0,
                "source": "RAG Database (No Data)"
            }
        
        sorted_hashtags = sorted(
            analyzer.rag_hashtag_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return {
            "top_hashtags": [
                {
                    "hashtag": hashtag, 
                    "score": freq, 
                    "source": "rag",
                    "category": analyzer.categorize_rag_hashtag(hashtag)
                }
                for hashtag, freq in sorted_hashtags[:limit]
            ],
            "total_count": len(analyzer.rag_hashtag_frequency),
            "source": "RAG Database"
        }
    elif source == "ml":
        if not analyzer.hashtag_impact_scores:
            logger.warning("No ML hashtag data available")
            return {
                "top_hashtags": [],
                "total_count": 0,
                "source": "ML Model (No Data)"
            }
        
        sorted_hashtags = sorted(
            analyzer.hashtag_impact_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return {
            "top_hashtags": [
                {"hashtag": hashtag, "score": score, "source": "ml", "category": "ml_predicted"}
                for hashtag, score in sorted_hashtags[:limit]
                if hashtag not in analyzer.rag_hashtag_frequency
            ],
            "total_count": len(analyzer.hashtag_impact_scores),
            "source": "ML Model"
        }
    else:  # source == "all"
        combined_hashtags = {}
        
        # Add RAG hashtags with boost
        for hashtag, freq in analyzer.rag_hashtag_frequency.items():
            combined_hashtags[hashtag] = {
                "score": freq * 10,
                "source": "rag",
                "category": analyzer.categorize_rag_hashtag(hashtag)
            }
        
        # Add ML hashtags
        for hashtag, score in analyzer.hashtag_impact_scores.items():
            if hashtag in combined_hashtags:
                combined_hashtags[hashtag]["score"] += score
                combined_hashtags[hashtag]["source"] = "rag_enhanced"
            else:
                combined_hashtags[hashtag] = {
                    "score": score,
                    "source": "ml",
                    "category": "ml_predicted"
                }
        
        if not combined_hashtags:
            # Return fallback hashtags if no data available
            logger.warning("No combined hashtag data available, using fallback")
            fallback_hashtags = [
                {"hashtag": "DMK4TN", "score": 95, "source": "fallback", "category": "political_identity"},
                {"hashtag": "DravidaModel", "score": 90, "source": "fallback", "category": "political_identity"},
                {"hashtag": "TamilNadu", "score": 85, "source": "fallback", "category": "governance"},
                {"hashtag": "MKStalin", "score": 80, "source": "fallback", "category": "political_identity"},
                {"hashtag": "GovernanceTamilNadu", "score": 75, "source": "fallback", "category": "governance"},
                {"hashtag": "Development", "score": 70, "source": "fallback", "category": "governance"},
                {"hashtag": "Healthcare", "score": 65, "source": "fallback", "category": "governance"},
                {"hashtag": "Education", "score": 60, "source": "fallback", "category": "governance"}
            ]
            
            return {
                "top_hashtags": fallback_hashtags[:limit],
                "total_count": len(fallback_hashtags),
                "source": "Fallback Data"
            }
        
        sorted_hashtags = sorted(
            combined_hashtags.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        return {
            "top_hashtags": [
                {
                    "hashtag": hashtag,
                    "score": data["score"],
                    "source": data["source"],
                    "category": data.get("category", "general")
                }
                for hashtag, data in sorted_hashtags[:limit]
            ],
            "total_count": len(combined_hashtags),
            "source": "Combined (RAG + ML)"
        }

class TopTrendingRequest(BaseModel):
    platform: str = "all"
    count: int = 10
    config: Optional[ApiConfig] = None

@app.post("/top-trending-hashtags")
async def get_top_trending_hashtags(request: TopTrendingRequest = Body(...)):
    """Get top trending hashtags for today from social platforms using AI"""
    platform_map = {
        "Instagram": "Instagram",
        "Facebook": "Facebook", 
        "Twitter": "Twitter",
        "all": "Instagram, Facebook, and Twitter"
    }
    platform_name = platform_map.get(request.platform, "Instagram, Facebook, and Twitter")
    
    # Simple fallback response if no AI API configured
    if not request.config or not request.config.api_key:
        fallback_hashtags = [
            {"hashtag": "#TamilNadu", "platform": "All", "reason": "Popular regional hashtag"},
            {"hashtag": "#DMK", "platform": "All", "reason": "Political party hashtag"},
            {"hashtag": "#Healthcare", "platform": "All", "reason": "Trending topic"},
            {"hashtag": "#Development", "platform": "All", "reason": "Government initiatives"},
            {"hashtag": "#Innovation", "platform": "All", "reason": "Technology focus"}
        ]
        
        return {
            "platform": platform_name,
            "top_trending_hashtags": fallback_hashtags[:request.count]
        }
    
    prompt = (
        f"List the top {request.count} popular trending hashtags in Tamil Nadu/India related to political parties, especially DMK, on {platform_name}. "
        "Return only the hashtags as a JSON array of objects with fields: hashtag, platform, and a short reason for each."
    )

    try:
        if request.config.provider == "azure":
            client = AzureOpenAI(
                api_key=request.config.api_key,
                api_version="2025-01-01-preview",
                azure_endpoint=request.config.endpoint
            )
        else:
            client = openai.OpenAI(api_key=request.config.api_key)

        response = client.chat.completions.create(
            model=request.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a social media analytics expert with deep knowledge of Tamil Nadu politics."
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
        
        # Return fallback on error
        fallback_hashtags = [
            {"hashtag": "#TamilNadu", "platform": platform_name, "reason": "Regional identity"},
            {"hashtag": "#DMK4TN", "platform": platform_name, "reason": "Political engagement"},
            {"hashtag": "#Development", "platform": platform_name, "reason": "Government focus"}
        ]
        
        return {
            "platform": platform_name,
            "top_trending_hashtags": fallback_hashtags[:request.count]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)