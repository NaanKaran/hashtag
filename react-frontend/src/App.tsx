import React, { useState, useEffect } from 'react';
import { Hash, Target, TrendingUp, Globe, Zap, Copy, Check, Brain, Settings, AlertCircle, Wifi, WifiOff, Sparkles, BarChart3, Filter, Heart, Languages, Smile, Frown, Meh, Database, Star, Clock, Lightbulb, Sliders } from 'lucide-react';

interface Prediction {
  hashtag: string;
  score: number;
  category: string;
  reasoning: string;
  strategy?: string;
  sentiment?: string;
  language?: string;
  source: string;
  frequency_in_rag?: number;
  similar_content_examples?: string[];
}

interface RAGAnalysis {
  total_rag_hashtags: number;
  existing_hashtags_found: number;
  new_hashtags_suggested: number;
  rag_enhanced_hashtags: number;
  top_rag_hashtags: Array<{
    hashtag: string;
    frequency: number;
    category: string;
  }>;
}

interface Analysis {
  themes?: string[];
  keywords?: string[];
  content_type?: string;
  language?: string;
  target_audience?: string;
  sentiment?: string;
}

interface SentimentAnalysis {
  sentiment: string;
  confidence: number;
  positive_score: number;
  negative_score: number;
  neutral_score: number;
  emotional_tone?: string;
  key_emotions?: string[];
  sentiment_keywords?: string[];
  associated_party?: string;
  party_confidence?: number;
}

interface ApiConfig {
  provider: string;
  apiKey: string;
  endpoint: string;
  model: string;
}

interface HashtagStrategy {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

const HashtagPredictor: React.FC = () => {
  const [content, setContent] = useState<string>('');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [copiedHashtag, setCopiedHashtag] = useState<string>('');
  const [analysisDetails, setAnalysisDetails] = useState<Analysis | null>(null);
  const [sentimentAnalysis, setSentimentAnalysis] = useState<SentimentAnalysis | null>(null);
  const [ragAnalysis, setRagAnalysis] = useState<RAGAnalysis | null>(null);
  // Fix: Ensure correct API config keys and fallback for missing config
  const [apiConfig, setApiConfig] = useState<ApiConfig>(() => {
    const saved = localStorage.getItem('hashtagPredictor_apiConfig');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        // Backward compatibility: support both apiKey and api_key
        return {
          provider: parsed.provider || 'azure',
          apiKey: parsed.apiKey || parsed.api_key || '',
          endpoint: parsed.endpoint || '',
          model: parsed.model || 'gpt-3.5-turbo',
        };
      } catch {
        // fallback
      }
    }
    return {
      provider: 'azure',
      apiKey: '',
      endpoint: '',
      model: 'gpt-3.5-turbo',
    };
  });
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [apiStatus, setApiStatus] = useState<string>('disconnected');
  const [topHashtags, setTopHashtags] = useState<Array<{hashtag: string, score: number, source?: string, category?: string}>>([]);
  const [predictionSource, setPredictionSource] = useState<string>('');
  
  // Enhanced state for new features
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [sentimentFilter, setSentimentFilter] = useState<string>('all');
  const [enableSentimentAnalysis, setEnableSentimentAnalysis] = useState<boolean>(true);
  const [languagePreference, setLanguagePreference] = useState<string>('both');
  const [copiedTopHashtag, setCopiedTopHashtag] = useState<string>('');
  const [ragStats, setRagStats] = useState<any>(null);
  const [hashtagSourceFilter, setHashtagSourceFilter] = useState<string>('all');
  const [predictionMethod, setPredictionMethod] = useState('auto');
  const [maxHashtags, setMaxHashtags] = useState(10);
  const API_BASE_URL = 'http://localhost:8000';

  // Hashtag strategies
  const hashtagStrategies: HashtagStrategy[] = [
    {
      id: 'piggybacking',
      name: 'Piggybacking',
      description: 'Leverage popular trending hashtags',
      icon: <TrendingUp className="w-4 h-4" />
    },
    {
      id: 'hijacking',
      name: 'Hijacking',
      description: 'Repurpose trending hashtags for your content',
      icon: <Zap className="w-4 h-4" />
    },
    {
      id: 'semantic_shifting',
      name: 'Semantic Shifting',
      description: 'Use hashtags with evolving meanings',
      icon: <Brain className="w-4 h-4" />
    },
    {
      id: 'linking_pairing',
      name: 'Linking / Pairing',
      description: 'Connect complementary hashtags',
      icon: <Globe className="w-4 h-4" />
    },
    {
      id: 'seeding',
      name: 'Seeding',
      description: 'Plant hashtags for future growth',
      icon: <Sparkles className="w-4 h-4" />
    },
    {
      id: 'challenges',
      name: 'Challenges',
      description: 'Create or join hashtag challenges',
      icon: <Hash className="w-4 h-4" />
    },
    {
      id: 'clustering',
      name: 'Clustering',
      description: 'Group related hashtags together',
      icon: <BarChart3 className="w-4 h-4" />
    },
    {
      id: 'mutation',
      name: 'Mutation',
      description: 'Evolve hashtags for different contexts',
      icon: <Filter className="w-4 h-4" />
    }
  ];

  const sampleContents: string[] = [
    "Tamil Nadu government announces new employment scheme for youth development and skill training programs.",
    "Healthcare infrastructure development in Tamil Nadu with new hospitals and medical facilities for rural areas.",
    "Women empowerment initiatives and safety measures implemented across Tamil Nadu for gender equality.",
    "AI and technology startup ecosystem in Chennai with new incubation centers and funding opportunities.",
    "Agricultural modernization program with smart farming techniques and support for farmers.",
    "Education reforms and digital literacy programs launched across Tamil Nadu districts."
  ];

  useEffect(() => {
    checkApiHealth();
    loadTopHashtags();
    loadSavedConfig();
    loadRagStats();
  }, []);

  const loadSavedConfig = (): void => {
    try {
      const saved = localStorage.getItem('hashtagPredictor_apiConfig');
      if (saved) {
        setApiConfig(JSON.parse(saved));
      }
    } catch (error) {
      console.error('Error loading API config:', error);
    }
  };

  const checkApiHealth = async (): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        const data = await response.json();
        setApiStatus(data.model_trained ? 'connected' : 'partial');
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('disconnected');
      console.error('API health check failed:', error);
    }
  };

  const loadTopHashtags = async (): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/top-hashtags?limit=10`);
      if (response.ok) {
        const data = await response.json();
        setTopHashtags(data.top_hashtags || []);
      } else {
        setTopHashtags([]);
      }
    } catch (error) {
      setTopHashtags([]);
      console.error('Failed to load top hashtags:', error);
    }
  };

  const loadRagStats = async (): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/rag-stats`);
      if (response.ok) {
        const data = await response.json();
        setRagStats(data);
      }
    } catch (error) {
      console.error('Failed to load RAG stats:', error);
    }
  };

  const handleStrategyChange = (strategyId: string): void => {
    setSelectedStrategies(prev => 
      prev.includes(strategyId) 
        ? prev.filter(id => id !== strategyId)
        : [...prev, strategyId]
    );
  };

  const handlePredict = async (): Promise<void> => {
    if (!content.trim()) return;
        if (predictionMethod === 'ai_only' && (!apiConfig.apiKey)) {
      setError('AI-only prediction method requires API configuration. Please configure your API settings.');
      return;
    }

    setLoading(true);
    setError('');
    setPredictions([]);
    setAnalysisDetails(null);
    setSentimentAnalysis(null);
    setRagAnalysis(null);

    try {
      const allStrategyKeys = [
        'piggybacking',
        'hijacking',
        'semantic_shifting',
        'linking_pairing',
        'seeding',
        'challenges',
        'clustering',
        'mutation'
      ];
      const strategiesObj = allStrategyKeys.reduce((acc, key) => {
        acc[key] = selectedStrategies.includes(key);
        return acc;
      }, {} as Record<string, boolean>);

      let payload: any = {
        content: content,
        max_hashtags: maxHashtags,
        strategies: strategiesObj,
        enable_sentiment_analysis: enableSentimentAnalysis,
        language_preference: languagePreference,
        sentiment: sentimentFilter !== 'all' ? sentimentFilter : undefined,
        prediction_method: predictionMethod
      };

      // Only add config if apiKey is present
       if (apiConfig.apiKey && predictionMethod !== 'rag_only') {
        payload = {
          ...payload,
          config: {
            provider: apiConfig.provider,
            api_key: apiConfig.apiKey,
            endpoint: apiConfig.endpoint,
            model: apiConfig.model
          }
        };
      }

      const response = await fetch(`${API_BASE_URL}/predict-hashtags`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail ?? `HTTP ${response.status}`);
      }

      const result = await response.json();
      setPredictions(result.hashtags ?? []);
      setAnalysisDetails(result.analysis ?? {});
      setSentimentAnalysis(result.sentiment_analysis ?? null);
      setRagAnalysis(result.rag_analysis ?? null);
      setPredictionSource(result.source ?? 'Unknown');

    } catch (err: any) {
      setError(err.message);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const copyHashtag = (hashtag: string): void => {
    navigator.clipboard.writeText(`#${hashtag}`);
    setCopiedHashtag(hashtag);
    setTimeout(() => setCopiedHashtag(''), 2000);
  };

  // Fix: Defensive copy for all hashtags
  const copyAllHashtags = () => {
    const allHashtags = predictions.slice(0, 10).map(p => `#${p.hashtag}`).join(' ');
    if (allHashtags) {
      navigator.clipboard.writeText(allHashtags);
      setCopiedHashtag('all');
      setTimeout(() => setCopiedHashtag(''), 2000);
    }
  };

  const saveApiConfig = (): void => {
    localStorage.setItem('hashtagPredictor_apiConfig', JSON.stringify(apiConfig));
    setShowSettings(false);
    checkApiHealth();
  };

  const getStatusIcon = (): React.ReactNode => {
    switch (apiStatus) {
      case 'connected':
        return <Wifi className="w-3 h-3 text-green-500" />;
      case 'partial':
        return <Wifi className="w-3 h-3 text-yellow-500" />;
      case 'error':
        return <WifiOff className="w-3 h-3 text-red-500" />;
      default:
        return <WifiOff className="w-3 h-3 text-gray-500" />;
    }
  };

  const getStatusText = (): string => {
    switch (apiStatus) {
      case 'connected':
        return apiConfig.apiKey ? `Connected to ${apiConfig.provider.toUpperCase()} AI + RAG` : 'ML Model + RAG Ready';
      case 'partial':
        return 'ML Model + RAG Ready (AI Optional)';
      case 'error':
        return 'API Error - Using Fallback';
      default:
        return 'Connecting to API...';
    }
  };

    const getPredictionMethodInfo = () => {
    switch (predictionMethod) {
      case 'rag_only':
        return 'Using proven hashtags from historical campaign data';
      case 'ai_only':
        return 'Using AI creativity for fresh hashtag suggestions';
      case 'auto':
        return 'Smart combination of AI creativity + proven historical data';
      default:
        return '';
    }
  };

  const getPredictionMethodIcon = () => {
    switch (predictionMethod) {
      case 'rag_only':
        return <Database className="w-4 h-4 text-emerald-600" />;
      case 'ai_only':
        return <Brain className="w-4 h-4 text-purple-600" />;
      case 'auto':
        return <Sparkles className="w-4 h-4 text-blue-600" />;
      default:
        return <Hash className="w-4 h-4 text-gray-600" />;
    }
  };

  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      'government': 'bg-blue-100 text-blue-700',
      'industry': 'bg-green-100 text-green-700',
      'trending': 'bg-purple-100 text-purple-700',
      'rag_existing': 'bg-emerald-100 text-emerald-700',
      'rag_similar': 'bg-teal-100 text-teal-700',
      'rag_frequent': 'bg-cyan-100 text-cyan-700',
      'rag_enhanced': 'bg-indigo-100 text-indigo-700',
      'ml_predicted': 'bg-orange-100 text-orange-700',
      'tn_politics': 'bg-amber-100 text-amber-700',
      'sentiment_based': 'bg-rose-100 text-rose-700',
      'default': 'bg-gray-100 text-gray-700'
    };
    return colors[category] || colors.default;
  };

  const getSourceIcon = (source: string): React.ReactNode => {
    switch (source) {
      case 'existing':
        return <Database className="w-4 h-4 text-emerald-600" />;
      case 'rag_enhanced':
        return <Star className="w-4 h-4 text-indigo-600" />;
      case 'new':
        return <Lightbulb className="w-4 h-4 text-orange-600" />;
      default:
        return <Hash className="w-4 h-4 text-gray-600" />;
    }
  };

  const getSourceColor = (source: string): string => {
    switch (source) {
      case 'existing':
        return 'bg-emerald-100 text-emerald-700';
      case 'rag_enhanced':
        return 'bg-indigo-100 text-indigo-700';
      case 'new':
        return 'bg-orange-100 text-orange-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  const getSentimentIcon = (sentiment: string): React.ReactNode => {
    switch (sentiment) {
      case 'positive':
        return <Smile className="w-4 h-4 text-green-600" />;
      case 'negative':
        return <Frown className="w-4 h-4 text-red-600" />;
      default:
        return <Meh className="w-4 h-4 text-gray-600" />;
    }
  };

  const getSentimentColor = (sentiment: string): string => {
    switch (sentiment) {
      case 'positive':
        return 'bg-green-100 text-green-700';
      case 'negative':
        return 'bg-red-100 text-red-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  const getLanguageIcon = (language: string): React.ReactNode => {
    switch (language) {
      case 'tamil':
        return <span className="text-xs font-bold text-orange-600">TM</span>;
      case 'english':
        return <span className="text-xs font-bold text-blue-600">EN</span>;
      case 'mixed':
        return <span className="text-xs font-bold text-purple-600">MX</span>;
      default:
        return <Languages className="w-3 h-3 text-gray-600" />;
    }
  };

  const getLanguageColor = (language: string): string => {
    switch (language) {
      case 'tamil':
        return 'bg-orange-100 text-orange-700';
      case 'english':
        return 'bg-blue-100 text-blue-700';
      case 'mixed':
        return 'bg-purple-100 text-purple-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-lg">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI Hashtag Predictor with RAG
            </h1>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 bg-white hover:bg-gray-50 rounded-lg transition-colors duration-200 shadow-md border border-gray-200"
            >
              <Settings className="w-5 h-5 text-gray-600" />
            </button>
          </div>
          <p className="text-gray-600 text-lg max-w-3xl mx-auto">
            Advanced AI-powered hashtag prediction with RAG database of proven high-performing hashtags from historical campaigns
          </p>
          <div className="flex items-center justify-center gap-4 mt-3">
            <div className="flex items-center gap-2">
              {getStatusIcon()}
              <span className="text-sm text-gray-600">{getStatusText()}</span>
            </div>
            {ragStats && (
              <div className="flex items-center gap-2">
                <Database className="w-3 h-3 text-emerald-500" />
                <span className="text-sm text-gray-600">
                  RAG: {ragStats.total_rag_hashtags} proven hashtags
                </span>
              </div>
            )}
          </div>
        </div>

        {/* API Configuration Modal */}
        {showSettings && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md max-h-screen overflow-y-auto">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                AI API Configuration
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Provider</label>
                  <select
                    value={apiConfig.provider}
                    onChange={(e) => setApiConfig({...apiConfig, provider: e.target.value})}
                    className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="azure">Azure OpenAI</option>
                    <option value="openai">OpenAI</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">API Key</label>
                  <input
                    type="password"
                    value={apiConfig.apiKey}
                    onChange={(e) => setApiConfig({...apiConfig, apiKey: e.target.value})}
                    placeholder="Enter your API key"
                    className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                {apiConfig.provider === 'azure' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Azure Endpoint</label>
                    <input
                      type="text"
                      value={apiConfig.endpoint}
                      onChange={(e) => setApiConfig({...apiConfig, endpoint: e.target.value})}
                      placeholder="https://your-resource.openai.azure.com"
                      className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Model</label>
                  <select
                    value={apiConfig.model}
                    onChange={(e) => setApiConfig({...apiConfig, model: e.target.value})}
                    className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-4o">GPT-4o</option>
                  </select>
                </div>

                <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg border border-blue-200">
                  <div className="flex items-start gap-2">
                    <Sparkles className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <strong>Enhanced with RAG:</strong> Our system now uses a database of proven high-performing hashtags from historical campaigns to provide better recommendations.
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={saveApiConfig}
                  className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors duration-200 font-medium"
                >
                  Save Configuration
                </button>
                <button
                  onClick={() => setShowSettings(false)}
                  className="flex-1 bg-gray-300 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-400 transition-colors duration-200 font-medium"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Input Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <div className="flex items-center gap-2 mb-4">
                <Target className="w-5 h-5 text-blue-600" />
                <h2 className="text-xl font-semibold text-gray-800">Content Analysis</h2>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Enter your content (Tamil/English)
                  </label>
                  <textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="Enter your content here for hashtag analysis..."
                    className="w-full h-32 p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-700 placeholder-gray-400"
                  />
                  <div className="text-xs text-gray-500 mt-1 flex justify-between">
                    <span>{content.length} characters</span>
                    <span className="text-blue-600">
                      {content.length > 0 ? '✓ Ready for RAG analysis' : 'Enter content to start'}
                    </span>
                  </div>
                </div>
                {/* Prediction Method Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                    <Sliders className="w-4 h-4" />
                    Prediction Method
                  </label>
                  <div className="space-y-3">
                    <label className={`flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-md ${
                      predictionMethod === 'auto'
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                    }`}>
                      <input
                        type="radio"
                        name="predictionMethod"
                        value="auto"
                        checked={predictionMethod === 'auto'}
                        onChange={(e) => setPredictionMethod(e.target.value)}
                        className="w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                      />
                      <div className="flex items-center gap-2 flex-1">
                        <Sparkles className="w-4 h-4 text-blue-600" />
                        <div>
                          <div className="font-medium text-gray-800">Auto (Recommended)</div>
                          <div className="text-xs text-gray-600">Smart combination of AI creativity + proven historical data</div>
                        </div>
                      </div>
                    </label>

                    <label className={`flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-md ${
                      predictionMethod === 'rag_only'
                        ? 'border-emerald-500 bg-emerald-50'
                        : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                    }`}>
                      <input
                        type="radio"
                        name="predictionMethod"
                        value="rag_only"
                        checked={predictionMethod === 'rag_only'}
                        onChange={(e) => setPredictionMethod(e.target.value)}
                        className="w-4 h-4 text-emerald-600 border-gray-300 focus:ring-emerald-500"
                      />
                      <div className="flex items-center gap-2 flex-1">
                        <Database className="w-4 h-4 text-emerald-600" />
                        <div>
                          <div className="font-medium text-gray-800">RAG Only</div>
                          <div className="text-xs text-gray-600">Use proven hashtags from historical campaign data</div>
                        </div>
                      </div>
                    </label>

                    <label className={`flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-md ${
                      predictionMethod === 'ai_only'
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                    }`}>
                      <input
                        type="radio"
                        name="predictionMethod"
                        value="ai_only"
                        checked={predictionMethod === 'ai_only'}
                        onChange={(e) => setPredictionMethod(e.target.value)}
                        className="w-4 h-4 text-purple-600 border-gray-300 focus:ring-purple-500"
                      />
                      <div className="flex items-center gap-2 flex-1">
                        <Brain className="w-4 h-4 text-purple-600" />
                        <div>
                          <div className="font-medium text-gray-800">
                            AI Only 
                            {predictionMethod === 'ai_only' && !apiConfig.apiKey && (
                              <span className="text-red-600 text-xs ml-1">(API Required)</span>
                            )}
                          </div>
                          <div className="text-xs text-gray-600">Use AI creativity for fresh hashtag suggestions</div>
                        </div>
                      </div>
                    </label>
                  </div>
                  
                  <div className="mt-2 p-2 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="flex items-start gap-2">
                      {getPredictionMethodIcon()}
                      <p className="text-xs text-gray-600">{getPredictionMethodInfo()}</p>
                    </div>
                  </div>
                </div>

                {/* Hashtag Count Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                    <Hash className="w-4 h-4" />
                    Number of Hashtags
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min="5"
                      max="20"
                      value={maxHashtags}
                      onChange={(e) => setMaxHashtags(parseInt(e.target.value))}
                      className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-700">{maxHashtags}</span>
                      <span className="text-xs text-gray-500">hashtags</span>
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>5 (Focused)</span>
                    <span>10 (Default)</span>
                    <span>20 (Comprehensive)</span>
                  </div>
                </div>

                {/* Language Preference */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                    <Languages className="w-4 h-4" />
                    Hashtag Language Preference
                  </label>
                  <div className="flex gap-3">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="language"
                        value="tamil"
                        checked={languagePreference === 'tamil'}
                        onChange={(e) => setLanguagePreference(e.target.value)}
                        className="w-4 h-4 text-orange-600 border-gray-300 focus:ring-orange-500"
                      />
                      <span className="text-sm text-gray-700 flex items-center gap-1">
                        <span className="text-orange-600 font-bold">TM</span>
                        Tamil
                      </span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="language"
                        value="english"
                        checked={languagePreference === 'english'}
                        onChange={(e) => setLanguagePreference(e.target.value)}
                        className="w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="text-sm text-blue-700">English</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="language"
                        value="both"
                        checked={languagePreference === 'both'}
                        onChange={(e) => setLanguagePreference(e.target.value)}
                        className="w-4 h-4 text-purple-600 border-gray-300 focus:ring-purple-500"
                      />
                      <span className="text-sm text-purple-700">Both / Mixed</span>
                    </label>
                  </div>
                </div>

                {error && (
                  <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                    <span className="text-red-700 text-sm">{error}</span>
                  </div>
                )}

                <div className="flex flex-wrap gap-3">
                  <button
                    onClick={handlePredict}
                    disabled={!content.trim() || loading || apiStatus === 'disconnected'}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-all duration-200 shadow-lg hover:shadow-xl"
                  >
                    {loading ? (
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Brain className="w-5 h-5" />
                    )}
                    {loading ? 'AI + RAG Analyzing...' : 'Predict with RAG'}
                  </button>
                  
                  {predictions.length > 0 && (
                    <button
                      onClick={copyAllHashtags}
                      className="flex items-center gap-2 px-4 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 font-medium transition-all duration-200 shadow-lg"
                    >
                      {copiedHashtag === 'all' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      Copy Top 10
                    </button>
                  )}
                  
                  <button
                    onClick={checkApiHealth}
                    className="flex items-center gap-2 px-4 py-3 bg-gray-600 text-white rounded-xl hover:bg-gray-700 font-medium transition-all duration-200"
                  >
                    <Wifi className="w-4 h-4" />
                    Check Status
                  </button>
                </div>

                {/* Sample Content Buttons */}
                <div>
                  <p className="text-sm text-gray-600 mb-3 flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    Try sample content:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {sampleContents.map((sample, index) => (
                      <button
                        key={index}
                        onClick={() => setContent(sample)}
                        className="text-left p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors duration-200 hover:border-gray-300"
                      >
                        {sample.length > 80 ? `${sample.substring(0, 80)}...` : sample}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Strategy Selection */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 mt-6">
              <div className="flex items-center gap-2 mb-4">
                <Filter className="w-5 h-5 text-purple-600" />
                <h3 className="text-xl font-semibold text-gray-800">Hashtag Strategies</h3>
                <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
                  {selectedStrategies.length} selected
                </span>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {hashtagStrategies.map((strategy) => (
                  <label
                    key={strategy.id}
                    className={`flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-md ${
                      selectedStrategies.includes(strategy.id)
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedStrategies.includes(strategy.id)}
                      onChange={() => handleStrategyChange(strategy.id)}
                      className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                    />
                    <div className="flex items-center gap-2 flex-1">
                      {strategy.icon}
                      <div>
                        <div className="font-medium text-gray-800">{strategy.name}</div>
                        <div className="text-xs text-gray-600">{strategy.description}</div>
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Enhanced Sentiment Analysis Options */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 mt-6">
              <div className="flex items-center gap-2 mb-4">
                <Heart className="w-5 h-5 text-pink-600" />
                <h3 className="text-xl font-semibold text-gray-800">Sentiment Analysis</h3>
              </div>
              
              <div className="space-y-4">
                <label className="flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-md border-gray-200 bg-gray-50 hover:border-gray-300">
                  <input
                    type="checkbox"
                    checked={enableSentimentAnalysis}
                    onChange={(e) => setEnableSentimentAnalysis(e.target.checked)}
                    className="w-4 h-4 text-pink-600 border-gray-300 rounded focus:ring-pink-500"
                  />
                  <div>
                    <div className="font-medium text-gray-800">Enable Enhanced Sentiment Analysis</div>
                    <div className="text-xs text-gray-600">Analyze hashtag sentiment, emotional tone, and political alignment</div>
                  </div>
                </label>

                {enableSentimentAnalysis && (
                  <div className="ml-7 space-y-3">
                    <p className="text-sm font-medium text-gray-700 mb-2">Filter by sentiment:</p>
                    <div className="flex gap-2 flex-wrap">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="sentiment"
                          value="all"
                          checked={sentimentFilter === 'all'}
                          onChange={(e) => setSentimentFilter(e.target.value)}
                          className="w-4 h-4 text-gray-600 border-gray-300 focus:ring-gray-500"
                        />
                        <span className="text-sm text-gray-700">All</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="sentiment"
                          value="positive"
                          checked={sentimentFilter === 'positive'}
                          onChange={(e) => setSentimentFilter(e.target.value)}
                          className="w-4 h-4 text-green-600 border-gray-300 focus:ring-green-500"
                        />
                        <span className="text-sm text-green-700 flex items-center gap-1">
                          <Smile className="w-3 h-3" />
                          Positive
                        </span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="sentiment"
                          value="negative"
                          checked={sentimentFilter === 'negative'}
                          onChange={(e) => setSentimentFilter(e.target.value)}
                          className="w-4 h-4 text-red-600 border-gray-300 focus:ring-red-500"
                        />
                        <span className="text-sm text-red-700 flex items-center gap-1">
                          <Frown className="w-3 h-3" />
                          Negative
                        </span>
                      </label>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* RAG Analysis Results */}
            {ragAnalysis && (
              <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 mt-6">
                <div className="flex items-center gap-2 mb-4">
                  <Database className="w-5 h-5 text-emerald-600" />
                  <h3 className="text-xl font-semibold text-gray-800">RAG Database Analysis</h3>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="text-center p-3 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg border border-emerald-200">
                    <div className="text-sm text-gray-600">Existing</div>
                    <div className="font-semibold text-emerald-800">{ragAnalysis.existing_hashtags_found}</div>
                  </div>
                  <div className="text-center p-3 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg border border-indigo-200">
                    <div className="text-sm text-gray-600">Enhanced</div>
                    <div className="font-semibold text-indigo-800">{ragAnalysis.rag_enhanced_hashtags}</div>
                  </div>
                  <div className="text-center p-3 bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg border border-orange-200">
                    <div className="text-sm text-gray-600">New</div>
                    <div className="font-semibold text-orange-800">{ragAnalysis.new_hashtags_suggested}</div>
                  </div>
                  <div className="text-center p-3 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                    <div className="text-sm text-gray-600">Total RAG</div>
                    <div className="font-semibold text-purple-800">{ragAnalysis.total_rag_hashtags}</div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Top RAG Hashtags</h4>
                  <div className="flex flex-wrap gap-2">
                    {ragAnalysis.top_rag_hashtags.slice(0, 8).map((tag, index) => (
                      <span 
                        key={index}
                        className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded-md text-xs font-medium flex items-center gap-1"
                        title={`Used ${tag.frequency} times`}
                      >
                        #{tag.hashtag} ({tag.frequency})
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Enhanced Predictions Section */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Globe className="w-5 h-5 text-green-600" />
                  <h2 className="text-xl font-semibold text-gray-800">AI + RAG Predictions</h2>
                </div>
                {predictions.length > 0 && (
                  <div className="flex items-center gap-2">
                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                      {predictions.length} hashtags
                    </span>
                    {predictionSource.includes('RAG') && (
                      <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-medium flex items-center gap-1">
                        <Database className="w-3 h-3" />
                        RAG Enhanced
                      </span>
                    )}
                  </div>
                )}
              </div>

              {loading && (
                <div className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                                       <p className="text-gray-600 font-medium">
                      {predictionMethod === 'rag_only' ? 'RAG analyzing content...' : 
                       predictionMethod === 'ai_only' ? 'AI analyzing content...' : 
                       'AI + RAG analyzing content...'}
                    </p>
                    <p className="text-gray-500 text-sm mt-1">
                      {predictionMethod === 'rag_only' ? 'Matching against proven hashtag database' : 
                       predictionMethod === 'ai_only' ? 'Generating creative hashtag suggestions' : 
                       'Combining AI creativity with proven hashtags'}
                    </p>
                  </div>
                </div>
              )}

              {!loading && predictions.length === 0 && content && (
                <div className="text-center py-12 text-gray-500">
                  <Hash className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="font-medium">No relevant hashtags found</p>
                  <p className="text-sm mt-1">Try different content or check your API configuration</p>
                </div>
              )}

              {!loading && predictions.length === 0 && !content && (
                <div className="text-center py-12 text-gray-500">
                  <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="font-medium">Enter content to get RAG-enhanced predictions</p>
                  <p className="text-sm mt-1">Our system will match against proven high-performing hashtags</p>
                </div>
              )}

              {predictions.length > 0 && (
                <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                  {predictions.map((prediction, index) => (
                    <div 
                      key={prediction.hashtag}
                      className="p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl border border-gray-100 hover:shadow-lg transition-all duration-200 hover:border-blue-200"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <span className="w-7 h-7 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                            {index + 1}
                          </span>
                          <span className="font-bold text-gray-800 text-lg">{prediction.hashtag}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-bold text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                            {prediction.score}
                          </span>
                          
                          {/* Source indicator */}
                          <span
                            className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getSourceColor(prediction.source)}`}
                            title={`Source: ${prediction.source}`}
                          >
                            {getSourceIcon(prediction.source)}
                            {prediction.source}
                          </span>

                          {/* RAG frequency indicator */}
                          {prediction.frequency_in_rag && (
                            <span
                              className="flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700"
                              title={`Used ${prediction.frequency_in_rag} times in historical data`}
                            >
                              <Clock className="w-3 h-3" />
                              {prediction.frequency_in_rag}x
                            </span>
                          )}
                          
                          {/* Show sentiment if available */}
                          {prediction.sentiment && (
                            <span
                              className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium capitalize ${getSentimentColor(prediction.sentiment)}`}
                              title={`Sentiment: ${prediction.sentiment}`}
                            >
                              {getSentimentIcon(prediction.sentiment)}
                              {prediction.sentiment}
                            </span>
                          )}
                          
                          {/* Show language if available */}
                          {prediction.language && (
                            <span
                              className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getLanguageColor(prediction.language)}`}
                              title={`Language: ${prediction.language}`}
                            >
                              {getLanguageIcon(prediction.language)}
                            </span>
                          )}
                          
                          <button
                            onClick={() => copyHashtag(prediction.hashtag)}
                            className="p-2 hover:bg-blue-100 rounded-lg transition-colors duration-200"
                            title="Copy hashtag"
                          >
                            {copiedHashtag === prediction.hashtag ? (
                              <Check className="w-4 h-4 text-green-600" />
                            ) : (
                              <Copy className="w-4 h-4 text-gray-500" />
                            )}
                          </button>
                        </div>
                      </div>
                      
                      <div className="text-sm text-gray-600 mb-3 leading-relaxed">
                        {prediction.reasoning}
                      </div>

                      {/* Show similar content examples for RAG hashtags */}
                      {prediction.similar_content_examples && prediction.similar_content_examples.length > 0 && (
                        <div className="mb-3">
                          <div className="text-xs text-gray-500 mb-1 font-medium">Similar historical content:</div>
                          <div className="space-y-1">
                            {prediction.similar_content_examples.slice(0, 2).map((example, idx) => (
                              <div key={idx} className="text-xs text-gray-600 bg-gray-50 p-2 rounded italic">
                                "{example}"
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium capitalize ${getCategoryColor(prediction.category)}`}>
                          {prediction.category.replace('_', ' ')}
                        </span>
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500 ease-out"
                              style={{ width: `${Math.min(prediction.score, 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500 font-medium">
                            {Math.min(prediction.score, 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Enhanced Top Hashtags Section with Source Filter */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-purple-600" />
                  <h3 className="text-xl font-semibold text-gray-800">Top Performing Hashtags</h3>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={hashtagSourceFilter}
                    onChange={(e) => {
                      setHashtagSourceFilter(e.target.value);
                      loadTopHashtags();
                    }}
                    className="text-sm p-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="all">All Sources</option>
                    <option value="rag">RAG Database</option>
                    <option value="ml">ML Model</option>
                  </select>
                  <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
                    {hashtagSourceFilter === 'rag' ? 'Historical Data' : hashtagSourceFilter === 'ml' ? 'ML Predictions' : 'Combined'}
                  </span>
                </div>
              </div>
              
              {topHashtags.length > 0 && (
                <div className="grid grid-cols-1 gap-2">
                  {topHashtags.slice(0, 10).map((hashtag, index) => (
                    <div 
                      key={hashtag.hashtag}
                      className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-100 hover:shadow-md transition-all duration-200"
                    >
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                          {index + 1}
                        </span>
                        <span className="font-medium text-gray-800">#{hashtag.hashtag}</span>
                        {hashtag.source && (
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSourceColor(hashtag.source)}`}>
                            {hashtag.source === 'rag' ? 'Historical' : hashtag.source === 'ml' ? 'ML' : hashtag.source}
                          </span>
                        )}
                        {hashtag.category && (
                          <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full text-xs">
                            {hashtag.category}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600 font-medium">
                          {hashtag.score.toFixed(0)}
                        </span>
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(`#${hashtag.hashtag}`);
                            setCopiedTopHashtag(hashtag.hashtag);
                            setTimeout(() => setCopiedTopHashtag(''), 2000);
                          }}
                          className="p-1 hover:bg-purple-100 rounded transition-colors duration-200"
                          title="Copy hashtag"
                        >
                          {copiedTopHashtag === hashtag.hashtag ? (
                            <Check className="w-3 h-3 text-green-600" />
                          ) : (
                            <Copy className="w-3 h-3 text-gray-500" />
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* RAG Database Stats */}
            {ragStats && (
              <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
                <div className="flex items-center gap-2 mb-4">
                  <Database className="w-5 h-5 text-emerald-600" />
                  <h3 className="text-xl font-semibold text-gray-800">RAG Database Stats</h3>
                  <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-medium">
                    Live Database
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="text-center p-3 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg border border-emerald-200">
                    <div className="text-sm text-gray-600">Total Hashtags</div>
                    <div className="font-semibold text-emerald-800">{ragStats.total_rag_hashtags}</div>
                  </div>
                  <div className="text-center p-3 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                    <div className="text-sm text-gray-600">Content Examples</div>
                    <div className="font-semibold text-blue-800">{ragStats.total_contexts}</div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Categories Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    {ragStats.categories && Object.entries(ragStats.categories).map(([category, count]) => (
                      <span 
                        key={category}
                        className="px-2 py-1 bg-blue-100 text-blue-700 rounded-md text-xs font-medium"
                      >
                        {category}: {count as number}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Top Trending Hashtags Component */}
            <TopTrendingHashtags apiConfig={apiConfig} />
          </div>
        </div>
      </div>
    </div>
  );
};

// TopTrendingHashtags component
interface TopTrendingHashtagsProps {
  apiConfig: ApiConfig;
}

const TopTrendingHashtags: React.FC<TopTrendingHashtagsProps> = ({ apiConfig }) => {
  const [platform, setPlatform] = useState<string>('all');
  const [hashtags, setHashtags] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [copiedHashtag, setCopiedHashtag] = useState<string>('');

  const API_BASE_URL = 'http://localhost:8000';
  const { apiKey, provider, endpoint, model } = apiConfig;

  const fetchTrending = async (selectedPlatform: string): Promise<void> => {
    setLoading(true);
    setError('');
    setHashtags([]);
    try {
      const configToSend = apiKey
        ? { api_key: apiKey, provider, endpoint, model }
        : undefined;
      const resp = await fetch(`${API_BASE_URL}/top-trending-hashtags`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          platform: selectedPlatform,
          count: 10,
          config: configToSend,
        }),
      });
      if (!resp.ok) throw new Error('Failed to fetch trending hashtags');
      const data = await resp.json();
      setHashtags(data.top_trending_hashtags || []);
    } catch (e: any) {
      setError(e.message || 'Error fetching trending hashtags');
    } finally {
      setLoading(false);
    }
  };

  const copyHashtag = (hashtag: string): void => {
    navigator.clipboard.writeText(`#${hashtag}`);
    setCopiedHashtag(hashtag);
    setTimeout(() => setCopiedHashtag(''), 2000);
  };

  useEffect(() => {
    fetchTrending(platform);
  }, [platform]);

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-pink-600" />
        <h3 className="text-xl font-semibold text-gray-800">Top Trending Hashtags</h3>
        <span className="px-2 py-1 bg-pink-100 text-pink-700 rounded-full text-xs font-medium">
          Live (AI Powered)
        </span>
      </div>
      <div className="mb-4 flex gap-2">
        <select
          value={platform}
          onChange={e => setPlatform(e.target.value)}
          className="p-2 border border-gray-200 rounded-lg text-sm"
        >
          <option value="all">All Platforms</option>
          <option value="Instagram">Instagram</option>
          <option value="Facebook">Facebook</option>
          <option value="Twitter">Twitter</option>
        </select>
        <button
          onClick={() => fetchTrending(platform)}
          className="px-3 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
          disabled={loading}
        >
          Refresh
        </button>
      </div>
      {loading && <div className="text-gray-500 py-6">Loading trending hashtags...</div>}
      {error && <div className="text-red-600 py-2">{error}</div>}
      {!loading && !error && hashtags.length > 0 && (
        <div className="space-y-2">
          {hashtags.map((item, idx) => (
            <div key={item.hashtag + idx} className="flex items-center justify-between p-2 bg-gradient-to-r from-pink-50 to-blue-50 rounded-lg border border-pink-100">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 bg-pink-500 text-white rounded-full flex items-center justify-center text-xs font-bold">{idx + 1}</span>
                <span className="font-medium text-gray-800 text-sm">{item.hashtag}</span>
                <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{item.platform}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 max-w-24 truncate">{item.reason}</span>
                <button
                  onClick={() => copyHashtag(item.hashtag)}
                  className="p-1 hover:bg-pink-100 rounded transition-colors duration-200"
                  title="Copy hashtag"
                >
                  {copiedHashtag === item.hashtag ? (
                    <Check className="w-4 h-4 text-green-600" />
                  ) : (
                    <Copy className="w-4 h-4 text-gray-500" />
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
      {!loading && !error && hashtags.length === 0 && (
        <div className="text-gray-400 py-6">No trending hashtags found.</div>
      )}
    </div>
  );
};

export default HashtagPredictor;