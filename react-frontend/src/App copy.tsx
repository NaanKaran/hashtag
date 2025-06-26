import React, { useState, useEffect } from 'react';
import { Hash, Target, TrendingUp, Globe, Zap, Copy, Check, AlertCircle, Wifi, WifiOff } from 'lucide-react';

const HashtagPredictor = () => {
  const [content, setContent] = useState('');
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [copiedHashtag, setCopiedHashtag] = useState('');
  const [analysisDetails, setAnalysisDetails] = useState(null);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('unknown');
  const [processingTime, setProcessingTime] = useState(0);

  // API Configuration
  const API_BASE_URL = 'http://localhost:8000'; // Change this to your backend URL

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        setApiStatus('online');
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      setApiStatus('offline');
      console.error('API health check failed:', error);
    }
  };

  const handlePredict = async () => {
    if (!content.trim()) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: content,
          max_predictions: 15
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get predictions');
      }

      const data = await response.json();
      
      setPredictions(data.predictions);
      setAnalysisDetails(data.analysis);
      setProcessingTime(data.processing_time);
      setApiStatus('online');
      
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error.message || 'Failed to connect to AI service. Please check if the backend is running.');
      setApiStatus('offline');
      
      // Fallback to mock data for demo purposes
      setMockPredictions();
    } finally {
      setLoading(false);
    }
  };

  // Fallback mock data when API is unavailable
  const setMockPredictions = () => {
    const mockPredictions = [
      { hashtag: 'tamilnadu', score: 95, themes: ['government', 'development'], reasoning: 'High relevance to Tamil Nadu content', confidence: 0.92 },
      { hashtag: 'employment', score: 88, themes: ['employment', 'youth'], reasoning: 'Employment-related keywords detected', confidence: 0.85 },
      { hashtag: 'development', score: 82, themes: ['development'], reasoning: 'Development theme identified', confidence: 0.78 },
      { hashtag: 'digitaltamilnadu', score: 76, themes: ['technology', 'development'], reasoning: 'Technology and development focus', confidence: 0.72 },
      { hashtag: 'youthempowerment', score: 70, themes: ['youth', 'employment'], reasoning: 'Youth-related content detected', confidence: 0.68 }
    ];
    
    const mockAnalysis = {
      themes: { government: 0.8, employment: 0.6, development: 0.7 },
      keywords: ['tamil', 'nadu', 'government', 'youth', 'employment'],
      content_length: content.length,
      word_count: content.split(' ').length
    };
    
    setPredictions(mockPredictions);
    setAnalysisDetails(mockAnalysis);
    setProcessingTime(0.5);
  };

  const copyHashtag = (hashtag) => {
    navigator.clipboard.writeText(`#${hashtag}`);
    setCopiedHashtag(hashtag);
    setTimeout(() => setCopiedHashtag(''), 2000);
  };

  const copyAllHashtags = () => {
    const allHashtags = predictions.slice(0, 10).map(p => `#${p.hashtag}`).join(' ');
    navigator.clipboard.writeText(allHashtags);
    setCopiedHashtag('all');
    setTimeout(() => setCopiedHashtag(''), 2000);
  };

  const sampleContents = [
    "தமிழ்நாடு அரசு இளைஞர்களுக்கான புதிய வேலைவாய்ப்பு திட்டத்தை அறிவித்துள்ளது. தொழில்நுட்ப பயிற்சி மற்றும் திறன் மேம்பாட்டு திட்டங்கள் மூலம் இளைஞர்கள் மேம்பட்ட வேலைகளை பெறலாம்.",
    "Tamil Nadu government launches new employment scheme for youth development and skill training programs to create better job opportunities.",
    "Healthcare infrastructure development in Tamil Nadu with new hospitals and medical facilities for rural areas.",
    "Women empowerment initiatives and safety measures implemented across Tamil Nadu for gender equality and social justice.",
    "Digital Tamil Nadu initiative brings technology and innovation to transform governance and public services for citizens."
  ];

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-blue-600';
    return 'text-gray-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl">
              <Hash className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI Hashtag Predictor
            </h1>
            <div className="flex items-center gap-2 ml-4">
              {apiStatus === 'online' ? (
                <div className="flex items-center gap-1 text-green-600">
                  <Wifi className="w-4 h-4" />
                  <span className="text-xs">API Online</span>
                </div>
              ) : (
                <div className="flex items-center gap-1 text-red-600">
                  <WifiOff className="w-4 h-4" />
                  <span className="text-xs">API Offline</span>
                </div>
              )}
            </div>
          </div>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            ML-powered hashtag prediction using TF-IDF and SVD for Tamil content analysis
          </p>
          {processingTime > 0 && (
            <p className="text-sm text-gray-500 mt-2">
              Last prediction: {processingTime.toFixed(2)}s processing time
            </p>
          )}
        </div>

        {/* API Status Warning */}
        {apiStatus === 'offline' && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-yellow-600" />
              <div>
                <h3 className="font-medium text-yellow-800">Backend API Unavailable</h3>
                <p className="text-sm text-yellow-700">
                  The Python backend is not running. Start the FastAPI server with: <code className="bg-yellow-100 px-1 rounded">uvicorn main:app --reload</code>
                </p>
                <p className="text-xs text-yellow-600 mt-1">Demo mode: Using fallback predictions</p>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-600" />
              <div>
                <h3 className="font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-6">
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
                    placeholder="Enter your content here... தமிழில் அல்லது ஆங்கிலத்தில் உள்ளடக்கத்தை உள்ளிடவும்"
                    className="w-full h-32 p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-700"
                  />
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={handlePredict}
                    disabled={!content.trim() || loading}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-all duration-200 shadow-lg hover:shadow-xl"
                  >
                    {loading ? (
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Zap className="w-5 h-5" />
                    )}
                    {loading ? 'Analyzing...' : 'Predict Hashtags'}
                  </button>
                  
                  {predictions.length > 0 && (
                    <button
                      onClick={copyAllHashtags}
                      className="flex items-center gap-2 px-4 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 font-medium transition-all duration-200"
                    >
                      {copiedHashtag === 'all' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      Copy Top 10
                    </button>
                  )}
                </div>

                {/* Sample Content Buttons */}
                <div>
                  <p className="text-sm text-gray-600 mb-2">Try sample content:</p>
                  <div className="space-y-2">
                    {sampleContents.map((sample, index) => (
                      <button
                        key={index}
                        onClick={() => setContent(sample)}
                        className="text-left w-full p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors duration-200"
                      >
                        {sample.length > 80 ? `${sample.substring(0, 80)}...` : sample}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Details */}
            {analysisDetails && (
              <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 mt-6">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  <h3 className="text-xl font-semibold text-gray-800">Content Analysis</h3>
                </div>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-700 mb-2">Detected Themes</h4>
                    <div className="space-y-2">
                      {Object.keys(analysisDetails.themes || {}).length > 0 ? (
                        Object.entries(analysisDetails.themes).map(([theme, score]) => (
                          <div key={theme} className="flex items-center justify-between">
                            <span className="capitalize text-gray-600">{theme}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                                  style={{ width: `${Math.min(score * 100, 100)}%` }}
                                />
                              </div>
                              <span className="text-sm text-gray-500">{(score * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <p className="text-gray-500 text-sm">No specific themes detected</p>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-700 mb-2">Key Terms & Stats</h4>
                    <div className="space-y-3">
                      <div className="flex flex-wrap gap-2">
                        {(analysisDetails.keywords || []).slice(0, 8).map((keyword, index) => (
                          <span 
                            key={index}
                            className="px-2 py-1 bg-blue-100 text-blue-700 rounded-md text-sm"
                          >
                            {keyword}
                          </span>
                        ))}
                      </div>
                      <div className="text-sm text-gray-600 space-y-1">
                        <div>Content Length: {analysisDetails.content_length || 0} characters</div>
                        <div>Word Count: {analysisDetails.word_count || 0} words</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Predictions Section */}
          <div>
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 sticky top-4">
              <div className="flex items-center gap-2 mb-4">
                <Globe className="w-5 h-5 text-green-600" />
                <h2 className="text-xl font-semibold text-gray-800">AI Predictions</h2>
              </div>

              {loading && (
                <div className="flex items-center justify-center py-8">
                  <div className="text-center">
                    <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                    <p className="text-gray-600">
                      {apiStatus === 'online' ? 'Processing with AI...' : 'Generating predictions...'}
                    </p>
                  </div>
                </div>
              )}

              {!loading && predictions.length === 0 && content && (
                <div className="text-center py-8 text-gray-500">
                  <Hash className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No relevant hashtags found. Try different content.</p>
                </div>
              )}

              {!loading && predictions.length === 0 && !content && (
                <div className="text-center py-8 text-gray-500">
                  <Hash className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Enter content above to get AI hashtag predictions</p>
                </div>
              )}

              {predictions.length > 0 && (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {predictions.map((prediction, index) => (
                    <div 
                      key={prediction.hashtag}
                      className="p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl border border-gray-100 hover:shadow-md transition-all duration-200"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="w-6 h-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                            {index + 1}
                          </span>
                          <span className="font-semibold text-gray-800">#{prediction.hashtag}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-medium ${getScoreColor(prediction.score)}`}>
                            {prediction.score}
                          </span>
                          <button
                            onClick={() => copyHashtag(prediction.hashtag)}
                            className="p-1 hover:bg-blue-100 rounded transition-colors duration-200"
                          >
                            {copiedHashtag === prediction.hashtag ? (
                              <Check className="w-4 h-4 text-green-600" />
                            ) : (
                              <Copy className="w-4 h-4 text-gray-500" />
                            )}
                          </button>
                        </div>
                      </div>
                      
                      <div className="text-xs text-gray-600 mb-2">
                        {prediction.reasoning}
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div className="flex flex-wrap gap-1">
                          {prediction.themes.map((theme, idx) => (
                            <span 
                              key={idx}
                              className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs capitalize"
                            >
                              {theme}
                            </span>
                          ))}
                        </div>
                        {prediction.confidence && (
                          <span className={`text-xs px-2 py-1 rounded-full ${getConfidenceColor(prediction.confidence)}`}>
                            {(prediction.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HashtagPredictor;