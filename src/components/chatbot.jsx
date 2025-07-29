import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './chatbot.css'; // Make sure your CSS file is named correctly
import medicAvatar from '../assets/medic-avatar.png';

const TOTAL_CLINICAL_QUESTIONS = 13;

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(`session_guest_${Math.random().toString(36).substring(2)}`);
  const [username, setUsername] = useState('Guest');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const chatContainerRef = useRef(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const BACKEND_URL = 'https://localhost:8443';

  // Create axios instance with interceptors for JWT
  const api = axios.create({
    baseURL: BACKEND_URL,
  });

  // Request interceptor to add JWT token
  api.interceptors.request.use(
    (config) => {
      const token = localStorage.getItem('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => Promise.reject(error)
  );

  // Response interceptor to handle token refresh
  api.interceptors.response.use(
    (response) => response,
    async (error) => {
      const originalRequest = error.config;
      
      if (error.response?.status === 401 && !originalRequest._retry) {
        originalRequest._retry = true;
        
        try {
          const refreshToken = localStorage.getItem('refresh_token');
          if (!refreshToken) {
            throw new Error('No refresh token available');
          }
          
          const response = await axios.post(`${BACKEND_URL}/auth/refresh`, {
            refresh_token: refreshToken
          });
          
          const { access_token, refresh_token: newRefreshToken } = response.data;
          localStorage.setItem('access_token', access_token);
          localStorage.setItem('refresh_token', newRefreshToken);
          
          // Retry the original request with new token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return api(originalRequest);
        } catch (refreshError) {
          console.error('Token refresh failed:', refreshError);
          handleLogout();
          return Promise.reject(refreshError);
        }
      }
      
      return Promise.reject(error);
    }
  );

  // Check login status on mount
  useEffect(() => {
    const accessToken = localStorage.getItem('access_token');
    const refreshToken = localStorage.getItem('refresh_token');
    const storedUsername = localStorage.getItem('username');
    let currentSessionId = sessionId;
    
    if (accessToken && refreshToken && storedUsername) {
      setUsername(storedUsername);
      setIsLoggedIn(true);
      const newSessionId = `session_${storedUsername}_${Math.random().toString(36).substring(2)}`;
      setSessionId(newSessionId);
      currentSessionId = newSessionId;
    }
    
    // Get initial greeting from backend with correct session ID
    getInitialGreeting(currentSessionId);
  }, []);
  
  // Function to get initial greeting from backend
  const getInitialGreeting = async (sessionIdToUse = sessionId) => {
    try {
      // Send empty message to trigger greeting
      const response = await api.post('/api/query/test', {
        message: "__INITIAL_GREETING__",
        session_id: sessionIdToUse,
      });
      const data = response.data;
      
      if (data.response) {
        setMessages([{
          role: 'bot',
          content: data.response,
          timestamp: new Date().toLocaleTimeString(),
        }]);
      }
    } catch (error) {
      console.error('Failed to get initial greeting:', error);
      // Fallback greeting if API fails
      setMessages([{
        role: 'bot',
        content: 'Hello! May I know your name?',
        timestamp: new Date().toLocaleTimeString(),
      }]);
    }
  };

  // Scroll to bottom of chat
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Submit query to backend
  const submitQuery = async () => {
    if (!input.trim()) return;
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await api.post('/api/query/test', {
        message: input,
        session_id: sessionId,
      });
      const data = response.data;

      // Handle stepwise clinical interview
      if (data.next_question) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            content: data.next_question,
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      } else if (data.diagnosis) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            content: `<strong>Diagnosis Summary:</strong><br>${data.diagnosis}`,
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      } else if (data.response) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            content: data.response,
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            content: "I'm here to help. Please provide more information.",
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: `Error: Failed to process query - ${error.message}`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle document upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    try {
      const response = await api.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const data = response.data;
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: `<strong>Document processed:</strong> ${data.file_name || file.name}<br>${data.analysis?.response || 'No summary available.'}`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: `Error: Failed to process document - ${error.message}`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Handle logout
  const handleLogout = async () => {
    try {
      // Call backend logout endpoint
      await api.post('/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear all authentication data
      localStorage.removeItem('username');
      localStorage.removeItem('user_name');
      localStorage.removeItem('user_email');
      localStorage.removeItem('user_id');
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      setIsLoggedIn(false);
      setUsername('Guest');
      setSessionId(`session_guest_${Math.random().toString(36).substring(2)}`);
      setMessages([]);
      navigate('/login');
    }
  };

  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  };

  
  const handlePlusClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  
  const answeredQuestions = messages.filter(msg => msg.role === 'user').length;


  const lastBotMessage = messages.slice().reverse().find(msg => msg.role === 'bot');
  const inputPlaceholder = lastBotMessage
    ? lastBotMessage.content.replace(/<[^>]+>/g, '').split('<br>')[0]
    : 'Describe your symptoms or answer the question...';

  return (
    <div className="bg-gradient-to-r from-blue-50 to-white py-12">
      <div className="w-full px-4 md:px-8 lg:px-12">
        <div className="flex flex-col md:flex-row gap-6">
          {isLoggedIn && (
            <div className="w-full md:w-1/4 bg-blue-100 p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold text-blue-800 mb-4">Welcome, {username}</h2>
              <h3 className="text-lg font-medium text-gray-700 mb-2">Upload Medical Document</h3>
              <input
                type="file"
                accept=".pdf,.txt,.docx"
                onChange={handleFileUpload}
                disabled={isLoading}
                className="w-full p-2 mb-4 border border-gray-300 rounded-lg hidden"
                ref={fileInputRef}
              />
              <button
                className="w-full bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition"
                onClick={handleLogout}
                disabled={isLoading}
              >
                Logout
              </button>
            </div>
          )}
          <div className="w-full">
            <div className="bg-white p-8 rounded-lg shadow-md">
              <div className="main-header">
                <h1 className="text-3xl font-bold text-gray-800 mb-2">CURA</h1>
                <p className="text-gray-600">Your AI-powered medical assistant</p>
              </div>
              {/* Progress Bar */}
              <div className="progress-bar mb-4">
                <span>Progress: {answeredQuestions}/{TOTAL_CLINICAL_QUESTIONS} questions answered</span>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full"
                    style={{ width: `${(answeredQuestions / TOTAL_CLINICAL_QUESTIONS) * 100}%` }}
                  ></div>
                </div>
              </div>
              {/* Chat Area */}
              <div className="chat-area flex flex-col" ref={chatContainerRef} style={{ maxHeight: '400px', overflowY: 'auto' }}>
                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`chat-message-wrapper ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`chat-message ${msg.role === 'user' ? 'user' : 'bot'} p-4 mb-4 rounded-lg`}>
                      <div className="flex items-start gap-2">
                        {msg.role === 'bot' ? (
                          <img 
                            src={medicAvatar} 
                            alt="Medical Assistant" 
                            className="w-8 h-8 rounded-full flex-shrink-0"
                          />
                        ) : (
                          <span className="font-semibold text-blue-600">You</span>
                        )}
                        <div className="flex-1">
                          <div 
                            dangerouslySetInnerHTML={{ __html: msg.content }}
                            className="mb-2"
                          />
                          <small className="text-gray-500 text-xs">{msg.timestamp}</small>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              {/* Input Area */}
              <div className="chat-input mt-4 flex gap-4 items-center">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your medical query..."
                  disabled={isLoading}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 resize-vertical"
                />
                <button
                  className="bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition w-10 flex items-center justify-center"
                  onClick={handlePlusClick}
                  disabled={isLoading}
                  title="Upload File"
                >
                  +
                </button>
                <button
                  className="bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition w-24"
                  onClick={submitQuery}
                  disabled={isLoading}
                >
                  {isLoading ? 'Sending...' : 'Send'}
                </button>
                <input
                  type="file"
                  accept=".pdf,.txt,.docx"
                  onChange={handleFileUpload}
                  disabled={isLoading}
                  className="hidden"
                  ref={fileInputRef}
                />
              </div>
            </div>
            <div className="footer mt-6 bg-blue-600 text-white p-4 rounded-lg text-center">
              <p><strong className="text-yellow-300 text-2xl">⚠️ Important Medical Disclaimer:</strong></p>
              <strong className="text-l"><p>This chatbot provides informational content only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.</p>
              <p><strong>In case of emergency, call your local emergency services immediately.</strong></p></strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;