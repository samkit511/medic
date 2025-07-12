import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './Chatbot.css';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(`session_guest_${Math.random().toString(36).substring(2)}`);
  const [username, setUsername] = useState('Guest');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const chatContainerRef = useRef(null);
  const navigate = useNavigate();

  const BACKEND_URL = 'http://localhost:8000';

  // Check login status on mount
  useEffect(() => {
    const storedUsername = localStorage.getItem('username');
    if (storedUsername) {
      setUsername(storedUsername);
      setIsLoggedIn(true);
      setSessionId(`session_${storedUsername}_${Math.random().toString(36).substring(2)}`);
    }
  }, []);

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
    setMessages([...messages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/query`, {
        message: input,
        session_id: sessionId,
      });
      const data = response.data;
      const botMessage = {
        role: 'bot',
        content: `
          ${data.emergency ? '<div class="emergency-alert">EMERGENCY: SEEK IMMEDIATE MEDICAL ATTENTION!</div>' : ''}
          <div class="urgency-${data.urgency_level || 'low'}">
            ${data.response}<br>
            <strong>Confidence:</strong> ${(data.confidence || 0).toFixed(2)}<br>
            <strong>Explanation:</strong> ${data.explanation || 'N/A'}<br>
            <strong>Urgency:</strong> ${data.urgency_level || 'N/A'}<br>
            ${data.disclaimer ? `<strong>Disclaimer:</strong> ${data.disclaimer}` : ''}
            ${data.possible_conditions?.length ? `<br><strong>Possible Conditions:</strong> ${data.possible_conditions.join(', ')}` : ''}
          </div>
        `,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'bot',
        content: `Error: Failed to process query - ${error.message}`,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
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
      const response = await axios.post(`${BACKEND_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const data = response.data;
      const botMessage = {
        role: 'bot',
        content: `
          ${data.analysis?.emergency ? '<div class="emergency-alert">EMERGENCY: SEEK IMMEDIATE MEDICAL ATTENTION!</div>' : ''}
          <div class="urgency-${data.analysis?.urgency_level || 'low'}">
            Document processed: ${data.filename}<br>
            ${data.analysis?.response || 'No analysis available.'}<br>
            <strong>Confidence:</strong> ${(data.analysis?.confidence || 0).toFixed(2)}<br>
            <strong>Explanation:</strong> ${data.analysis?.explanation || 'N/A'}<br>
            <strong>Urgency:</strong> ${data.analysis?.urgency_level || 'N/A'}<br>
            ${data.analysis?.possible_conditions?.length ? `<strong>Possible Conditions:</strong> ${data.analysis.possible_conditions.join(', ')}` : ''}
          </div>
        `,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'bot',
        content: `Error: Failed to process document - ${error.message}`,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('username');
    setIsLoggedIn(false);
    setUsername('Guest');
    setSessionId(`session_guest_${Math.random().toString(36).substring(2)}`);
    setMessages([]);
    navigate('/login');
  };

  // Handle key press for Enter to submit
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  };

  return (
    <div className="chatbot-container container my-5">
      <div className="main-header">
        <h1>Clinical Diagnostics Chatbot</h1>
        <p>Your AI-powered medical assistant</p>
      </div>

      {isLoggedIn && (
        <div className="sidebar">
          <h2>Welcome, {username}</h2>
          <h3>Upload Medical Document</h3>
          <input
            type="file"
            accept=".pdf,.txt,.docx"
            onChange={handleFileUpload}
            disabled={isLoading}
          />
          <button className="btn btn-danger mt-3" onClick={handleLogout}>
            Logout
          </button>
        </div>
      )}

      <div className="chat-area" ref={chatContainerRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.role === 'user' ? 'user' : 'bot'}`} dangerouslySetInnerHTML={{ __html: `${msg.role === 'user' ? 'You' : 'Bot'}: ${msg.content}<br><small>${msg.timestamp}</small>` }} />
        ))}
      </div>

      <div className="chat-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter your medical query..."
          disabled={isLoading}
        />
        <button className="btn btn-primary" onClick={submitQuery} disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>

      <div className="footer">
        <p><strong>⚠️ Important Medical Disclaimer:</strong></p>
        <p>This chatbot provides informational content only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.</p>
        <p><strong>In case of emergency, call your local emergency services immediately.</strong></p>
      </div>
    </div>
  );
};

export default Chatbot;