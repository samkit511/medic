// src/components/auth.jsx
import React from 'react';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { useNavigate } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';

const Auth = () => {
  const clientId = '761585986838-odetbtp33g2gtrp82ikrvmbt51r5f250.apps.googleusercontent.com';
  const navigate = useNavigate();

  const handleSuccess = async (credentialResponse) => {
    try {
      // Send the Google ID token to backend for verification and JWT creation
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          id_token: credentialResponse.credential
        })
      });

      if (!response.ok) {
        throw new Error(`Login failed: ${response.status}`);
      }

      const data = await response.json();
      
      // Store JWT tokens and user info
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);
      localStorage.setItem('username', data.user.name || 'Unknown User');
      localStorage.setItem('user_email', data.user.email || '');
      localStorage.setItem('user_id', data.user.id || '');
      
      navigate('/chatbot');
    } catch (error) {
      console.error('Login Error:', error);
      alert('Failed to process login. Please try again.');
    }
  };

  const handleError = () => {
    console.error('Google OAuth Login Failed');
    alert('Login failed. Please check your credentials and try again.');
  };

  return (
    <div className="flex flex-col items-center justify-center p-8 bg-white rounded-lg shadow-md">
      <img
        src="/src/assets/medic-avatar.png"
        alt="Medic Avatar"
        className="w-24 h-24 mb-4 rounded-full shadow"
      />
      <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome to CURA</h2>
      <p className="text-gray-600 text-center mb-6 max-w-md">
        Sign in with Google to access your personalized CURA experience. Your privacy and security are our top priorities.
      </p>
      <GoogleOAuthProvider clientId={clientId}>
        <GoogleLogin
          onSuccess={handleSuccess}
          onError={handleError}
          shape="rectangular"
          theme="filled_blue"
          size="large"
          text="signin_with"
          logo_alignment="left"
        />
      </GoogleOAuthProvider>
    </div>
  );
};

export default Auth;
