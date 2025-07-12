// src/components/auth.jsx
import React from 'react';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { useNavigate } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';

const Auth = () => {
  const clientId = '';
  const navigate = useNavigate();

  const handleSuccess = (credentialResponse) => {
    try {
      const userInfo = jwtDecode(credentialResponse.credential);
      localStorage.setItem('user_name', userInfo.name || 'Unknown User');
      navigate('/chatbot');
    } catch (error) {
      console.error('JWT Decode Error:', error);
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
