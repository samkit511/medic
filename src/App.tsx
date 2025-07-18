import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, NavLink } from 'react-router-dom';
import Landing from './components/landing';
import Auth from './components/auth';
import ForgotPassword from './components/forgot_password';
import Feedback from './components/feedback';
import Chatbot from './components/Chatbot';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  const [count, setCount] = useState(0);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check authentication status on mount
  React.useEffect(() => {
    const checkAuth = () => {
      const username = localStorage.getItem('username') || localStorage.getItem('user_name');
      setIsAuthenticated(!!username);
    };
    
    checkAuth();
    
    // Listen for storage changes (login/logout)
    const handleStorageChange = () => {
      checkAuth();
    };
    
    window.addEventListener('storage', handleStorageChange);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('username');
    localStorage.removeItem('user_name');
    localStorage.removeItem('user_email');
    localStorage.removeItem('user_id');
    setIsAuthenticated(false);
    window.location.href = '/login';
  };

  return (
    <Router>
      {/* Add Tailwind CSS CDN */}
      <link
        href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
        rel="stylesheet"
      />
      <nav className="bg-blue-600 shadow-lg">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <a className="flex items-center" href="/">
              <img
                src="/src/assets/logo.png"
                alt="Medic Logo"
                className="h-10 w-10 mr-2"
              />
              <span className="text-white text-xl font-semibold">CURA</span>
            </a>
            <div className="space-x-6">
              <NavLink
                className={({ isActive }) =>
                  isActive
                    ? "text-white font-medium border-b-2 border-white pb-1"
                    : "text-gray-200 hover:text-white transition"
                }
                to="/about-the-bot"
              >
                About the Bot
              </NavLink>
              <NavLink
                className={({ isActive }) =>
                  isActive
                    ? "text-white font-medium border-b-2 border-white pb-1"
                    : "text-gray-200 hover:text-white transition"
                }
                to="/recognition"
              >
                Recognition
              </NavLink>
              <NavLink
                className={({ isActive }) =>
                  isActive
                    ? "text-white font-medium border-b-2 border-white pb-1"
                    : "text-gray-200 hover:text-white transition"
                }
                to="/feedback"
              >
                Feedback
              </NavLink>
              {isAuthenticated ? (
                <>
                  <NavLink
                    className={({ isActive }) =>
                      isActive
                        ? "text-white font-medium border-b-2 border-white pb-1"
                        : "text-gray-200 hover:text-white transition"
                    }
                    to="/chatbot"
                  >
                    Chatbot
                  </NavLink>
                  <button
                    onClick={handleLogout}
                    className="text-gray-200 hover:text-white transition"
                  >
                    Logout
                  </button>
                </>
              ) : (
                <NavLink
                  className={({ isActive }) =>
                    isActive
                      ? "text-white font-medium border-b-2 border-white pb-1"
                      : "text-gray-200 hover:text-white transition"
                  }
                  to="/login"
                >
                  Login
                </NavLink>
              )}
            </div>
          </div>
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route
          path="/about-the-bot"
          element={
            <>
              <div className="bg-gradient-to-r from-blue-50 to-white py-12">
                <div className="w-full px-4 md:px-8 lg:px-12">
                  <div className="flex flex-col items-center mb-8">
                    <img
                      alt="Medic Avatar"
                      className="rounded-full mb-4"
                      src="/src/assets/medic-avatar.png"
                      style={{ maxWidth: "200px" }}
                    />
                    <a  href="/login" class="text-white bg-blue-700 text-xl hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-semibold rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800" >Try CURA</a>
                  </div>
                  <h2 className="text-3xl font-bold text-gray-800 mb-6">
                    About Our AI Medical Assistant Chatbot
                  </h2>
                  <p className="text-gray-600 leading-relaxed mb-8">
                    Welcome to our AI Medical Assistant Chatbot, a cutting-edge tool designed to provide accessible, reliable, and empathetic healthcare support. Powered by advanced large language models (LLMs) like BioMistral-7B and DeepSeek-Med, our chatbot leverages artificial intelligence (AI), machine learning (ML), and natural language processing (NLP) to offer preliminary symptom analysis and general health guidance. Our mission is to bridge healthcare gaps, especially for rural and underserved communities, by delivering 24/7 medical assistance with a focus on inclusivity and user trust.
                  </p>
                  <h3 className="text-2xl font-semibold text-blue-600 mb-4">
                    Key Features
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {[
                      {
                        title: "Accurate Symptom Analysis",
                        description:
                          "Using few-shot prompting and chain-of-thought techniques, our chatbot achieves high diagnostic accuracy, drawing from benchmarks like PubMedQA and MedQA.",
                      },
                      {
                        title: "Multilingual Support",
                        description:
                          "Designed to serve diverse populations, the chatbot supports regional languages and dialects, ensuring accessibility for non-English speakers.",
                      },
                      {
                        title: "Emotional Intelligence",
                        description:
                          "Integrated affective computing enables the chatbot to interpret emotional tones and stress levels, delivering empathetic and human-like responses.",
                      },
                      {
                        title: "Privacy-First Design",
                        description:
                          "Built with federated learning, our chatbot trains on-device to protect user data, complying with regulations like HIPAA and GDPR.",
                      },
                      {
                        title: "Seamless Integration",
                        description:
                          "Connects with electronic health records (EHRs) and real-time data from wearables for personalized health insights and continuity of care.",
                      },
                      {
                        title: "Document Upload",
                        description:
                          "Users can upload medical documents for context-aware, accurate responses tailored to their health history.",
                      },
                    ].map((feature, index) => (
                      <div
                        key={index}
                        className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300"
                      >
                        <h4 className="text-lg font-semibold text-gray-800 mb-2">
                          {feature.title}
                        </h4>
                        <p className="text-gray-600">{feature.description}</p>
                      </div>
                    ))}
                  </div>
                  <p className="text-gray-600 leading-relaxed mt-8">
                    Whether you're seeking quick health advice, exploring symptoms, or monitoring long-term conditions like diabetes or hypertension, our chatbot is here to empower you with knowledge and support, without replacing the expertise of human doctors.
                  </p>
                </div>
              </div>
            </>
          }
        />
        <Route
          path="/recognition"
          element={
            <div className="bg-gradient-to-r from-blue-50 to-white py-12">
              <div className="w-full px-4 md:px-8 lg:px-12">
                <h2 className="text-3xl font-bold text-gray-800 mb-6">
                  Recognition
                </h2>
                <p className="text-gray-600 leading-relaxed mb-8">
                  Our AI Medical Assistant Chatbot has been recognized for its innovative approach to transforming healthcare delivery. Below are some of the milestones and accolades that highlight our commitment to excellence:
                </p>
                <div className="relative">
                  <div className="absolute left-0 top-0 h-full w-1 bg-blue-500"></div>
                  <div className="space-y-12 ml-6">
                    {[
                      {
                        title: "Innovation in Healthcare Award 2024",
                        description:
                          "Received for pioneering AI-driven healthcare solutions that improve access for underserved communities.",
                      },
                      {
                        title: "MedTech Breakthrough Award",
                        description:
                          "Recognized for outstanding contributions to medical technology with our privacy-first, multilingual chatbot.",
                      },
                      {
                        title: "Global Health Equity Challenge Finalist",
                        description:
                          "Selected as a finalist for addressing healthcare disparities through AI-powered symptom analysis and support.",
                      },
                      {
                        title: "AI Excellence Certification",
                        description:
                          "Certified for achieving high diagnostic accuracy and empathetic responses in medical AI applications.",
                      },
                    ].map((milestone, index) => (
                      <div
                        key={index}
                        className="relative flex items-center"
                      >
                        <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-4 h-4 bg-blue-500 rounded-full"></div>
                        <div className="ml-8 w-full bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
                          <h4 className="text-lg font-semibold text-gray-800">
                            {milestone.title}
                          </h4>
                          <p className="text-gray-600">{milestone.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          }
        />
        <Route
          path="/login"
          element={
            <div className="bg-gradient-to-r from-blue-50 to-white py-12">
              <div className="w-full px-4 md:px-8 lg:px-12">
                <div className="flex flex-col md:flex-row gap-6">
                  <div className="w-full md:w-1/4 bg-blue-100 p-6 rounded-lg shadow-md">
                    <h3 className="text-xl font-semibold text-blue-800 mb-4">
                      Welcome to CURA
                    </h3>
                    <p className="text-gray-600">
                      Your trusted AI-powered health companion, available 24/7 to support your well-being.
                    </p>
                  </div>
                  <div className="w-full md:w-3/4">
                    <div className="bg-white p-8 rounded-lg shadow-md">
                      <h2 className="text-2xl font-bold text-gray-800 mb-6">
                        Login to Your Account
                      </h2>
                      <Auth />
                      {/* <div className="mt-4 text-sm text-blue-600">
                        <a href="/forgot-password" className="hover:underline">
                          Forgot Password?
                        </a>
                      </div> */}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          }
        />
        <Route
          path="/forgot-password"
          element={<ForgotPassword />}
        />
        <Route
          path="/feedback"
          element={
            <div className="bg-gradient-to-r from-blue-50 to-white py-12">
              <div className="w-full px-4 md:px-8 lg:px-12">
                <div className="flex flex-col md:flex-row gap-6">
                  <div className="w-full h-screen md:w-1/4 bg-blue-100 p-6 rounded-lg shadow-md">
                    <h3 className="text-xl font-semibold text-blue-800 mb-4">
                      Share Your Thoughts
                    </h3>
                    <p className="text-gray-600">
                      Help us improve CURA with your valuable feedback.
                    </p>
                  </div>
                  <div className="w-full md:w-3/4">
                    <div className="bg-white p-8 rounded-lg shadow-md">
                      <h2 className="text-2xl font-bold text-gray-800 mb-6">
                        Provide Your Feedback
                      </h2>
                      <Feedback />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          }
        />
        <Route path="/chatbot" element={
          <ProtectedRoute>
            <Chatbot />
          </ProtectedRoute>
        } />
      </Routes>
    </Router>
  );
}

export default App;
