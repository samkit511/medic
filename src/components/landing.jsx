import React from 'react';

const Landing = () => {
  return (
    <div className="flex flex-col items-center justify-center text-center  mt-5 px-4 bg-gradient-to-b from-white to-blue-50">
      <img
        src="/src/assets/medic-avatar.png"
        alt="Medic Avatar"
        className="w-32 h-32 md:w-40 md:h-40 rounded-full mb-4 shadow-md"
      />
      <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-2">
        CURA
        <br />
        <span className="text-blue-600 text- md:text-xl">Your Health Ally</span>
      </h1>
      <p className="text-gray-600 max-w-md text-sm md:text-base mb-4 font-medium leading-relaxed">
        Empower your well-being with AI-driven insights. From symptom analysis to document reviews, MediBot delivers trusted guidance for your health journey.
      </p>
      <a
        href="/login"
        className="bg-blue-700 hover:bg-blue-800 text-white font-semibold py-2 px-6 rounded-lg shadow transition"
      >
        Try CURA
      </a>
    </div>
  );
};

export default Landing;
