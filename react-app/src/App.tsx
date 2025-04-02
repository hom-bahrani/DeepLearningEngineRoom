import React from 'react';
import './App.css';
import DeepLearningVisualizer from './Visualisation';

// We'll use inline styles until Tailwind is properly configured
const appStyle = {
  width: '100%',
  minHeight: '100vh',
  padding: '20px',
  backgroundColor: '#f3f4f6'
};

function App() {
  return (
    <div className="App" style={appStyle}>
      <DeepLearningVisualizer />
    </div>
  );
}

export default App;
