import React from 'react'
import CropDiseaseDetector from './Pages/CropDiseaseDetector'
import './App.css'

function App() {
  return (
    <div className="App">
      <div className="app-content">
        <header className="app-header">
          <h1 className="app-title">Minori AI</h1>
          <p className="app-subtitle">Advanced Crop Disease Detection System</p>
        </header>
        
        <main>
          <CropDiseaseDetector/>
        </main>
        
        <footer style={{ 
          textAlign: 'center', 
          marginTop: '2rem', 
          padding: '1rem',
          color: '#6b7280',
          fontSize: '0.9rem'
        }}>
          <p>&copy; 2025 Minori AI - Empowering Agriculture with Technology</p>
        </footer>
      </div>
    </div>
  )
}

export default App