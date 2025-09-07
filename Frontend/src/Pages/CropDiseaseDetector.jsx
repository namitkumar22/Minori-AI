import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Upload, Loader2, AlertCircle, CheckCircle, Play, Square } from 'lucide-react';

const CropDiseaseDetector = () => {
  const [selectedCrop, setSelectedCrop] = useState('');
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isRealTimeMode, setIsRealTimeMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [stream, setStream] = useState(null);
  const [frameCount, setFrameCount] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  // Styles object
  const styles = {
    container: {
      maxWidth: '1024px',
      margin: '0 auto',
      padding: '24px',
      backgroundColor: 'white',
      borderRadius: '8px',
      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    },
    title: {
      fontSize: '32px',
      fontWeight: 'bold',
      textAlign: 'center',
      marginBottom: '32px',
      color: '#166534',
    },
    label: {
      display: 'block',
      fontSize: '18px',
      fontWeight: '500',
      color: '#374151',
      marginBottom: '8px',
    },
    select: {
      width: '100%',
      padding: '12px',
      border: '1px solid #d1d5db',
      borderRadius: '8px',
      fontSize: '16px',
      marginBottom: '24px',
    },
    buttonContainer: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '16px',
      marginBottom: '16px',
    },
    button: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '8px 16px',
      borderRadius: '8px',
      fontWeight: '500',
      border: 'none',
      cursor: 'pointer',
      fontSize: '14px',
    },
    buttonPrimary: {
      backgroundColor: '#3b82f6',
      color: 'white',
    },
    buttonDanger: {
      backgroundColor: '#ef4444',
      color: 'white',
    },
    buttonSuccess: {
      backgroundColor: '#10b981',
      color: 'white',
    },
    buttonWarning: {
      backgroundColor: '#f97316',
      color: 'white',
    },
    buttonSecondary: {
      backgroundColor: '#8b5cf6',
      color: 'white',
    },
    buttonIndigo: {
      backgroundColor: '#6366f1',
      color: 'white',
    },
    buttonDisabled: {
      backgroundColor: '#d1d5db',
      color: '#6b7280',
      cursor: 'not-allowed',
    },
    videoContainer: {
      position: 'relative',
      backgroundColor: '#f3f4f6',
      borderRadius: '8px',
      overflow: 'hidden',
      aspectRatio: '4/3',
      marginBottom: '24px',
    },
    video: {
      width: '100%',
      height: '100%',
      objectFit: 'cover',
    },
    placeholderContainer: {
      width: '100%',
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#6b7280',
      textAlign: 'center',
    },
    overlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },
    overlayText: {
      color: 'white',
      textAlign: 'center',
      fontSize: '18px',
    },
    liveIndicator: {
      position: 'absolute',
      top: '16px',
      right: '16px',
      backgroundColor: '#10b981',
      color: 'white',
      padding: '4px 12px',
      borderRadius: '20px',
      fontSize: '14px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    pulse: {
      width: '8px',
      height: '8px',
      backgroundColor: 'white',
      borderRadius: '50%',
      animation: 'pulse 2s infinite',
    },
    statusContainer: {
      marginBottom: '16px',
      padding: '12px',
      backgroundColor: '#fef3c7',
      border: '1px solid #fbbf24',
      borderRadius: '8px',
    },
    statusFlex: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    statusLeft: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    statusDot: {
      width: '12px',
      height: '12px',
      backgroundColor: '#10b981',
      borderRadius: '50%',
      animation: 'pulse 2s infinite',
    },
    errorContainer: {
      marginBottom: '24px',
      padding: '16px',
      backgroundColor: '#fef2f2',
      border: '1px solid #fecaca',
      borderRadius: '8px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      color: '#b91c1c',
    },
    resultContainer: {
      marginBottom: '24px',
      padding: '24px',
      backgroundColor: '#f0fdf4',
      border: '1px solid #bbf7d0',
      borderRadius: '8px',
    },
    resultHeader: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      marginBottom: '16px',
      color: '#166534',
    },
    resultTitle: {
      fontSize: '20px',
      fontWeight: '600',
    },
    gridContainer: {
      display: 'grid',
      gap: '16px',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    },
    infoSection: {
      marginTop: '16px',
    },
    infoTitle: {
      fontWeight: '500',
      color: '#374151',
      marginBottom: '8px',
    },
    solutionBox: {
      backgroundColor: 'white',
      padding: '16px',
      borderRadius: '8px',
      border: '1px solid #e5e7eb',
      maxHeight: '256px',
      overflowY: 'auto',
    },
    instructionsContainer: {
      backgroundColor: '#eff6ff',
      padding: '16px',
      borderRadius: '8px',
      border: '1px solid #bfdbfe',
    },
    instructionsTitle: {
      fontWeight: '500',
      color: '#1e40af',
      marginBottom: '8px',
    },
    instructionsList: {
      color: '#1d4ed8',
      fontSize: '14px',
      listStyle: 'decimal',
      paddingLeft: '20px',
    },
    instructionItem: {
      marginBottom: '4px',
    },
    highlightBox: {
      marginTop: '12px',
      padding: '12px',
      backgroundColor: '#dbeafe',
      borderRadius: '6px',
      border: '1px solid #93c5fd',
    },
    highlightText: {
      color: '#1e40af',
      fontSize: '14px',
      fontWeight: '500',
    },
    hiddenInput: {
      display: 'none',
    },
    hiddenCanvas: {
      display: 'none',
    },
  };

  const initWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const clientId = Math.random().toString(36).substring(7);
    const wsUrl = `ws://localhost:8000/ws/real-time-detection/${clientId}`;
    
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
    };
    
    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'result') {
        if (message.data.success) {
          setResult(message.data);
          setError('');
        } else if (message.data.error) {
          setError(message.data.error);
        }
        setIsProcessing(false);
      }
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error. Please refresh the page.');
    };
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480,
          frameRate: 30
        }
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsCameraOn(true);
      setError('');
      
      initWebSocket();
    } catch (err) {
      setError('Failed to access camera. Please check permissions.');
      console.error('Camera error:', err);
    }
  }, [initWebSocket]);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsCameraOn(false);
    setIsRealTimeMode(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [stream]);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    return canvas.toDataURL('image/jpeg', 0.8);
  }, []);

  const sendFrameForProcessing = useCallback(async (frameData) => {
    if (!selectedCrop || !frameData) return;

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        frame: frameData,
        crop: selectedCrop
      }));
      setIsProcessing(true);
    } else {
      try {
        const response = await fetch('http://localhost:8000/process-frame', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            frame: frameData,
            crop: selectedCrop
          }),
        });

        const data = await response.json();
        if (data.success) {
          setResult(data);
          setError('');
        } else if (data.error) {
          setError(data.error);
        }
      } catch (err) {
        setError('Failed to process frame. Please try again.');
        console.error('Processing error:', err);
      }
      setIsProcessing(false);
    }
  }, [selectedCrop]);

  const startRealTimeDetection = useCallback(() => {
    if (!selectedCrop || !isCameraOn) {
      setError('Please select a crop and ensure camera is active');
      return;
    }

    setIsRealTimeMode(true);
    setFrameCount(0);
    
    intervalRef.current = setInterval(() => {
      const frameData = captureFrame();
      if (frameData) {
        sendFrameForProcessing(frameData);
        setFrameCount(prev => prev + 1);
      }
    }, 2000);
  }, [selectedCrop, isCameraOn, captureFrame, sendFrameForProcessing]);

  const stopRealTimeDetection = useCallback(() => {
    setIsRealTimeMode(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsProcessing(false);
  }, []);

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !selectedCrop) {
      setError('Please select a crop and ensure camera is active');
      return;
    }

    const frameData = captureFrame();
    if (frameData) {
      sendFrameForProcessing(frameData);
    }
  }, [selectedCrop, captureFrame, sendFrameForProcessing]);

  const handleFileUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (file && selectedCrop) {
      const reader = new FileReader();
      reader.onload = (e) => {
        sendFrameForProcessing(e.target.result);
      };
      reader.readAsDataURL(file);
    } else if (!selectedCrop) {
      setError('Please select a crop first');
    }
  }, [selectedCrop, sendFrameForProcessing]);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [stream]);

  const getButtonStyle = (baseStyle, conditionStyle, isDisabled) => {
    if (isDisabled) {
      return { ...styles.button, ...styles.buttonDisabled };
    }
    return { ...styles.button, ...baseStyle, ...conditionStyle };
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>
        Real-Time Crop Disease Detection System
      </h1>

      <div>
        <label style={styles.label}>
          Select Crop Type:
        </label>
        <select
          value={selectedCrop}
          onChange={(e) => {
            setSelectedCrop(e.target.value);
            stopRealTimeDetection();
            setResult(null);
          }}
          style={styles.select}
        >
          <option value="">Choose a crop...</option>
          <option value="rice">Rice</option>
          <option value="wheat">Wheat</option>
        </select>
      </div>

      <div>
        <div style={styles.buttonContainer}>
          <button
            onClick={isCameraOn ? stopCamera : startCamera}
            style={getButtonStyle(isCameraOn ? styles.buttonDanger : styles.buttonPrimary)}
          >
            <Camera size={20} />
            {isCameraOn ? 'Stop Camera' : 'Start Camera'}
          </button>

          <button
            onClick={isRealTimeMode ? stopRealTimeDetection : startRealTimeDetection}
            disabled={!isCameraOn || !selectedCrop}
            style={getButtonStyle(
              isRealTimeMode ? styles.buttonWarning : styles.buttonSuccess,
              {},
              !isCameraOn || !selectedCrop
            )}
          >
            {isRealTimeMode ? <Square size={20} /> : <Play size={20} />}
            {isRealTimeMode ? 'Stop Real-Time' : 'Start Real-Time Detection'}
          </button>

          <button
            onClick={captureImage}
            disabled={!isCameraOn || !selectedCrop || isRealTimeMode}
            style={getButtonStyle(
              styles.buttonSecondary,
              {},
              !isCameraOn || !selectedCrop || isRealTimeMode
            )}
          >
            {isProcessing && !isRealTimeMode ? (
              <Loader2 size={20} />
            ) : (
              <Camera size={20} />
            )}
            Single Capture
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={styles.hiddenInput}
          />
          <button
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
            disabled={!selectedCrop || isRealTimeMode}
            style={getButtonStyle(
              styles.buttonIndigo,
              {},
              !selectedCrop || isRealTimeMode
            )}
          >
            <Upload size={20} />
            Upload Image
          </button>
        </div>

        {isRealTimeMode && (
          <div style={styles.statusContainer}>
            <div style={styles.statusFlex}>
              <div style={styles.statusLeft}>
                <div style={styles.statusDot}></div>
                <span style={{ color: '#92400e', fontWeight: '500' }}>Real-time detection active</span>
              </div>
              <div style={{ color: '#b45309', fontSize: '14px' }}>
                Frames processed: {frameCount}
                {isProcessing && <Loader2 size={16} style={{ display: 'inline', marginLeft: '8px' }} />}
              </div>
            </div>
          </div>
        )}

        <div style={styles.videoContainer}>
          {isCameraOn ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={styles.video}
            />
          ) : (
            <div style={styles.placeholderContainer}>
              <div>
                <Camera size={64} style={{ margin: '0 auto 16px' }} />
                <p>Camera not active</p>
                <p style={{ fontSize: '14px' }}>Click 'Start Camera' to begin</p>
              </div>
            </div>
          )}

          {isProcessing && !isRealTimeMode && (
            <div style={styles.overlay}>
              <div style={styles.overlayText}>
                <Loader2 size={48} style={{ margin: '0 auto 16px' }} />
                <p>Analyzing frame...</p>
              </div>
            </div>
          )}

          {isRealTimeMode && (
            <div style={styles.liveIndicator}>
              <div style={styles.pulse}></div>
              LIVE
            </div>
          )}
        </div>

        <canvas ref={canvasRef} style={styles.hiddenCanvas} />
      </div>

      {error && (
        <div style={styles.errorContainer}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div style={styles.resultContainer}>
          <div style={styles.resultHeader}>
            <CheckCircle size={24} />
            <h2 style={styles.resultTitle}>
              {isRealTimeMode ? 'Live Detection Results' : 'Detection Results'}
            </h2>
          </div>

          <div style={styles.gridContainer}>
            <div>
              <h3 style={styles.infoTitle}>Crop Information:</h3>
              <p style={{ color: '#6b7280' }}>Type: <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>{result.crop}</span></p>
              <p style={{ color: '#6b7280' }}>Disease: <span style={{ fontWeight: '500' }}>{result.detected_disease}</span></p>
              {result.processing_time && (
                <p style={{ color: '#6b7280', fontSize: '14px' }}>Processing time: {result.processing_time.toFixed(2)}s</p>
              )}
              {isRealTimeMode && (
                <p style={{ color: '#6b7280', fontSize: '14px' }}>Mode: Real-time detection</p>
              )}
            </div>
          </div>

          <div style={styles.infoSection}>
            <h3 style={styles.infoTitle}>Treatment Solution:</h3>
            <div style={styles.solutionBox}>
              <p style={{ color: '#374151', whiteSpace: 'pre-wrap' }}>{result.solution}</p>
            </div>
          </div>
        </div>
      )}

      <div style={styles.instructionsContainer}>
        <h3 style={styles.instructionsTitle}>How to use:</h3>
        <ol style={styles.instructionsList}>
          <li style={styles.instructionItem}>Select the crop type (Rice or Wheat)</li>
          <li style={styles.instructionItem}>Start the camera to begin video streaming</li>
          <li style={styles.instructionItem}><strong>Real-time Mode:</strong> Automatically analyzes frames every 2 seconds</li>
          <li style={styles.instructionItem}><strong>Single Capture:</strong> Analyze a single frame on demand</li>
          <li style={styles.instructionItem}><strong>Upload:</strong> Upload an existing image for analysis</li>
          <li style={styles.instructionItem}>View real-time results and treatment recommendations</li>
        </ol>
        
        <div style={styles.highlightBox}>
          <p style={styles.highlightText}>
            🎥 Real-time mode processes live video frames automatically for continuous monitoring
          </p>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default CropDiseaseDetector;