import React, { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";
import { Camera, Loader2, Settings, User, Users, AlertCircle, Scan, Shield, Zap, Database } from "lucide-react";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detectedFaces, setDetectedFaces] = useState(0);
  const [modelStatus, setModelStatus] = useState({
    ssd: false,
    recognition: false,
    landmark: false
  });
  const [processingFPS, setProcessingFPS] = useState(0);
  const [lastProcessingTime, setLastProcessingTime] = useState(0);

  useEffect(() => {
    const loadModels = async () => {
      try {
        setIsLoading(true);
        await faceapi.nets.ssdMobilenetv1.loadFromUri("/models");
        setModelStatus(prev => ({ ...prev, ssd: true }));
        
        await faceapi.nets.faceRecognitionNet.loadFromUri("/models");
        setModelStatus(prev => ({ ...prev, recognition: true }));
        
        await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
        setModelStatus(prev => ({ ...prev, landmark: true }));
        
        console.log("Models loaded successfully");
      } catch (error) {
        console.error("Error loading models:", error);
        setError("Failed to load face detection models");
      } finally {
        setIsLoading(false);
      }
    };

    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing webcam:", error);
        setError("Failed to access webcam. Please make sure you have granted camera permissions.");
      }
    };

    const getLabeledFaceDescriptions = async () => {
      const labels = ["Virat", "Messi", "Prakash"];
      return Promise.all(
        labels.map(async (label) => {
          const descriptions = [];
          for (let i = 1; i <= 2; i++) {
            try {
              const img = await faceapi.fetchImage(`/labels/${label}/${i}.png`);
              const detections = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor();
              if (detections) {
                descriptions.push(detections.descriptor);
              }
            } catch (error) {
              console.error(`Error loading image for ${label}:`, error);
            }
          }
          return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
      );
    };

    const setupFaceDetection = async () => {
      try {
        const labeledFaceDescriptors = await getLabeledFaceDescriptions();
        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

        if (!videoRef.current || !canvasRef.current) return;

        const displaySize = {
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight
        };
        faceapi.matchDimensions(canvasRef.current, displaySize);

        const processFrame = async () => {
          if (!videoRef.current || !canvasRef.current) return;
          
          const startTime = performance.now();
          
          const detections = await faceapi
            .detectAllFaces(videoRef.current)
            .withFaceLandmarks()
            .withFaceDescriptors();

          setDetectedFaces(detections.length);

          const resizedDetections = faceapi.resizeResults(detections, displaySize);
          const context = canvasRef.current.getContext("2d");
          if (!context) return;

          context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

          const results = resizedDetections.map((d) =>
            faceMatcher.findBestMatch(d.descriptor)
          );
          
          results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box;
            new faceapi.draw.DrawBox(box, {
              label: result.toString(),
              boxColor: "#00ff00",
              drawLabelOptions: {
                fontColor: "#fff",
                backgroundColor: "#00ff00cc"
              }
            }).draw(canvasRef.current);
          });

          const endTime = performance.now();
          const processingTime = endTime - startTime;
          setLastProcessingTime(processingTime);
          setProcessingFPS(Math.round(1000 / processingTime));

          requestAnimationFrame(processFrame);
        };

        processFrame();
      } catch (error) {
        console.error("Error in face detection setup:", error);
        setError("Failed to initialize face detection");
      }
    };

    loadModels()
      .then(startWebcam)
      .then(() => {
        if (videoRef.current) {
          videoRef.current.onplay = setupFaceDetection;
        }
      });
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 bg-pattern relative overflow-hidden">
      {/* Circuit animation lines */}
      <div className="circuit-line" />
      <div className="circuit-line" />
      <div className="circuit-line" />
      <div className="circuit-line" />

      {/* Ambient background lights */}
      <div className="ambient-light bg-green-500 w-[500px] h-[500px] -top-[250px] -right-[250px]" />
      <div className="ambient-light bg-blue-500 w-[500px] h-[500px] -bottom-[250px] -left-[250px]" />

      {/* Scanning effect */}
      <div className="scan-effect" />

      {/* Header */}
      <div className="relative z-10 text-center pt-8 mb-8">
        <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-2">
          <Camera className="w-8 h-8" />
          AI Face Recognition System
        </h1>
        <p className="text-gray-400">Advanced real-time face detection and recognition system</p>
      </div>

      {/* Main Content */}
      <div className="relative z-10 w-full max-w-6xl mx-auto flex flex-col lg:flex-row gap-8 items-start justify-center px-4">
        {/* Video Container */}
        <div className="flex-1">
          <div className="relative rounded-lg overflow-hidden shadow-2xl bg-black glow">
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                <div className="text-center">
                  <Loader2 className="w-8 h-8 animate-spin text-white mx-auto mb-2" />
                  <p className="text-white">Loading AI models...</p>
                </div>
              </div>
            )}
            {error && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                <div className="text-center p-4">
                  <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
                  <p className="text-white">{error}</p>
                </div>
              </div>
            )}
            <video
              ref={videoRef}
              className="min-w-[600px] h-[450px] object-cover"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
            />
          </div>
        </div>

        {/* Status Panel */}
        <div className="lg:w-80 w-full space-y-4">
          {/* System Status */}
          <div className="bg-gray-800/80 rounded-lg p-6 shadow-xl glow">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              System Status
            </h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">SSD Model</span>
                <span className={`px-2 py-1 rounded text-xs ${modelStatus.ssd ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                  {modelStatus.ssd ? 'LOADED' : 'LOADING'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Recognition Model</span>
                <span className={`px-2 py-1 rounded text-xs ${modelStatus.recognition ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                  {modelStatus.recognition ? 'LOADED' : 'LOADING'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Landmark Model</span>
                <span className={`px-2 py-1 rounded text-xs ${modelStatus.landmark ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                  {modelStatus.landmark ? 'LOADED' : 'LOADING'}
                </span>
              </div>
            </div>
          </div>

          {/* Performance Stats */}
          <div className="bg-gray-800/80 rounded-lg p-6 shadow-xl glow">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Performance
            </h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Processing Time</span>
                <span className="text-white">{lastProcessingTime.toFixed(1)} ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">FPS</span>
                <span className="text-white">{processingFPS}</span>
              </div>
            </div>
          </div>

          {/* Detection Stats */}
          <div className="bg-gray-800/80 rounded-lg p-6 shadow-xl glow">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Users className="w-5 h-5" />
              Detection Stats
            </h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Detected Faces</span>
                <span className="text-2xl font-bold text-white">{detectedFaces}</span>
              </div>
              <div className="mt-4">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min((detectedFaces / 5) * 100, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="bg-gray-800/50 rounded-lg p-6 glow">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <User className="w-5 h-5" />
              Instructions
            </h2>
            <ul className="text-gray-400 text-sm space-y-2">
              <li className="flex items-center gap-2">
                <Shield className="w-4 h-4" />
                Ensure good lighting conditions
              </li>
              <li className="flex items-center gap-2">
                <Scan className="w-4 h-4" />
                Face the camera directly
              </li>
              <li className="flex items-center gap-2">
                <Users className="w-4 h-4" />
                Keep a distance of 0.5-1m from camera
              </li>
              <li className="flex items-center gap-2">
                <Database className="w-4 h-4" />
                Avoid rapid movements
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10 mt-8 text-center pb-4">
        <p className="text-gray-500 text-sm">
          Powered by face-api.js and TensorFlow.js
        </p>
      </div>
    </div>
  );
}

export default App;