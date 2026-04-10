"use client";
import { useState, useEffect, useRef } from 'react';
import DeviceForm from '../components/DeviceForm';
import DatasetSelector from '../components/DatasetSelector';
import MetricsDashboard from '../components/MetricsDashboard';
import { Settings, Zap, Terminal } from 'lucide-react';

export default function Home() {
  const [specs, setSpecs] = useState({
    ram_gb: 16,
    cpu_cores: 8,
    gpu_type: "Nvidia RTX 3000/4000 series",
    vram_gb: 8
  });
  
  const [dataset, setDataset] = useState("mnist");
  const [recommendation, setRecommendation] = useState(null);
  
  const [isTraining, setIsTraining] = useState(false);
  const [epochData, setEpochData] = useState([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  
  const ws = useRef(null);

  // Fetch recommendation when specs or dataset change
  useEffect(() => {
    const fetchRec = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const res = await fetch(`${apiUrl}/recommend`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...specs, dataset_name: dataset })
        });
        const data = await res.json();
        setRecommendation(data);
      } catch (e) {
        console.error("Error fetching recommendation", e);
      }
    };
    const timer = setTimeout(fetchRec, 500); // debounce
    return () => clearTimeout(timer);
  }, [specs, dataset]);

  const startTraining = () => {
    if (!recommendation || isTraining) return;
    
    setIsTraining(true);
    setEpochData([]);
    setCurrentEpoch(0);
    setConfusionMatrix(null);
    
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const wsUrl = apiUrl.replace(/^http/, 'ws') + '/train_ws';
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onopen = () => {
      ws.current.send(JSON.stringify({
        dataset_name: dataset,
        learning_rate: recommendation.learning_rate,
        epochs: recommendation.epochs,
        batch_size: recommendation.batch_size,
        optimizer_name: recommendation.optimizer_name
      }));
    };
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'epoch_end') {
        setCurrentEpoch(data.epoch);
        setEpochData(prev => [...prev, data]);
      } else if (data.type === 'training_complete') {
        if (data.confusion_matrix) {
          setConfusionMatrix(data.confusion_matrix);
        }
        setIsTraining(false);
        ws.current.close();
      } else if (data.type === 'error') {
        setIsTraining(false);
        ws.current.close();
      }
    };
    
    ws.current.onerror = () => {
      setIsTraining(false);
    };
  };

  return (
    <div className="container">
      <header style={{ marginBottom: '3rem', textAlign: 'center' }}>
        <h1 className="title-glow" style={{ fontSize: '3rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
          <Zap size={40} color="var(--accent)" /> AI Hyperparameter Optimizer
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem', maxWidth: '600px', margin: '0 auto' }}>
          Intelligent tuning based on real hardware constraints and dataset specifics.
        </p>
      </header>

      <div className="dashboard-grid">
        {/* LEFT COLUMN: Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
          <DeviceForm specs={specs} setSpecs={setSpecs} />
          <DatasetSelector dataset={dataset} setDataset={setDataset} />
          
          <div className="glass-panel" style={{ marginBottom: '1.5rem', background: 'rgba(111, 87, 255, 0.05)', borderColor: 'var(--accent)' }}>
            <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent)' }}>
              <Settings size={20} /> Optimization Rules
            </h3>
            
            {recommendation ? (
              <>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: 1.5 }}>
                  {recommendation.rationale}
                </p>
                <div className="rec-grid">
                  <div className="rec-item">
                    <div className="rec-label">Batch Size</div>
                    <div className="rec-value">{recommendation.batch_size}</div>
                  </div>
                  <div className="rec-item">
                    <div className="rec-label">Epochs</div>
                    <div className="rec-value">{recommendation.epochs}</div>
                  </div>
                  <div className="rec-item">
                    <div className="rec-label">Learning Rate</div>
                    <div className="rec-value">{recommendation.learning_rate}</div>
                  </div>
                  <div className="rec-item">
                    <div className="rec-label">Optimizer</div>
                    <div className="rec-value" style={{ textTransform: 'uppercase' }}>{recommendation.optimizer_name}</div>
                  </div>
                </div>
              </>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>Calculating constraints...</p>
            )}
          </div>

          <button 
            className="primary-btn" 
            onClick={startTraining}
            disabled={isTraining || !recommendation}
          >
            {isTraining ? <><Terminal className="spinner" size={20}/> Executing Training...</> : <><Terminal size={20}/> Initialize Training Loop</>}
          </button>
        </div>

        {/* RIGHT COLUMN: Metrics */}
        <div>
          <MetricsDashboard 
            isTraining={isTraining} 
            epochData={epochData} 
            currentEpoch={currentEpoch}
            totalEpochs={recommendation?.epochs || 0}
            recs={recommendation}
            confusionMatrix={confusionMatrix}
          />
        </div>
      </div>
    </div>
  );
}
