"use client";
import { useState, useRef } from 'react';
import { Database, Image as ImageIcon, Upload, Loader } from 'lucide-react';

export default function DatasetSelector({ dataset, setDataset }) {
  const [isUploading, setIsUploading] = useState(false);
  const [customDatasets, setCustomDatasets] = useState([]);
  const fileInputRef = useRef(null);

  const baseDatasets = [
    { id: "mnist", name: "MNIST Digits", desc: "Classic 10-class handwritten digits, 28x28 grayscale. Fast training.", icon: <Database size={24} /> },
    { id: "emnist", name: "EMNIST Letters", desc: "Extended MNIST focusing on digits. Larger dataset, takes more epochs.", icon: <ImageIcon size={24} /> }
  ];

  const datasets = [...baseDatasets, ...customDatasets];

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      const newDataset = {
        id: data.dataset_id,
        name: `Custom (Upload)`,
        desc: `Uploaded ZIP with ${data.num_classes} classes.`,
        icon: <Upload size={24} />
      };
      
      setCustomDatasets(prev => [...prev, newDataset]);
      setDataset(newDataset.id);
    } catch (err) {
      console.error("Upload failed", err);
      alert("Failed to upload dataset.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
      <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Database size={20} className="title-glow" /> Target Dataset
      </h3>
      
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {datasets.map((d) => (
          <div 
            key={d.id} 
            className={`dataset-card ${dataset === d.id ? 'active' : ''}`}
            onClick={() => setDataset(d.id)}
          >
            <div style={{ color: dataset === d.id ? 'var(--accent)' : 'var(--text-secondary)' }}>
              {d.icon}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: '1.05rem', color: dataset === d.id ? 'white' : 'var(--text-secondary)' }}>
                {d.name}
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                {d.desc}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '1rem', textAlign: 'center' }}>
        <input 
          type="file" 
          accept=".zip" 
          style={{ display: 'none' }} 
          ref={fileInputRef}
          onChange={handleUpload}
        />
        <button 
          className="secondary-btn" 
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          style={{ width: '100%', padding: '0.75rem', background: 'rgba(255,255,255,0.05)', border: '1px dashed var(--border-color)', color: 'var(--text-secondary)', borderRadius: '8px', cursor: isUploading ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
        >
          {isUploading ? <Loader size={18} className="spinner" /> : <Upload size={18} />}
          {isUploading ? "Uploading & Analyzing..." : "Upload Custom Dataset (.zip)"}
        </button>
      </div>
    </div>
  );
}
