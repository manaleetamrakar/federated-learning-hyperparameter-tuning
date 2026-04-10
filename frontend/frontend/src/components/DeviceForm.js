"use client";
import { Cpu, MemoryStick, MonitorPlay } from 'lucide-react';

export default function DeviceForm({ specs, setSpecs }) {
  const handleChange = (e) => {
    const { name, value } = e.target;
    setSpecs(prev => ({
      ...prev,
      [name]: ["ram_gb", "cpu_cores", "vram_gb"].includes(name) ? Number(value) : value
    }));
  };

  return (
    <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
      <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Cpu size={20} className="title-glow" /> System Specifications
      </h3>
      
      <div className="input-group">
        <label className="input-label">System RAM (GB)</label>
        <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
          <MemoryStick size={18} style={{ position: 'absolute', left: '12px', color: 'var(--text-secondary)' }} />
          <input 
            type="number" 
            name="ram_gb" 
            value={specs.ram_gb} 
            onChange={handleChange}
            className="styled-input"
            style={{ paddingLeft: '2.5rem' }}
            min="2" max="256"
          />
        </div>
      </div>

      <div className="input-group">
        <label className="input-label">CPU Cores</label>
        <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
          <Cpu size={18} style={{ position: 'absolute', left: '12px', color: 'var(--text-secondary)' }} />
          <input 
            type="number" 
            name="cpu_cores" 
            value={specs.cpu_cores} 
            onChange={handleChange}
            className="styled-input"
            style={{ paddingLeft: '2.5rem' }}
            min="1" max="128"
          />
        </div>
      </div>

      <div className="input-group">
        <label className="input-label">GPU Type</label>
        <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
          <MonitorPlay size={18} style={{ position: 'absolute', left: '12px', color: 'var(--text-secondary)' }} />
          <select 
            name="gpu_type" 
            value={specs.gpu_type} 
            onChange={handleChange}
            className="styled-input"
            style={{ paddingLeft: '2.5rem' }}
          >
            <option value="None">None (CPU Only)</option>
            <option value="Nvidia RTX 3000/4000 series">Nvidia RTX Series</option>
            <option value="Nvidia GTX series">Nvidia GTX Series</option>
            <option value="AMD Radeon RX series">AMD Radeon RX</option>
            <option value="Apple Silicon (M1/M2/M3)">Apple M-Series</option>
          </select>
        </div>
      </div>

      {specs.gpu_type !== "None" && (
        <div className="input-group">
          <label className="input-label">GPU VRAM (GB)</label>
          <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
            <MemoryStick size={18} style={{ position: 'absolute', left: '12px', color: 'var(--text-secondary)' }} />
            <input 
              type="number" 
              name="vram_gb" 
              value={specs.vram_gb} 
              onChange={handleChange}
              className="styled-input"
              style={{ paddingLeft: '2.5rem' }}
              min="1" max="80"
            />
          </div>
        </div>
      )}
    </div>
  );
}
