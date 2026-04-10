"use client";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, CheckCircle, TrendingUp } from 'lucide-react';

export default function MetricsDashboard({ isTraining, epochData, currentEpoch, totalEpochs, recs, confusionMatrix }) {

  if (!isTraining && epochData.length === 0) {
    return (
      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '400px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <Activity size={48} style={{ opacity: 0.5, marginBottom: '1rem' }} />
        <h2>Awaiting Training Sequence</h2>
        <p>Configure your system and click "Initialize Training Loop".</p>
      </div>
    );
  }

  const finalTrainAcc = epochData.length ? epochData[epochData.length - 1].accuracy : 0;
  const finalValAcc = epochData.length ? epochData[epochData.length - 1].val_accuracy : 0;

  return (
    <div className="glass-panel" style={{ height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={20} className={isTraining ? "title-glow spinner" : "title-glow"} /> 
          {isTraining ? `Training Loop Running (Epoch ${currentEpoch}/${totalEpochs})` : 'Training Completed'}
        </h3>
        
        {!isTraining && epochData.length > 0 && (
          <div className="metrics-badge badge-green">
            <CheckCircle size={14} /> Completed
          </div>
        )}
      </div>

      {epochData.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
           <div className="rec-item" style={{ background: 'rgba(74, 222, 128, 0.05)', borderColor: 'rgba(74, 222, 128, 0.2)' }}>
              <div className="rec-label">Train Accuracy</div>
              <div className="rec-value" style={{ color: '#4ade80' }}>{(finalTrainAcc * 100).toFixed(2)}%</div>
           </div>
           <div className="rec-item" style={{ background: 'rgba(96, 165, 250, 0.05)', borderColor: 'rgba(96, 165, 250, 0.2)' }}>
              <div className="rec-label">Val Accuracy</div>
              <div className="rec-value" style={{ color: '#60a5fa' }}>{(finalValAcc * 100).toFixed(2)}%</div>
           </div>
        </div>
      )}

      {epochData.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div>
            <h4 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><TrendingUp size={16}/> Accuracy Curve</h4>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={epochData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.5)" tick={{fill: 'rgba(255,255,255,0.5)'}} />
                  <YAxis stroke="rgba(255,255,255,0.5)" tick={{fill: 'rgba(255,255,255,0.5)'}} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" name="Train Acc" stroke="#4ade80" strokeWidth={2} activeDot={{ r: 8 }} />
                  <Line type="monotone" dataKey="val_accuracy" name="Val Acc" stroke="#60a5fa" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <h4 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><TrendingUp size={16} color="#f87171"/> Loss Curve</h4>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={epochData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.5)" tick={{fill: 'rgba(255,255,255,0.5)'}} />
                  <YAxis stroke="rgba(255,255,255,0.5)" tick={{fill: 'rgba(255,255,255,0.5)'}} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="loss" name="Train Loss" stroke="#f87171" strokeWidth={2} activeDot={{ r: 8 }} />
                  <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#fb923c" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {confusionMatrix && (
            <div>
              <h4 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={16} color="#c084fc"/> Confusion Matrix (Test Data)
              </h4>
              <div className="chart-container" style={{ overflowX: 'auto', padding: '1rem' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', color: 'white', fontSize: '0.85rem' }}>
                  <tbody>
                    {confusionMatrix.map((row, i) => (
                      <tr key={i}>
                        {row.map((val, j) => {
                          const maxVal = Math.max(...confusionMatrix.flat());
                          const intensity = maxVal > 0 ? val / maxVal : 0;
                          return (
                            <td 
                              key={j} 
                              title={`Actual Class: ${i} | Predicted Class: ${j} | Count: ${val}`}
                              style={{ 
                                padding: '0.5rem', 
                                border: '1px solid rgba(255,255,255,0.05)',
                                backgroundColor: `rgba(168, 85, 247, ${intensity * 0.8})`, 
                                cursor: 'pointer'
                              }}
                            >
                              {val === 0 ? '' : val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: '300px' }}>
          {isTraining && <div className="spinner"><Activity size={48} color="var(--accent)" /></div>}
        </div>
      )}

    </div>
  );
}
