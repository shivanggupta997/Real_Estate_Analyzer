import React from 'react';
import TrendChart from './TrendChat';
import DataTable from './DataTable';

function ChatMessage({ message }) {
  const { sender, text, analysis } = message;

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        marginBottom: '20px'
      }}
    >
      <div
        className={`message ${sender}`}
        style={{
          backgroundColor: sender === 'user' ? '#e0f7fa' : '#ffffff',
          borderRadius: '12px',
          padding: '20px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
          maxWidth: '95%',            
          width: '1000px',            
          fontFamily: 'sans-serif',
          overflowX: 'auto'           
        }}
      >
        {text && (
          <div
            className="text-content"
            style={{ marginBottom: '10px', fontWeight: '500' }}
          >
            {text}
          </div>
        )}

        {analysis && (
          <div className="analysis-content">
            {analysis.summary && (
              <div
                className="summary"
                style={{ marginBottom: '15px', fontSize: '16px' }}
              >
                {analysis.summary}
              </div>
            )}

            {analysis.chart_data && (
              <div
                className="chart-container-inner"
                style={{
                  position: 'relative',
                  height: '400px',         
                  marginBottom: '15px',
                  width: '100%'
                }}
              >
                <TrendChart chartData={analysis.chart_data} />
              </div>
            )}

            {analysis.table_data && analysis.table_data.length > 0 && (
              <div
                className="table-container"
                style={{ overflowX: 'auto', marginTop: '10px' }}
              >
                <DataTable data={analysis.table_data} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatMessage;
