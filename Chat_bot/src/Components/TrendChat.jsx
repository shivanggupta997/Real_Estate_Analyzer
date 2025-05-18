import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function TrendChart({ chartData }) {
  if (!chartData || !chartData.labels || !chartData.datasets) {
    return <p>No chart data available.</p>;
  }

  const data = {
    labels: chartData.labels,
    datasets: chartData.datasets.map(ds => ({
      ...ds,
      fill: false, // Or true if you want filled area charts
    })),
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false, 
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: chartData.type === 'price_trend' ? 'Price Trend Analysis' : 'Demand Comparison',
      },
    },
    scales: {
        y: {
            beginAtZero: false 
        }
    }
  };

  return <Line data={data} options={options} />;
}

export default TrendChart;