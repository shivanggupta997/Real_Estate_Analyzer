import React from 'react';

function DataTable({ data }) {
  if (!data || data.length === 0) {
    return <p>No table data available.</p>;
  }

  const headers = Object.keys(data[0]);

  return (
    <table>
      <thead>
        <tr>
          {headers.map(header => <th key={header}>{header.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</th>)}
        </tr>
      </thead>
      <tbody>
        {data.map((row, index) => (
          <tr key={index}>
            {headers.map(header => <td key={header}>{row[header]}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default DataTable;