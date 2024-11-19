import React, { useState, useEffect } from 'react';
import { ResponsiveLine } from '@nivo/line';
import styled from 'styled-components';
import axios from 'axios';

const DashboardContainer = styled.div`
  padding: 2rem;
  background: #1a1a1a;
  color: #ffffff;
`;

const ChartContainer = styled.div`
  height: 400px;
  background: #242424;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 2rem;
`;

const SignalsTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  background: #242424;
  border-radius: 8px;
  
  th, td {
    padding: 1rem;
    text-align: left;
    border-bottom: 1px solid #333;
  }
  
  th {
    background: #333;
  }
`;

const Dashboard = () => {
  const [portfolioData, setPortfolioData] = useState([]);
  const [signals, setSignals] = useState([]);

  useEffect(() => {
    // Fetch portfolio data
    const fetchData = async () => {
      const response = await axios.get('/api/portfolio');
      const data = response.data;
      
      // Transform data for chart
      const chartData = [{
        id: 'Portfolio Value',
        data: data.orders.map(order => ({
          x: order.date,
          y: data.available_funds + data.reserved
        }))
      }];
      
      setPortfolioData(chartData);
      setSignals(data.orders.slice(-10)); // Get last 10 signals
    };

    fetchData();
  }, []);

  return (
    <DashboardContainer>
      <h1>Money Printer Dashboard</h1>
      
      <ChartContainer>
        <ResponsiveLine
          data={portfolioData}
          margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
          xScale={{ type: 'time', format: '%Y-%m-%d' }}
          yScale={{ type: 'linear' }}
          axisBottom={{
            format: '%b %d',
            tickRotation: -45
          }}
          pointSize={10}
          pointColor={{ theme: 'background' }}
          pointBorderWidth={2}
          pointBorderColor={{ from: 'serieColor' }}
          enablePointLabel={true}
          theme={{
            textColor: '#ffffff',
            grid: {
              line: {
                stroke: '#333'
              }
            }
          }}
        />
      </ChartContainer>

      <h2>Recent Signals</h2>
      <SignalsTable>
        <thead>
          <tr>
            <th>Date</th>
            <th>Symbol</th>
            <th>Type</th>
            <th>Price</th>
            <th>Quantity</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((signal, index) => (
            <tr key={index}>
              <td>{signal.date}</td>
              <td>{signal.symbol}</td>
              <td style={{color: signal.type === 'buy' ? '#4caf50' : '#f44336'}}>
                {signal.type.toUpperCase()}
              </td>
              <td>Rs. {signal.price}</td>
              <td>{signal.quantity}</td>
            </tr>
          ))}
        </tbody>
      </SignalsTable>
    </DashboardContainer>
  );
};

export default Dashboard;
