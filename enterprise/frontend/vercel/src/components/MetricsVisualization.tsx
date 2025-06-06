import React from 'react';
import dynamic from 'next/dynamic';
import { Box, CircularProgress } from '@mui/material';
import { colorScales } from '@/lib/constants';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface MetricsVisualizationProps {
  data: any;
  height?: string;
  width?: string;
}

const MetricsVisualization: React.FC<MetricsVisualizationProps> = ({
  data,
  height = '70vh',
  width = '100%'
}) => {
  // Calculate metrics for visualization
  const calculateMetrics = () => {
    if (!data) return null;
    
    // Calculate depth for each node
    const depths: {[key: number]: number} = {0: 0}; // Root node has depth 0
    
    for (let i = 1; i < data.node_count; i++) {
      if (i in data.parents) {
        const parent = data.parents[i];
        depths[i] = depths[parent] + 1;
      }
    }
    
    // Group metrics by depth
    const metricsByDepth: {[key: number]: {
      count: number,
      total_visits: number,
      total_value: number
    }} = {};
    
    for (let i = 0; i < data.node_count; i++) {
      const depth = depths[i] || 0;
      
      if (!(depth in metricsByDepth)) {
        metricsByDepth[depth] = {
          count: 0,
          total_visits: 0,
          total_value: 0
        };
      }
      
      metricsByDepth[depth].count += 1;
      metricsByDepth[depth].total_visits += data.visits[i];
      metricsByDepth[depth].total_value += data.values[i];
    }
    
    // Calculate averages
    const depthMetrics = Object.entries(metricsByDepth).map(([depth, metrics]) => ({
      depth: parseInt(depth),
      count: metrics.count,
      avg_visits: metrics.total_visits / metrics.count,
      avg_value: metrics.total_value / metrics.count
    }));
    
    // Calculate exploration vs. exploitation data
    const c = 1.41; // Exploration constant
    const maxVisits = Math.max(...data.visits);
    
    const explorationData = data.visits.map((visits: number, i: number) => {
      if (visits === 0) return null;
      
      const exploitation = data.values[i];
      const exploration = c * Math.sqrt(Math.log(maxVisits) / visits);
      
      return {
        node: i,
        exploitation,
        exploration,
        ucb: exploitation + exploration,
        visits
      };
    }).filter(Boolean);
    
    return {
      depthMetrics,
      explorationData
    };
  };
  
  // Create traces for the metrics visualization
  const createTraces = () => {
    if (!data) return [];
    
    const metrics = calculateMetrics();
    if (!metrics) return [];
    
    const { depthMetrics, explorationData } = metrics;
    
    // Calculate bin counts for histograms
    const visitBins = 20;
    const valueBins = 20;
    
    return [
      // Visit distribution histogram
      {
        x: data.visits,
        type: 'histogram',
        name: 'Visit Distribution',
        marker: {
          color: colorScales.visits[5],
          line: {
            color: 'white',
            width: 0.5
          }
        },
        opacity: 0.75,
        xbins: {
          size: (Math.max(...data.visits) - Math.min(...data.visits)) / visitBins
        },
        hovertemplate: 'Visits: %{x}<br>Count: %{y}<extra></extra>'
      },
      
      // Value distribution histogram
      {
        x: data.values,
        type: 'histogram',
        name: 'Value Distribution',
        marker: {
          color: colorScales.values[5],
          line: {
            color: 'white',
            width: 0.5
          }
        },
        opacity: 0.75,
        xbins: {
          size: (Math.max(...data.values) - Math.min(...data.values)) / valueBins
        },
        hovertemplate: 'Value: %{x}<br>Count: %{y}<extra></extra>'
      },
      
      // Visits by depth
      {
        x: depthMetrics.map(m => m.depth),
        y: depthMetrics.map(m => m.avg_visits),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Avg Visits by Depth',
        marker: {
          color: colorScales.visits[6],
          size: 8
        },
        line: {
          color: colorScales.visits[6],
          width: 2
        },
        hovertemplate: 'Depth: %{x}<br>Avg Visits: %{y:.2f}<extra></extra>'
      },
      
      // Exploration vs. exploitation
      {
        x: explorationData.map(d => d.exploitation),
        y: explorationData.map(d => d.exploration),
        mode: 'markers',
        type: 'scatter',
        name: 'Exploration vs. Exploitation',
        marker: {
          color: colorScales.values[4],
          size: explorationData.map(d => 5 + 15 * Math.sqrt(d.visits / Math.max(...data.visits))),
          opacity: 0.7
        },
        text: explorationData.map(d => `Node: ${d.node}<br>Exploitation: ${d.exploitation.toFixed(3)}<br>Exploration: ${d.exploration.toFixed(3)}<br>Visits: ${d.visits}`),
        hoverinfo: 'text'
      }
    ];
  };
  
  // Create layout for the metrics visualization
  const createLayout = () => {
    return {
      grid: {
        rows: 2,
        columns: 2,
        pattern: 'independent'
      },
      showlegend: false,
      margin: {
        l: 50,
        r: 20,
        t: 60,
        b: 50
      },
      paper_bgcolor: '#FFFFFF',
      plot_bgcolor: '#FFFFFF',
      
      // Visit distribution
      'xaxis': {
        title: {
          text: 'Visit Count',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0, 0.48],
        row: 1,
        col: 1,
        showgrid: true,
        gridcolor: '#F1F5F9',
        zeroline: true,
        zerolinecolor: '#CBD5E1'
      },
      'yaxis': {
        title: {
          text: 'Count',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0.52, 1],
        row: 1,
        col: 1,
        showgrid: true,
        gridcolor: '#F1F5F9'
      },
      
      // Value distribution
      'xaxis2': {
        title: {
          text: 'Node Value',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0.52, 1],
        row: 1,
        col: 2,
        showgrid: true,
        gridcolor: '#F1F5F9',
        zeroline: true,
        zerolinecolor: '#CBD5E1'
      },
      'yaxis2': {
        title: {
          text: 'Count',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0.52, 1],
        row: 1,
        col: 2,
        showgrid: true,
        gridcolor: '#F1F5F9'
      },
      
      // Visits by depth
      'xaxis3': {
        title: {
          text: 'Tree Depth',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0, 0.48],
        row: 2,
        col: 1,
        showgrid: true,
        gridcolor: '#F1F5F9',
        zeroline: true,
        zerolinecolor: '#CBD5E1'
      },
      'yaxis3': {
        title: {
          text: 'Mean Visits',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0, 0.48],
        row: 2,
        col: 1,
        showgrid: true,
        gridcolor: '#F1F5F9'
      },
      
      // Exploration vs. exploitation
      'xaxis4': {
        title: {
          text: 'Exploitation (Value)',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0.52, 1],
        row: 2,
        col: 2,
        showgrid: true,
        gridcolor: '#F1F5F9',
        zeroline: true,
        zerolinecolor: '#CBD5E1'
      },
      'yaxis4': {
        title: {
          text: 'Exploration (UCB)',
          font: {
            size: 12,
            color: '#334155'
          }
        },
        domain: [0, 0.48],
        row: 2,
        col: 2,
        showgrid: true,
        gridcolor: '#F1F5F9'
      },
      
      annotations: [
        {
          text: 'Visit Distribution',
          xref: 'paper',
          yref: 'paper',
          x: 0.24,
          y: 1,
          xanchor: 'center',
          yanchor: 'bottom',
          showarrow: false,
          font: {
            size: 14,
            color: '#334155',
            weight: 600
          }
        },
        {
          text: 'Value Distribution',
          xref: 'paper',
          yref: 'paper',
          x: 0.76,
          y: 1,
          xanchor: 'center',
          yanchor: 'bottom',
          showarrow: false,
          font: {
            size: 14,
            color: '#334155',
            weight: 600
          }
        },
        {
          text: 'Visit Count Over Depth',
          xref: 'paper',
          yref: 'paper',
          x: 0.24,
          y: 0.48,
          xanchor: 'center',
          yanchor: 'bottom',
          showarrow: false,
          font: {
            size: 14,
            color: '#334155',
            weight: 600
          }
        },
        {
          text: 'Exploration vs. Exploitation',
          xref: 'paper',
          yref: 'paper',
          x: 0.76,
          y: 0.48,
          xanchor: 'center',
          yanchor: 'bottom',
          showarrow: false,
          font: {
            size: 14,
            color: '#334155',
            weight: 600
          }
        }
      ],
      
      font: {
        family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        size: 12,
        color: '#334155'
      }
    };
  };
  
  // Create config for Plotly
  const createConfig = () => {
    return {
      displayModeBar: true,
      modeBarButtonsToRemove: [
        'autoScale2d',
        'lasso2d',
        'select2d',
        'toggleSpikelines',
      ],
      displaylogo: false,
      responsive: true,
    };
  };
  
  // If no data, show loading or placeholder
  if (!data) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height,
          width,
        }}
      >
        <CircularProgress />
      </Box>
    );
  }
  
  // Create traces and layout
  const traces = createTraces();
  const layout = createLayout();
  const config = createConfig();
  
  return (
    <Plot
      data={traces}
      layout={{
        ...layout,
        height: '100%',
        width: '100%',
      }}
      config={config}
      style={{ height, width }}
      useResizeHandler={true}
      className="metrics-plot"
    />
  );
};

export default MetricsVisualization;