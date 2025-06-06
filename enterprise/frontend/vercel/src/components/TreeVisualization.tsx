import React, { useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Box, CircularProgress } from '@mui/material';
import { nodeStateColors, colorScales } from '@/lib/constants';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface TreeVisualizationProps {
  data: any;
  layoutType: string;
  nodeSizeBy: string;
  height?: string;
  width?: string;
}

const TreeVisualization: React.FC<TreeVisualizationProps> = ({
  data,
  layoutType = 'radial',
  nodeSizeBy = 'visits',
  height = '70vh',
  width = '100%'
}) => {
  const plotRef = useRef<any>(null);
  
  // Calculate positions based on layout type
  const calculatePositions = () => {
    if (!data) return [];
    
    const positions: [number, number][] = Array(data.node_count).fill([0, 0]);
    
    if (layoutType === 'radial') {
      // Set root at center
      positions[0] = [0, 0];
      
      // Group nodes by depth
      const nodesByDepth: {[key: number]: number[]} = {0: [0]};
      const maxDepth = 0;
      
      // Calculate depths
      for (let i = 1; i < data.node_count; i++) {
        if (i in data.parents) {
          const parent = data.parents[i];
          let depth = 1;
          let current = parent;
          
          while (current !== 0) {
            current = data.parents[current];
            depth += 1;
          }
          
          if (!(depth in nodesByDepth)) {
            nodesByDepth[depth] = [];
          }
          nodesByDepth[depth].push(i);
        }
      }
      
      // Place nodes at each depth
      const radiusFactor = 1.618; // Golden ratio
      
      for (const depth in nodesByDepth) {
        if (depth === '0') continue;
        
        const nodes = nodesByDepth[depth];
        const radius = parseInt(depth) * radiusFactor;
        
        // Place nodes in a circle
        nodes.forEach((node, i) => {
          const angle = 2 * Math.PI * i / nodes.length;
          positions[node] = [
            radius * Math.cos(angle),
            radius * Math.sin(angle)
          ];
        });
      }
    } else {
      // Hierarchical layout
      // Set root at top
      positions[0] = [0, 0];
      
      // Group nodes by depth
      const nodesByDepth: {[key: number]: number[]} = {0: [0]};
      const maxDepth = 0;
      
      // Calculate depths
      for (let i = 1; i < data.node_count; i++) {
        if (i in data.parents) {
          const parent = data.parents[i];
          let depth = 1;
          let current = parent;
          
          while (current !== 0) {
            current = data.parents[current];
            depth += 1;
          }
          
          if (!(depth in nodesByDepth)) {
            nodesByDepth[depth] = [];
          }
          nodesByDepth[depth].push(i);
        }
      }
      
      // Place nodes at each depth
      const verticalSpacing = 2.0;
      
      for (const depth in nodesByDepth) {
        if (depth === '0') continue;
        
        const nodes = nodesByDepth[depth];
        const y = -parseInt(depth) * verticalSpacing;
        const width = nodes.length - 1;
        
        // Group nodes by parent
        const nodesByParent: {[key: number]: number[]} = {};
        nodes.forEach(node => {
          const parent = data.parents[node];
          if (!(parent in nodesByParent)) {
            nodesByParent[parent] = [];
          }
          nodesByParent[parent].push(node);
        });
        
        // Place nodes for each parent
        for (const parent in nodesByParent) {
          const parentNodes = nodesByParent[parent];
          const parentX = positions[parseInt(parent)][0];
          const width = parentNodes.length - 1 || 1;
          
          parentNodes.forEach((node, i) => {
            // Calculate offset from parent
            let offset = 0;
            if (parentNodes.length > 1) {
              offset = -width/2 + i * width/(parentNodes.length-1);
            }
            
            const x = parentX + offset;
            positions[node] = [x, y];
          });
        }
      }
    }
    
    return positions;
  };
  
  // Create traces for tree visualization
  const createTraces = () => {
    if (!data) return [];
    
    // Calculate node positions
    const positions = calculatePositions();
    
    // Extract data for nodes
    const nodeX: number[] = [];
    const nodeY: number[] = [];
    const nodeVisits: number[] = [];
    const nodeValues: number[] = [];
    const nodeColors: string[] = [];
    const hoverTexts: string[] = [];
    
    // Generate node data
    for (let i = 0; i < data.node_count; i++) {
      if (positions[i]) {
        nodeX.push(positions[i][0]);
        nodeY.push(positions[i][1]);
        nodeVisits.push(data.visits[i]);
        nodeValues.push(data.values[i]);
        nodeColors.push(nodeStateColors[data.states[i]] || nodeStateColors.explored);
        
        // Create hover text
        hoverTexts.push(`
          <b>Node ${i}</b><br>
          Visits: ${data.visits[i]}<br>
          Value: ${data.values[i].toFixed(3)}<br>
          State: ${data.states[i]}
        `);
      }
    }
    
    // Create edge data
    const edgeX: (number | null)[] = [];
    const edgeY: (number | null)[] = [];
    
    // Generate edge data
    for (let i = 1; i < data.node_count; i++) {
      if (i in data.parents) {
        const parent = data.parents[i];
        // Add edge from parent to child
        edgeX.push(positions[parent][0]);
        edgeX.push(positions[i][0]);
        edgeX.push(null); // Add null to create separation between edges
        
        edgeY.push(positions[parent][1]);
        edgeY.push(positions[i][1]);
        edgeY.push(null);
      }
    }
    
    // Calculate sizes based on nodeSizeBy
    const sizeArray = nodeSizeBy === 'visits' ? data.visits : data.values;
    
    // Normalize sizes for visual balance
    const minSize = 8;
    const maxSize = 25;
    const minVal = Math.min(...sizeArray.filter((v: number) => v > 0));
    const maxVal = Math.max(...sizeArray);
    
    // Calculate node sizes using logarithmic scale for better visualization
    const nodeSizes = sizeArray.map((val: number) => {
      if (maxVal === minVal) return minSize + (maxSize - minSize) / 2;
      // Use log scale for better distribution of sizes
      const normalizedSize = Math.log1p(val) / Math.log1p(maxVal);
      return minSize + normalizedSize * (maxSize - minSize);
    });
    
    // Create traces
    const traces = [
      // Edge trace
      {
        x: edgeX,
        y: edgeY,
        mode: 'lines',
        line: {
          width: 1,
          color: '#CBD5E1',
          shape: 'spline'
        },
        hoverinfo: 'none',
        type: 'scatter'
      },
      // Node trace
      {
        x: nodeX,
        y: nodeY,
        mode: 'markers',
        marker: {
          size: nodeSizes,
          color: nodeColors,
          line: {
            width: 1,
            color: '#FFFFFF'
          },
          opacity: 0.9,
        },
        text: hoverTexts,
        hoverinfo: 'text',
        type: 'scatter'
      }
    ];
    
    return traces;
  };
  
  // Create layout configuration
  const createLayout = () => {
    return {
      showlegend: false,
      hovermode: 'closest',
      margin: {
        t: 10,
        l: 10,
        r: 10,
        b: 10,
      },
      xaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        fixedrange: true,
      },
      yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        fixedrange: true,
      },
      paper_bgcolor: '#FFFFFF',
      plot_bgcolor: '#FFFFFF',
      dragmode: 'pan',
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
      scrollZoom: true,
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
      ref={plotRef}
      data={traces}
      layout={{
        ...layout,
        height: '100%',
        width: '100%',
      }}
      config={config}
      style={{ height, width }}
      useResizeHandler={true}
      className="visualization-plot"
    />
  );
};

export default TreeVisualization;