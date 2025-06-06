import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

/**
 * TreeGraph component for rendering MCTS trees with React
 * This component integrates with the Plotly-based visualization system
 */
const TreeGraph = (props) => {
  const { data, layout, config, style } = props;
  const graphRef = useRef(null);
  
  useEffect(() => {
    // Initialize the graph when data is available
    if (data && graphRef.current) {
      window.Plotly.react(graphRef.current, data, layout, config);
    }
    
    // Clean up on unmount
    return () => {
      if (graphRef.current) {
        window.Plotly.purge(graphRef.current);
      }
    };
  }, [data, layout, config]);
  
  return (
    <div
      ref={graphRef}
      style={style}
      className="tree-graph"
    />
  );
};

TreeGraph.propTypes = {
  data: PropTypes.array,
  layout: PropTypes.object,
  config: PropTypes.object,
  style: PropTypes.object
};

TreeGraph.defaultProps = {
  data: [],
  layout: {},
  config: {
    displayModeBar: true,
    scrollZoom: true,
    responsive: true
  },
  style: {
    width: '100%',
    height: '70vh'
  }
};

export default TreeGraph;