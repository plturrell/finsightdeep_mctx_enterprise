import React from 'react';

/**
 * Component for displaying MCTS search results
 */
const SearchResults = ({ results }) => {
  if (!results) {
    return null;
  }
  
  const { action, action_weights, search_statistics } = results;
  
  return (
    <div className="card mt-4">
      <div className="card-header">
        <h5 className="card-title">Search Results</h5>
      </div>
      <div className="card-body">
        {/* Search statistics */}
        <div className="mb-4">
          <h6 className="card-subtitle mb-2">Performance Metrics</h6>
          <div className="row">
            <div className="col-md-4">
              <div className="card bg-light">
                <div className="card-body text-center">
                  <h5>{search_statistics.duration_ms.toFixed(2)} ms</h5>
                  <p className="text-muted mb-0">Duration</p>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card bg-light">
                <div className="card-body text-center">
                  <h5>{search_statistics.num_expanded_nodes}</h5>
                  <p className="text-muted mb-0">Expanded Nodes</p>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card bg-light">
                <div className="card-body text-center">
                  <h5>{search_statistics.max_depth_reached}</h5>
                  <p className="text-muted mb-0">Max Depth</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Selected actions */}
        <div className="mb-4">
          <h6 className="card-subtitle mb-2">Selected Actions</h6>
          <div className="table-responsive">
            <table className="table table-hover">
              <thead>
                <tr>
                  <th scope="col">Batch Index</th>
                  <th scope="col">Selected Action</th>
                </tr>
              </thead>
              <tbody>
                {action.map((actionIndex, batchIndex) => (
                  <tr key={batchIndex}>
                    <td>{batchIndex}</td>
                    <td>{actionIndex}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* Action weights */}
        <div>
          <h6 className="card-subtitle mb-2">Action Weights</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead>
                <tr>
                  <th scope="col">Batch</th>
                  {action_weights[0] && action_weights[0].map((_, actionIndex) => (
                    <th key={actionIndex} scope="col">A{actionIndex}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {action_weights.map((weights, batchIndex) => (
                  <tr key={batchIndex}>
                    <td>{batchIndex}</td>
                    {weights.map((weight, actionIndex) => (
                      <td key={actionIndex}>{weight.toFixed(3)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* Action weights visualization */}
        <div className="mt-4">
          <h6 className="card-subtitle mb-3">Action Weights Visualization</h6>
          <div className="row">
            {action_weights.slice(0, 4).map((weights, batchIndex) => (
              <div key={batchIndex} className="col-md-6 mb-4">
                <div className="card">
                  <div className="card-header">
                    Batch {batchIndex}
                  </div>
                  <div className="card-body">
                    <div style={{ height: '200px', display: 'flex', alignItems: 'flex-end' }}>
                      {weights.map((weight, actionIndex) => {
                        const isSelected = action[batchIndex] === actionIndex;
                        const barHeight = `${Math.max(1, weight * 100)}%`;
                        return (
                          <div 
                            key={actionIndex}
                            style={{
                              flex: 1,
                              height: barHeight,
                              margin: '0 2px',
                              backgroundColor: isSelected ? '#007bff' : '#6c757d',
                              position: 'relative',
                              transition: 'height 0.3s ease',
                            }}
                            title={`Action ${actionIndex}: ${weight.toFixed(4)}`}
                          >
                            <div style={{
                              position: 'absolute',
                              bottom: '-25px',
                              width: '100%',
                              textAlign: 'center',
                              fontSize: '0.8rem',
                            }}>
                              {actionIndex}
                            </div>
                            {isSelected && (
                              <div style={{
                                position: 'absolute',
                                top: '-20px',
                                width: '100%',
                                textAlign: 'center',
                                color: '#007bff',
                                fontWeight: 'bold',
                              }}>
                                ★
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchResults;