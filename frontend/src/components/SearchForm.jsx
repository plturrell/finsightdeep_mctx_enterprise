import React, { useState } from 'react';
import mctsApi from '../services/api';

/**
 * Form component for configuring and running MCTS searches
 */
const SearchForm = ({ onSearchComplete }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Form state
  const [formData, setFormData] = useState({
    // Root input
    batch_size: 1,
    num_actions: 4,
    prior_logits: '[[0, 0, 0, 0]]',
    value: '[0]',
    embedding: '[0]',
    
    // Search parameters
    num_simulations: 32,
    max_depth: 50,
    max_num_considered_actions: 16,
    dirichlet_fraction: 0.25,
    dirichlet_alpha: 0.3,
    
    // Search type
    search_type: 'gumbel_muzero',
  });
  
  // Handle form input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };
  
  // Handle numeric input changes
  const handleNumericChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: parseFloat(value) || 0,
    }));
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      // Parse JSON arrays
      let priorLogits, value, embedding;
      
      try {
        priorLogits = JSON.parse(formData.prior_logits);
        value = JSON.parse(formData.value);
        embedding = JSON.parse(formData.embedding);
      } catch (parseError) {
        setError(`Invalid JSON format: ${parseError.message}`);
        setLoading(false);
        return;
      }
      
      // Prepare API request
      const rootInput = {
        prior_logits: priorLogits,
        value: value,
        embedding: embedding,
        batch_size: parseInt(formData.batch_size, 10),
        num_actions: parseInt(formData.num_actions, 10),
      };
      
      const searchParams = {
        num_simulations: parseInt(formData.num_simulations, 10),
        max_depth: parseInt(formData.max_depth, 10),
        max_num_considered_actions: parseInt(formData.max_num_considered_actions, 10),
        dirichlet_fraction: parseFloat(formData.dirichlet_fraction),
        dirichlet_alpha: parseFloat(formData.dirichlet_alpha),
      };
      
      // Run search
      const result = await mctsApi.runSearch(
        rootInput,
        searchParams,
        formData.search_type
      );
      
      // Pass results to parent component
      if (onSearchComplete) {
        onSearchComplete(result);
      }
    } catch (error) {
      setError(error.message || 'An error occurred');
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="card">
      <div className="card-header">
        <h5 className="card-title">MCTS Configuration</h5>
      </div>
      <div className="card-body">
        {error && (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label">Search Algorithm</label>
            <select
              name="search_type"
              value={formData.search_type}
              onChange={handleChange}
              className="form-select"
            >
              <option value="muzero">MuZero</option>
              <option value="gumbel_muzero">Gumbel MuZero</option>
              <option value="stochastic_muzero">Stochastic MuZero</option>
            </select>
          </div>
          
          <div className="row">
            <div className="col-md-6">
              <div className="mb-3">
                <label className="form-label">Batch Size</label>
                <input
                  type="number"
                  className="form-control"
                  name="batch_size"
                  value={formData.batch_size}
                  onChange={handleNumericChange}
                  min="1"
                  max="128"
                />
              </div>
            </div>
            
            <div className="col-md-6">
              <div className="mb-3">
                <label className="form-label">Number of Actions</label>
                <input
                  type="number"
                  className="form-control"
                  name="num_actions"
                  value={formData.num_actions}
                  onChange={handleNumericChange}
                  min="1"
                />
              </div>
            </div>
          </div>
          
          <div className="mb-3">
            <label className="form-label">Prior Logits (JSON array of arrays)</label>
            <textarea
              className="form-control"
              name="prior_logits"
              value={formData.prior_logits}
              onChange={handleChange}
              rows="3"
            />
            <div className="form-text">
              Example: [[0, 0, 0, 0]] for 1 batch item with 4 actions
            </div>
          </div>
          
          <div className="mb-3">
            <label className="form-label">Value (JSON array)</label>
            <input
              type="text"
              className="form-control"
              name="value"
              value={formData.value}
              onChange={handleChange}
            />
            <div className="form-text">
              Example: [0] for 1 batch item
            </div>
          </div>
          
          <div className="mb-3">
            <label className="form-label">Embedding (JSON array)</label>
            <input
              type="text"
              className="form-control"
              name="embedding"
              value={formData.embedding}
              onChange={handleChange}
            />
            <div className="form-text">
              Example: [0] for 1 batch item
            </div>
          </div>
          
          <div className="row">
            <div className="col-md-6">
              <div className="mb-3">
                <label className="form-label">Number of Simulations</label>
                <input
                  type="number"
                  className="form-control"
                  name="num_simulations"
                  value={formData.num_simulations}
                  onChange={handleNumericChange}
                  min="1"
                  max="1000"
                />
              </div>
            </div>
            
            <div className="col-md-6">
              <div className="mb-3">
                <label className="form-label">Maximum Depth</label>
                <input
                  type="number"
                  className="form-control"
                  name="max_depth"
                  value={formData.max_depth}
                  onChange={handleNumericChange}
                  min="1"
                />
              </div>
            </div>
          </div>
          
          <div className="row">
            <div className="col-md-4">
              <div className="mb-3">
                <label className="form-label">Max Considered Actions</label>
                <input
                  type="number"
                  className="form-control"
                  name="max_num_considered_actions"
                  value={formData.max_num_considered_actions}
                  onChange={handleNumericChange}
                  min="1"
                />
              </div>
            </div>
            
            <div className="col-md-4">
              <div className="mb-3">
                <label className="form-label">Dirichlet Fraction</label>
                <input
                  type="number"
                  className="form-control"
                  name="dirichlet_fraction"
                  value={formData.dirichlet_fraction}
                  onChange={handleNumericChange}
                  step="0.01"
                  min="0"
                  max="1"
                />
              </div>
            </div>
            
            <div className="col-md-4">
              <div className="mb-3">
                <label className="form-label">Dirichlet Alpha</label>
                <input
                  type="number"
                  className="form-control"
                  name="dirichlet_alpha"
                  value={formData.dirichlet_alpha}
                  onChange={handleNumericChange}
                  step="0.01"
                  min="0"
                />
              </div>
            </div>
          </div>
          
          <div className="d-grid gap-2">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
            >
              {loading ? 'Running Search...' : 'Run Search'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SearchForm;