/**
 * API service for interacting with the MCTX backend
 * 
 * This service provides methods to call the MCTX API endpoints
 * and handles error responses.
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1';

/**
 * Error class for API request failures
 */
export class APIError extends Error {
  constructor(message, statusCode, details) {
    super(message);
    this.name = 'APIError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * Handle API response errors
 * 
 * @param {Response} response - Fetch response object
 * @throws {APIError} Throws an APIError with details on failure
 */
const handleResponse = async (response) => {
  if (!response.ok) {
    const contentType = response.headers.get('content-type');
    
    if (contentType && contentType.includes('application/json')) {
      const errorData = await response.json();
      throw new APIError(
        errorData.message || 'An error occurred',
        response.status,
        errorData.details
      );
    } else {
      const text = await response.text();
      throw new APIError(
        text || `HTTP Error ${response.status}`,
        response.status
      );
    }
  }
  
  return response.json();
};

/**
 * MCTS API Service
 */
const mctsApi = {
  /**
   * Run a Monte Carlo Tree Search
   * 
   * @param {Object} rootInput - Root state input
   * @param {Object} searchParams - Search parameters
   * @param {string} searchType - Type of search algorithm
   * @returns {Promise<Object>} Search results
   * @throws {APIError} On request failure
   */
  runSearch: async (rootInput, searchParams, searchType = 'gumbel_muzero') => {
    try {
      const response = await fetch(`${API_BASE_URL}/mcts/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          root_input: rootInput,
          search_params: searchParams,
          search_type: searchType,
        }),
      });
      
      return handleResponse(response);
    } catch (error) {
      console.error('MCTS search error:', error);
      if (error instanceof APIError) {
        throw error;
      } else {
        throw new APIError(
          'Failed to connect to the MCTS service',
          0,
          { error: error.message }
        );
      }
    }
  },
  
  /**
   * Check API health status
   * 
   * @returns {Promise<Object>} Health status information
   * @throws {APIError} On request failure
   */
  checkHealth: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/mcts/health`);
      return handleResponse(response);
    } catch (error) {
      console.error('Health check error:', error);
      if (error instanceof APIError) {
        throw error;
      } else {
        throw new APIError(
          'Failed to connect to the MCTS service',
          0,
          { error: error.message }
        );
      }
    }
  },
};

export default mctsApi;