import axios from 'axios';

// API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Run an MCTS search
 * @param params Search parameters
 * @returns Search result
 */
export const runSearch = async (params: any) => {
  try {
    const response = await api.post('/api/search', {
      batch_size: params.batch_size,
      num_actions: params.num_actions,
      num_simulations: params.num_simulations,
      search_type: params.search_type,
      use_t4_optimizations: params.use_t4_optimizations,
      precision: params.precision,
      distributed: params.distributed,
      num_devices: params.num_devices,
      tensor_core_aligned: params.tensor_core_aligned || true,
    });
    
    return response.data;
  } catch (error) {
    console.error('Error running search:', error);
    throw error;
  }
};

/**
 * Get a previous search result
 * @param searchId Search ID
 * @returns Search result
 */
export const getSearchResult = async (searchId: string) => {
  try {
    const response = await api.get(`/api/search/${searchId}`);
    return response.data;
  } catch (error) {
    console.error('Error getting search result:', error);
    throw error;
  }
};

/**
 * Get visualization data for a search
 * @param searchId Search ID
 * @returns Visualization data
 */
export const getVisualizationData = async (searchId: string) => {
  try {
    const response = await api.get(`/api/visualization/${searchId}`);
    return response.data;
  } catch (error) {
    console.error('Error getting visualization data:', error);
    throw error;
  }
};

/**
 * Get daily search statistics
 * @returns Daily statistics
 */
export const getDailyStats = async () => {
  try {
    const response = await api.get('/api/search/stats/daily');
    return response.data;
  } catch (error) {
    console.error('Error getting daily stats:', error);
    throw error;
  }
};