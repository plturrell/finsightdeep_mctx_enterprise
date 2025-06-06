/**
 * Default search parameters
 */
export const defaultSearchParams = {
  batch_size: 4,
  num_actions: 16,
  num_simulations: 50,
  search_type: 'gumbel_muzero',
  use_t4_optimizations: true,
  precision: 'fp16',
  distributed: false,
  num_devices: 1,
  tensor_core_aligned: true,
  layout_type: 'radial',
  node_size_by: 'visits',
};

/**
 * Node state colors
 */
export const nodeStateColors = {
  unexplored: '#CBD5E1', // Slate 300
  explored: '#6366F1',   // Indigo 500
  selected: '#F59E0B',   // Amber 500
  pruned: '#CBD5E1',     // Slate 300
  optimal: '#10B981',    // Emerald 500
};

/**
 * Sequential color scales
 */
export const colorScales = {
  visits: [
    '#EEF2FF', // Indigo 50
    '#E0E7FF', // Indigo 100
    '#C7D2FE', // Indigo 200
    '#A5B4FC', // Indigo 300
    '#818CF8', // Indigo 400
    '#6366F1', // Indigo 500
    '#4F46E5', // Indigo 600
    '#4338CA', // Indigo 700
    '#3730A3', // Indigo 800
    '#312E81', // Indigo 900
  ],
  values: [
    '#ECFDF5', // Emerald 50
    '#D1FAE5', // Emerald 100
    '#A7F3D0', // Emerald 200
    '#6EE7B7', // Emerald 300
    '#34D399', // Emerald 400
    '#10B981', // Emerald 500
    '#059669', // Emerald 600
    '#047857', // Emerald 700
    '#065F46', // Emerald 800
    '#064E3B', // Emerald 900
  ],
};