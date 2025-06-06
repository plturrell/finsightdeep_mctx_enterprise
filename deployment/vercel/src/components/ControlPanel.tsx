import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Button,
  Typography,
  Divider,
  Box,
  Slider
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

interface ControlPanelProps {
  searchParams: any;
  onParamChange: (name: string, value: any) => void;
  onRunSearch: () => void;
  loading: boolean;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ 
  searchParams, 
  onParamChange, 
  onRunSearch,
  loading
}) => {
  const handleTextChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    onParamChange(name, value);
  };

  const handleSelectChange = (event: any) => {
    const { name, value } = event.target;
    onParamChange(name, value);
  };

  const handleSwitchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    onParamChange(name, checked);
  };

  const handleSliderChange = (name: string) => (event: any, newValue: number | number[]) => {
    onParamChange(name, newValue);
  };

  return (
    <>
      <Card 
        sx={{ 
          mb: 3,
          borderRadius: 2,
          overflow: 'hidden'
        }}
      >
        <CardHeader 
          title="Search Configuration" 
          sx={{ 
            bgcolor: '#f1f5f9',
            borderBottom: '1px solid #e2e8f0',
            '& .MuiCardHeader-title': {
              fontSize: '1.125rem',
              fontWeight: 600
            }
          }}
        />
        <CardContent>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="search-type-label">Search Algorithm</InputLabel>
            <Select
              labelId="search-type-label"
              id="search-type"
              name="search_type"
              value={searchParams.search_type}
              label="Search Algorithm"
              onChange={handleSelectChange}
            >
              <MenuItem value="muzero">MuZero</MenuItem>
              <MenuItem value="gumbel_muzero">Gumbel MuZero</MenuItem>
              <MenuItem value="stochastic_muzero">Stochastic MuZero</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            fullWidth
            id="batch-size"
            name="batch_size"
            label="Batch Size"
            type="number"
            value={searchParams.batch_size}
            onChange={handleTextChange}
            InputProps={{ inputProps: { min: 1, max: 256 } }}
            sx={{ mb: 2 }}
          />
          
          <TextField
            fullWidth
            id="num-actions"
            name="num_actions"
            label="Number of Actions"
            type="number"
            value={searchParams.num_actions}
            onChange={handleTextChange}
            InputProps={{ inputProps: { min: 2, max: 1024 } }}
            sx={{ mb: 2 }}
          />
          
          <Box sx={{ mb: 2 }}>
            <Typography id="num-simulations-slider" gutterBottom>
              Number of Simulations: {searchParams.num_simulations}
            </Typography>
            <Slider
              value={searchParams.num_simulations}
              onChange={handleSliderChange('num_simulations')}
              aria-labelledby="num-simulations-slider"
              valueLabelDisplay="auto"
              min={10}
              max={1000}
              step={10}
              marks={[
                { value: 10, label: '10' },
                { value: 500, label: '500' },
                { value: 1000, label: '1000' }
              ]}
            />
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
            Optimizations
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={searchParams.use_t4_optimizations}
                onChange={handleSwitchChange}
                name="use_t4_optimizations"
                color="primary"
              />
            }
            label="Use T4 GPU Optimizations"
            sx={{ mb: 1, display: 'block' }}
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={searchParams.distributed}
                onChange={handleSwitchChange}
                name="distributed"
                color="primary"
              />
            }
            label="Use Distributed Computation"
            sx={{ mb: 1, display: 'block' }}
          />
          
          {searchParams.distributed && (
            <TextField
              fullWidth
              id="num-devices"
              name="num_devices"
              label="Number of Devices"
              type="number"
              value={searchParams.num_devices}
              onChange={handleTextChange}
              InputProps={{ inputProps: { min: 1, max: 8 } }}
              sx={{ mb: 2, mt: 1 }}
            />
          )}
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="precision-label">Computation Precision</InputLabel>
            <Select
              labelId="precision-label"
              id="precision"
              name="precision"
              value={searchParams.precision}
              label="Computation Precision"
              onChange={handleSelectChange}
            >
              <MenuItem value="fp16">FP16 (half precision)</MenuItem>
              <MenuItem value="fp32">FP32 (full precision)</MenuItem>
            </Select>
          </FormControl>
          
          <Button
            fullWidth
            variant="contained"
            color="primary"
            onClick={onRunSearch}
            disabled={loading}
            startIcon={<PlayArrowIcon />}
            sx={{ mt: 2 }}
          >
            {loading ? 'Running...' : 'Run Search'}
          </Button>
        </CardContent>
      </Card>
      
      <Card 
        sx={{ 
          borderRadius: 2,
          overflow: 'hidden'
        }}
      >
        <CardHeader 
          title="Visualization Options" 
          sx={{ 
            bgcolor: '#f1f5f9',
            borderBottom: '1px solid #e2e8f0',
            '& .MuiCardHeader-title': {
              fontSize: '1.125rem',
              fontWeight: 600
            }
          }}
        />
        <CardContent>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="layout-type-label">Layout Type</InputLabel>
            <Select
              labelId="layout-type-label"
              id="layout-type"
              name="layout_type"
              value={searchParams.layout_type || 'radial'}
              label="Layout Type"
              onChange={handleSelectChange}
            >
              <MenuItem value="radial">Radial</MenuItem>
              <MenuItem value="hierarchical">Hierarchical</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="node-size-label">Node Size By</InputLabel>
            <Select
              labelId="node-size-label"
              id="node-size"
              name="node_size_by"
              value={searchParams.node_size_by || 'visits'}
              label="Node Size By"
              onChange={handleSelectChange}
            >
              <MenuItem value="visits">Visit Count</MenuItem>
              <MenuItem value="values">Node Value</MenuItem>
            </Select>
          </FormControl>
        </CardContent>
      </Card>
    </>
  );
};

export default ControlPanel;