import React from 'react';
import { Tabs, Tab, Box } from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import BarChartIcon from '@mui/icons-material/BarChart';

interface VisualizationTabsProps {
  activeTab: number;
  onChange: (newValue: number) => void;
}

const VisualizationTabs: React.FC<VisualizationTabsProps> = ({ activeTab, onChange }) => {
  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    onChange(newValue);
  };

  return (
    <Box sx={{ width: '100%', bgcolor: '#f1f5f9' }}>
      <Tabs
        value={activeTab}
        onChange={handleChange}
        textColor="primary"
        indicatorColor="primary"
        aria-label="visualization tabs"
      >
        <Tab 
          icon={<AccountTreeIcon />} 
          iconPosition="start" 
          label="Tree Visualization" 
          sx={{ 
            fontWeight: activeTab === 0 ? 600 : 500,
            color: activeTab === 0 ? '#4F46E5' : '#64748B'
          }}
        />
        <Tab 
          icon={<BarChartIcon />} 
          iconPosition="start" 
          label="Metrics Analysis" 
          sx={{ 
            fontWeight: activeTab === 1 ? 600 : 500,
            color: activeTab === 1 ? '#4F46E5' : '#64748B'
          }}
        />
      </Tabs>
    </Box>
  );
};

export default VisualizationTabs;