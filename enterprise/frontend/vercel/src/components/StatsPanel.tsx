import React from 'react';
import {
  Card,
  CardContent,
  Grid,
  Typography,
  Box,
  Chip
} from '@mui/material';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import LayersIcon from '@mui/icons-material/Layers';
import MemoryIcon from '@mui/icons-material/Memory';

interface StatsPanelProps {
  stats: any;
}

const StatsPanel: React.FC<StatsPanelProps> = ({ stats }) => {
  return (
    <Card sx={{ mt: 3, borderRadius: 2, overflow: 'hidden' }}>
      <CardContent>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              icon={<AccessTimeIcon />}
              title="Duration"
              value={`${stats.duration_ms.toFixed(2)} ms`}
              description="Total search time"
              color="#6366F1"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              icon={<AccountTreeIcon />}
              title="Expanded Nodes"
              value={stats.num_expanded_nodes}
              description="Nodes visited during search"
              color="#10B981"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              icon={<LayersIcon />}
              title="Max Depth"
              value={stats.max_depth_reached}
              description="Deepest path explored"
              color="#F59E0B"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              icon={<MemoryIcon />}
              title="Optimization"
              value={
                <Chip 
                  label={stats.optimized ? (stats.distributed_stats ? "Distributed" : "T4") : "None"} 
                  size="small"
                  color={stats.optimized ? "primary" : "default"}
                  sx={{ 
                    fontWeight: 500,
                    fontSize: '0.875rem',
                  }}
                />
              }
              description={stats.precision}
              color="#8B5CF6"
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

interface StatCardProps {
  icon: React.ReactNode;
  title: string;
  value: React.ReactNode;
  description: string;
  color: string;
}

const StatCard: React.FC<StatCardProps> = ({ icon, title, value, description, color }) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        p: 2,
        height: '100%',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: `${color}15`,
          color: color,
          borderRadius: '50%',
          width: 48,
          height: 48,
          mb: 1.5,
        }}
      >
        {icon}
      </Box>
      
      <Typography variant="subtitle2" color="textSecondary" gutterBottom>
        {title}
      </Typography>
      
      <Typography 
        variant="h5" 
        component="div" 
        sx={{ 
          fontWeight: 600,
          color: 'text.primary',
          mb: 0.5,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '40px'
        }}
      >
        {value}
      </Typography>
      
      <Typography variant="body2" color="textSecondary">
        {description}
      </Typography>
    </Box>
  );
};

export default StatsPanel;