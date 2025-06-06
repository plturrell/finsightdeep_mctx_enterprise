import { useState, useEffect } from 'react';
import Head from 'next/head';
import { 
  Container, 
  Typography, 
  Grid, 
  Card, 
  CardHeader, 
  CardContent,
  Button,
  Box,
  CircularProgress
} from '@mui/material';

import VisualizationTabs from '@/components/VisualizationTabs';
import ControlPanel from '@/components/ControlPanel';
import StatsPanel from '@/components/StatsPanel';
import { runSearch } from '@/lib/api';
import { defaultSearchParams } from '@/lib/constants';

export default function Home() {
  // State for visualization data
  const [visData, setVisData] = useState(null);
  // State for search parameters
  const [searchParams, setSearchParams] = useState(defaultSearchParams);
  // State for search statistics
  const [searchStats, setSearchStats] = useState(null);
  // Loading state
  const [loading, setLoading] = useState(false);
  // Active tab
  const [activeTab, setActiveTab] = useState(0);

  // Function to handle running a new search
  const handleRunSearch = async () => {
    setLoading(true);
    try {
      const result = await runSearch(searchParams);
      
      // Update visualization data
      setVisData(result.visualization_data);
      
      // Update statistics
      setSearchStats(result.statistics);
    } catch (error) {
      console.error('Error running search:', error);
      // Handle error state
    } finally {
      setLoading(false);
    }
  };

  // Function to handle parameter changes
  const handleParamChange = (name: string, value: any) => {
    setSearchParams(prev => ({
      ...prev,
      [name]: value
    }));
  };

  return (
    <>
      <Head>
        <title>MCTX Visualization</title>
        <meta name="description" content="Monte Carlo Tree Search Visualization" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <Box 
        sx={{ 
          bgcolor: '#0f172a', 
          color: 'white', 
          py: 3, 
          mb: 4,
          boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="h3" component="h1" gutterBottom>
            Monte Carlo Tree Search
          </Typography>
          <Typography variant="subtitle1">
            An elegant, information-rich visualization of MCTS algorithms
          </Typography>
        </Container>
      </Box>
      
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {/* Control Panel */}
          <Grid item xs={12} md={3}>
            <ControlPanel 
              searchParams={searchParams} 
              onParamChange={handleParamChange}
              onRunSearch={handleRunSearch}
              loading={loading}
            />
          </Grid>
          
          {/* Main Visualization */}
          <Grid item xs={12} md={9}>
            <Card 
              sx={{ 
                borderRadius: 2, 
                overflow: 'hidden',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                transition: 'box-shadow 0.2s, transform 0.2s',
                '&:hover': {
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                  transform: 'translateY(-2px)'
                }
              }}
            >
              <CardHeader 
                title={
                  <VisualizationTabs 
                    activeTab={activeTab}
                    onChange={(newValue) => setActiveTab(newValue)}
                  />
                }
                sx={{ 
                  bgcolor: '#f1f5f9',
                  borderBottom: '1px solid #e2e8f0',
                  p: 0
                }}
              />
              <CardContent sx={{ p: 0, height: '70vh', position: 'relative' }}>
                {loading ? (
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      justifyContent: 'center', 
                      alignItems: 'center',
                      height: '100%'
                    }}
                  >
                    <CircularProgress />
                  </Box>
                ) : !visData ? (
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      flexDirection: 'column',
                      justifyContent: 'center', 
                      alignItems: 'center',
                      height: '100%',
                      color: '#64748b',
                      textAlign: 'center',
                      p: 3
                    }}
                  >
                    <Box 
                      component="span" 
                      sx={{ 
                        fontSize: '3rem', 
                        mb: 2
                      }}
                    >
                      🔍
                    </Box>
                    <Typography variant="h5" gutterBottom>
                      Run a search to visualize the MCTS tree
                    </Typography>
                    <Typography variant="body1" color="textSecondary" sx={{ maxWidth: '400px', mb: 3 }}>
                      Configure the search parameters and click "Run Search" to generate a visualization
                    </Typography>
                    <Button 
                      variant="contained" 
                      color="primary" 
                      onClick={handleRunSearch}
                      disabled={loading}
                    >
                      Run Search
                    </Button>
                  </Box>
                ) : (
                  <div style={{ height: '100%', display: activeTab === 0 ? 'block' : 'none' }}>
                    {/* Tree visualization component */}
                    {visData && (
                      <div id="tree-visualization" style={{ height: '100%', width: '100%' }}>
                        {/* This is where the Plotly visualization will be rendered */}
                      </div>
                    )}
                  </div>
                )}
                
                <div style={{ height: '100%', display: activeTab === 1 ? 'block' : 'none' }}>
                  {/* Metrics visualization component */}
                  {visData && (
                    <div id="metrics-visualization" style={{ height: '100%', width: '100%' }}>
                      {/* This is where the metrics visualization will be rendered */}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
            
            {/* Statistics Panel */}
            {searchStats && (
              <StatsPanel stats={searchStats} />
            )}
          </Grid>
        </Grid>
      </Container>
    </>
  );
}