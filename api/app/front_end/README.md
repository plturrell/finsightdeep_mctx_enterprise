# MCTX Front-End Visualization System

A sophisticated visualization system for Monte Carlo Tree Search algorithms, designed with principles of clarity, purpose, and refinement in every detail.

## Installation

### Prerequisites

- Python 3.8+
- Node.js (for React components)
- MCTX library

### Setup

1. Install the required dependencies:

```bash
cd mctx/api/app/front_end
pip install -r requirements.txt
```

2. Install React dependencies:

```bash
npm install --prefix ./components
```

3. Make sure the MCTX library is installed:

```bash
cd /path/to/mctx
pip install -e .
```

## Running the Visualization

You can run the visualization system standalone:

```bash
cd mctx/api/app/front_end
python run_visualization.py --host 0.0.0.0 --port 8050
```

Then open your browser to `http://localhost:8050`

## Integration with MCTS Service

To integrate the visualization with the MCTS service in your own application:

```python
from api.app.services.mcts_service import MCTSService
from api.app.front_end.visualization_service import VisualizationService
from api.app.front_end.mcts_visualization import MCTSVisualization

# Create services
mcts_service = MCTSService()
vis_service = VisualizationService(mcts_service)

# Run a search
result = mcts_service.run_search(request)

# Convert to visualization data
vis_data = vis_service.convert_search_result_to_vis_data(result)

# Create visualization
vis = MCTSVisualization()
app = vis.create_dashboard(vis_data)
app.run_server()
```

## Components

### Design System (`design_system.py`)

The foundational design language that defines colors, typography, spacing, shadows, animation, and interactions.

### MCTS Visualization (`mcts_visualization.py`)

The primary visualization components for tree visualization, metrics panels, and interactive dashboards.

### Animation Transitions (`animation_transitions.py`)

A sophisticated animation system for creating elegant transitions between visualization states.

### Onboarding Experience (`onboarding.py`)

A guided tour system for introducing users to the interface.

### Visualization Service (`visualization_service.py`)

The service that connects the MCTS backend with the visualization front-end.

## Directory Structure

```
front_end/
├── __init__.py           # Package exports
├── assets/               # Static assets for visualization
├── components/           # React components
├── design_system.py      # Core design system
├── mcts_visualization.py # Tree visualization
├── animation_transitions.py # Animation system
├── onboarding.py         # Guided tour system
├── visualization_service.py # Backend connection
├── run_visualization.py  # Standalone server
└── requirements.txt      # Dependencies
```

## Troubleshooting

If you encounter issues:

1. Make sure all dependencies are installed
2. Check that the MCTX library is in your Python path
3. Verify that the asset directories exist
4. Look at the server logs for detailed error messages

## Credits

Designed and implemented with principles inspired by Jony Ive's approach to product design, focusing on clarity, purpose, and refinement in every detail.