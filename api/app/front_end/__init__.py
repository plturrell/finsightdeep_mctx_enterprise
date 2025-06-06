"""
MCTX Front-End Design System

A meticulously crafted design system and visualization interface 
for Monte Carlo Tree Search algorithms, inspired by Jony Ive's 
principles of clarity, purpose, and refinement in every detail.
"""

from .design_system import (
    Colors, Typography, Spacing, Shadows, BorderRadius, 
    Animation, InteractionPatterns, ZIndex, Layout, Breakpoints
)
from .mcts_visualization import MCTSVisualization, NodeState
from .animation_transitions import AnimationSystem
from .onboarding import OnboardingExperience, OnboardingStep, create_mcts_guided_tour

__all__ = [
    # Design system
    'Colors', 'Typography', 'Spacing', 'Shadows', 'BorderRadius',
    'Animation', 'InteractionPatterns', 'ZIndex', 'Layout', 'Breakpoints',
    
    # Visualization
    'MCTSVisualization', 'NodeState',
    
    # Animation
    'AnimationSystem',
    
    # Onboarding
    'OnboardingExperience', 'OnboardingStep', 'create_mcts_guided_tour',
]