"""
Elegant Onboarding Experience

A thoughtfully crafted onboarding system that introduces the MCTS
visualization interface with delightful, instructive guidance.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import json
from typing import Dict, List, Any, Optional, Callable

from .design_system import (
    Colors, Typography, Spacing, Shadows, BorderRadius, 
    Animation, InteractionPatterns, ZIndex
)


class OnboardingStep:
    """A single step in the onboarding experience."""
    
    def __init__(self, title: str, content: str, target_element: str,
                 position: str = "right", image: Optional[str] = None):
        """
        Initialize an onboarding step.
        
        Args:
            title: Title of the step
            content: Descriptive content
            target_element: CSS selector for the target element
            position: Position of tooltip (top, right, bottom, left)
            image: Optional image URL to include
        """
        self.title = title
        self.content = content
        self.target_element = target_element
        self.position = position
        self.image = image


class OnboardingExperience:
    """
    Sophisticated onboarding experience for MCTS visualization.
    
    Creates a delightful, informative introduction to the interface
    with perfect pacing and visual elegance.
    """
    
    def __init__(self):
        """Initialize the onboarding experience."""
        self.steps = []
        self.current_step = 0
        
    def add_step(self, step: OnboardingStep) -> None:
        """Add a step to the onboarding sequence."""
        self.steps.append(step)
        
    def reset(self) -> None:
        """Reset the onboarding sequence."""
        self.current_step = 0
        
    def create_onboarding_overlay(self, app: dash.Dash) -> None:
        """
        Create an elegant onboarding overlay for the app.
        
        Args:
            app: The Dash app to add onboarding to
        """
        # Add onboarding components to app layout
        app.layout.children.append(html.Div(
            id="onboarding-overlay",
            className="onboarding-overlay",
            style={"display": "none"},
            children=[
                html.Div(
                    id="onboarding-backdrop",
                    className="onboarding-backdrop",
                    n_clicks=0
                ),
                html.Div(
                    id="onboarding-tooltip",
                    className="onboarding-tooltip",
                    children=[
                        html.Div(
                            className="onboarding-tooltip-header",
                            children=[
                                html.H4(id="onboarding-title", className="onboarding-title"),
                                html.Button(
                                    "×",
                                    id="onboarding-close",
                                    className="onboarding-close",
                                    n_clicks=0
                                )
                            ]
                        ),
                        html.Div(
                            id="onboarding-content",
                            className="onboarding-content"
                        ),
                        html.Div(
                            id="onboarding-image-container",
                            className="onboarding-image-container",
                            style={"display": "none"},
                            children=[
                                html.Img(id="onboarding-image", className="onboarding-image")
                            ]
                        ),
                        html.Div(
                            className="onboarding-footer",
                            children=[
                                html.Div(
                                    className="onboarding-progress",
                                    children=[
                                        html.Span(id="onboarding-step-indicator")
                                    ]
                                ),
                                html.Div(
                                    className="onboarding-buttons",
                                    children=[
                                        html.Button(
                                            "Skip",
                                            id="onboarding-skip",
                                            className="onboarding-skip",
                                            n_clicks=0
                                        ),
                                        html.Button(
                                            "Next",
                                            id="onboarding-next",
                                            className="onboarding-next",
                                            n_clicks=0
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        ))
        
        # Add onboarding trigger button
        app.layout.children.append(html.Button(
            "Show Tutorial",
            id="show-onboarding",
            className="show-onboarding",
            n_clicks=0
        ))
        
        # Add onboarding state store
        app.layout.children.append(dcc.Store(
            id="onboarding-state",
            data={"current_step": 0, "total_steps": len(self.steps)}
        ))
        
        # Add custom CSS for onboarding
        app.index_string = app.index_string.replace(
            "</style>",
            self._get_onboarding_css() + "</style>"
        )
        
        # Define callbacks
        self._add_callbacks(app)
        
    def _get_onboarding_css(self) -> str:
        """Generate CSS for onboarding components."""
        return f"""
        .onboarding-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: {ZIndex.INDICES["modal"]};
            pointer-events: none;
        }}
        
        .onboarding-backdrop {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(15, 23, 42, 0.75);
            pointer-events: auto;
        }}
        
        .onboarding-tooltip {{
            position: absolute;
            background-color: {Colors.SLATE_100};
            border-radius: {BorderRadius.RADIUS["lg"]};
            box-shadow: {Shadows.ELEVATIONS["xl"]};
            width: 350px;
            max-width: 90vw;
            overflow: hidden;
            pointer-events: auto;
            transform: translate(-50%, -50%);
            transition: {Animation.transition("all", "500")};
        }}
        
        .onboarding-tooltip-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.25rem;
            background-color: {Colors.INDIGO_500};
            color: white;
        }}
        
        .onboarding-title {{
            margin: 0;
            font-size: {Typography.FONT_SIZE["lg"]};
            font-weight: {Typography.FONT_WEIGHT["semibold"]};
        }}
        
        .onboarding-close {{
            background: transparent;
            border: none;
            color: white;
            font-size: 1.5rem;
            line-height: 1;
            cursor: pointer;
            padding: 0;
            margin: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: {Animation.transition("transform", "200")};
        }}
        
        .onboarding-close:hover {{
            transform: scale(1.2);
        }}
        
        .onboarding-content {{
            padding: 1.25rem;
            color: {Colors.SLATE_700};
            font-size: {Typography.FONT_SIZE["base"]};
            line-height: {Typography.LINE_HEIGHT["relaxed"]};
        }}
        
        .onboarding-image-container {{
            padding: 0 1.25rem;
        }}
        
        .onboarding-image {{
            width: 100%;
            border-radius: {BorderRadius.RADIUS["md"]};
            margin-bottom: 1rem;
        }}
        
        .onboarding-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.25rem;
            background-color: {Colors.SLATE_200};
            border-top: 1px solid {Colors.SLATE_300};
        }}
        
        .onboarding-progress {{
            display: flex;
            align-items: center;
        }}
        
        .onboarding-step-indicator {{
            font-size: {Typography.FONT_SIZE["sm"]};
            color: {Colors.SLATE_500};
        }}
        
        .onboarding-buttons {{
            display: flex;
            gap: 0.75rem;
        }}
        
        .onboarding-skip {{
            background-color: transparent;
            border: none;
            color: {Colors.SLATE_500};
            font-size: {Typography.FONT_SIZE["sm"]};
            padding: 0.5rem 0.75rem;
            cursor: pointer;
            border-radius: {BorderRadius.RADIUS["default"]};
            transition: {Animation.transition("color", "200")};
        }}
        
        .onboarding-skip:hover {{
            color: {Colors.SLATE_700};
        }}
        
        .onboarding-next {{
            background-color: {Colors.INDIGO_500};
            border: none;
            color: white;
            font-size: {Typography.FONT_SIZE["sm"]};
            font-weight: {Typography.FONT_WEIGHT["medium"]};
            padding: 0.5rem 1rem;
            cursor: pointer;
            border-radius: {BorderRadius.RADIUS["default"]};
            transition: {Animation.transition("all", "200")};
            box-shadow: {Shadows.ELEVATIONS["sm"]};
        }}
        
        .onboarding-next:hover {{
            background-color: {Colors.INDIGO_600};
            transform: translateY(-1px);
            box-shadow: {Shadows.ELEVATIONS["md"]};
        }}
        
        .onboarding-next:active {{
            transform: translateY(1px);
            box-shadow: {Shadows.ELEVATIONS["xs"]};
        }}
        
        .show-onboarding {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background-color: {Colors.INDIGO_500};
            color: white;
            border: none;
            border-radius: {BorderRadius.RADIUS["full"]};
            width: 56px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: {Shadows.ELEVATIONS["lg"]};
            transition: {Animation.transition("all", "300")};
            z-index: {ZIndex.INDICES["fixed"]};
        }}
        
        .show-onboarding:hover {{
            background-color: {Colors.INDIGO_600};
            transform: translateY(-2px);
            box-shadow: {Shadows.ELEVATIONS["xl"]};
        }}
        
        .show-onboarding:active {{
            transform: translateY(1px);
            box-shadow: {Shadows.ELEVATIONS["md"]};
        }}
        
        .onboarding-highlight {{
            position: relative;
            z-index: {ZIndex.INDICES["popover"]};
            box-shadow: 0 0 0 4px {Colors.INDIGO_500}, {Shadows.ELEVATIONS["xl"]};
            border-radius: 4px;
            transition: {Animation.transition("box-shadow", "300")};
        }}
        
        .onboarding-arrow {{
            position: absolute;
            width: 12px;
            height: 12px;
            background-color: {Colors.SLATE_100};
            transform: rotate(45deg);
            z-index: {ZIndex.INDICES["tooltip"]};
        }}
        
        .onboarding-arrow-top {{
            top: -6px;
            left: 50%;
            margin-left: -6px;
        }}
        
        .onboarding-arrow-right {{
            right: -6px;
            top: 50%;
            margin-top: -6px;
        }}
        
        .onboarding-arrow-bottom {{
            bottom: -6px;
            left: 50%;
            margin-left: -6px;
        }}
        
        .onboarding-arrow-left {{
            left: -6px;
            top: 50%;
            margin-top: -6px;
        }}
        """
        
    def _add_callbacks(self, app: dash.Dash) -> None:
        """Add onboarding callbacks to the app."""
        # Show onboarding callback
        @app.callback(
            [Output("onboarding-overlay", "style"),
             Output("onboarding-state", "data")],
            [Input("show-onboarding", "n_clicks")],
            [State("onboarding-state", "data")]
        )
        def show_onboarding(n_clicks, state_data):
            if n_clicks and n_clicks > 0:
                # Reset state
                state_data["current_step"] = 0
                state_data["total_steps"] = len(self.steps)
                return {"display": "block"}, state_data
            return {"display": "none"}, state_data
        
        # Update step content callback
        @app.callback(
            [Output("onboarding-title", "children"),
             Output("onboarding-content", "children"),
             Output("onboarding-step-indicator", "children"),
             Output("onboarding-image-container", "style"),
             Output("onboarding-image", "src"),
             Output("onboarding-tooltip", "style"),
             Output("onboarding-next", "children")],
            [Input("onboarding-state", "data")]
        )
        def update_onboarding_content(state_data):
            if not state_data or "current_step" not in state_data:
                return "", "", "", {"display": "none"}, "", {}, "Next"
                
            current_step = state_data["current_step"]
            if current_step >= len(self.steps):
                return "", "", "", {"display": "none"}, "", {}, "Next"
                
            step = self.steps[current_step]
            
            # Position tooltip relative to target element
            position_style = {
                "position": "absolute",
                "transform": "none",
                "transition": Animation.transition("all", "500")
            }
            
            # Last step handling
            next_button = "Finish" if current_step == len(self.steps) - 1 else "Next"
            
            # Image handling
            image_style = {"display": "block"} if step.image else {"display": "none"}
            
            # Step indicator
            step_indicator = f"Step {current_step + 1} of {len(self.steps)}"
            
            return (
                step.title,
                step.content,
                step_indicator,
                image_style,
                step.image or "",
                position_style,
                next_button
            )
        
        # Navigation callbacks
        @app.callback(
            [Output("onboarding-state", "data", allow_duplicate=True),
             Output("onboarding-overlay", "style", allow_duplicate=True)],
            [Input("onboarding-next", "n_clicks"),
             Input("onboarding-skip", "n_clicks"),
             Input("onboarding-close", "n_clicks"),
             Input("onboarding-backdrop", "n_clicks")],
            [State("onboarding-state", "data")],
            prevent_initial_call=True
        )
        def handle_navigation(next_clicks, skip_clicks, close_clicks, backdrop_clicks, state_data):
            ctx = dash.callback_context
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "onboarding-next":
                # Advance to next step
                if state_data["current_step"] < state_data["total_steps"] - 1:
                    state_data["current_step"] += 1
                    return state_data, {"display": "block"}
                else:
                    # End of onboarding
                    return state_data, {"display": "none"}
            else:
                # Skip, close, or backdrop click - end onboarding
                return state_data, {"display": "none"}
                
    def create_default_onboarding(self) -> None:
        """Create a default onboarding experience for MCTS visualization."""
        # Welcome step
        self.add_step(OnboardingStep(
            title="Welcome to MCTS Visualization",
            content="This tool helps you explore and understand Monte Carlo Tree Search algorithms through interactive visualization. Let's take a quick tour of the interface.",
            target_element="body",
            position="center"
        ))
        
        # Tree visualization step
        self.add_step(OnboardingStep(
            title="Tree Visualization",
            content="This is the main visualization area. It shows the MCTS tree structure with nodes representing states and edges representing actions.",
            target_element="#tree-graph",
            position="left",
            image="/assets/tree_visualization.png"
        ))
        
        # Node meaning step
        self.add_step(OnboardingStep(
            title="Understanding Nodes",
            content="Nodes represent states in the search space. Their size indicates visit count, and color represents different states. Hover over nodes to see detailed information.",
            target_element="#tree-graph",
            position="left"
        ))
        
        # Controls step
        self.add_step(OnboardingStep(
            title="Visualization Controls",
            content="Use these controls to change the layout type, adjust how node sizes are calculated, and reset the view when needed.",
            target_element=".card:first-child",
            position="right"
        ))
        
        # Layout options step
        self.add_step(OnboardingStep(
            title="Layout Options",
            content="Choose between radial and hierarchical layouts to explore the tree from different perspectives.",
            target_element="#layout-select",
            position="right"
        ))
        
        # Metrics step
        self.add_step(OnboardingStep(
            title="Metrics Analysis",
            content="Switch to the Metrics tab to see detailed analytics about the search process, including visit distributions and exploration patterns.",
            target_element="#tabs",
            position="bottom"
        ))
        
        # Key metrics step
        self.add_step(OnboardingStep(
            title="Key Metrics",
            content="These cards show important summary statistics about the search, including maximum visits, average value, and exploration rate.",
            target_element=".metrics-card:first-child",
            position="right"
        ))
        
        # Final step
        self.add_step(OnboardingStep(
            title="You're Ready to Explore!",
            content="You now know the basics of the MCTS visualization interface. Click and drag to pan, scroll to zoom, and hover over elements to see more details. Enjoy exploring!",
            target_element="body",
            position="center"
        ))


def create_mcts_guided_tour(app: dash.Dash) -> None:
    """
    Create a sophisticated guided tour for MCTS visualization.
    
    Args:
        app: The Dash app to add the tour to
    """
    # Create onboarding experience
    onboarding = OnboardingExperience()
    onboarding.create_default_onboarding()
    onboarding.create_onboarding_overlay(app)