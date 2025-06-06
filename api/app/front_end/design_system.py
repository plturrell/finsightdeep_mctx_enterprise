"""
MCTX Design System

A meticulous design system for the MCTX visualization interface.
Follows Jony Ive's philosophy of simplicity, intentionality, and 
craftsmanship down to the pixel level.
"""

import colorsys
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional

class Colors:
    """
    A thoughtfully curated palette.
    
    Each color has been carefully selected to form a harmonious system,
    with mathematical relationships between hues and precise saturation
    levels. Naming follows a clear taxonomy that expresses purpose.
    """
    # Primary palette - natural, calm, focused
    SLATE_100 = "#F8FAFC"  # Background canvas
    SLATE_200 = "#E2E8F0"  # Subtle dividers
    SLATE_300 = "#CBD5E1"  # Disabled states
    SLATE_500 = "#64748B"  # Secondary text
    SLATE_700 = "#334155"  # Primary text
    SLATE_900 = "#0F172A"  # Headings
    
    # Accent colors - selected for their ability to complement
    INDIGO_50  = "#EEF2FF"  # Selected background
    INDIGO_100 = "#E0E7FF"  # Hover states
    INDIGO_500 = "#6366F1"  # Primary buttons
    INDIGO_600 = "#4F46E5"  # Primary active
    INDIGO_900 = "#312E81"  # Focus states
    
    # Functional colors - providing clear information hierarchies
    EMERALD_50  = "#ECFDF5"
    EMERALD_500 = "#10B981"  # Success states
    AMBER_50    = "#FFFBEB"
    AMBER_500   = "#F59E0B"  # Warning states
    RED_50      = "#FEF2F2"
    RED_500     = "#EF4444"  # Error states
    
    # Visualization color sequence - for data representation
    # Each carefully selected for distinguishability and harmony
    VISUALIZATION = [
        "#3B82F6",  # Blue
        "#8B5CF6",  # Violet
        "#EC4899",  # Pink
        "#F97316",  # Orange
        "#10B981",  # Emerald
        "#14B8A6",  # Teal
        "#06B6D4",  # Cyan
        "#0EA5E9",  # Light Blue
    ]
    
    @classmethod
    def get_sequential_palette(cls, base_color: str, steps: int = 5) -> List[str]:
        """
        Generate a mathematically perfect sequential palette.
        
        Creates a perceptually uniform gradient from light to dark
        for consistent data visualization.
        """
        # Convert hex to HSL
        h, l, s = cls._hex_to_hls(base_color)
        
        # Generate palette with precise lightness increments
        palette = []
        for i in range(steps):
            # Calculate new lightness with easing function for perceptual uniformity
            new_l = 0.95 - (0.7 * (i / (steps - 1)) ** 1.5)
            hex_color = cls._hls_to_hex(h, new_l, s)
            palette.append(hex_color)
            
        return palette
    
    @staticmethod
    def _hex_to_hls(hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to HLS colorspace."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return h, l, s
    
    @staticmethod
    def _hls_to_hex(h: float, l: float, s: float) -> str:
        """Convert HLS color to hex string."""
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        return hex_color


class Typography:
    """
    A comprehensive typographic system with intentional scale and harmony.
    
    Based on mathematical proportions inspired by classical design
    principles, ensuring perfect vertical rhythm and readability.
    """
    FONT_FAMILY = {
        "system": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        "mono": "SF Mono, SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace",
    }
    
    # Type scale with perfect mathematical proportion (1.25 ratio)
    FONT_SIZE = {
        "xs": "0.75rem",    # 12px
        "sm": "0.875rem",   # 14px
        "base": "1rem",     # 16px
        "lg": "1.125rem",   # 18px
        "xl": "1.25rem",    # 20px
        "2xl": "1.5rem",    # 24px
        "3xl": "1.875rem",  # 30px
        "4xl": "2.25rem",   # 36px
        "5xl": "3rem",      # 48px
    }
    
    # Precise line heights for optical alignment
    LINE_HEIGHT = {
        "none": "1",
        "tight": "1.25",
        "snug": "1.375",
        "normal": "1.5",
        "relaxed": "1.625",
        "loose": "2",
    }
    
    # Letter spacing for perfect readability at each size
    LETTER_SPACING = {
        "tighter": "-0.05em",
        "tight": "-0.025em",
        "normal": "0em",
        "wide": "0.025em",
        "wider": "0.05em",
        "widest": "0.1em",
    }
    
    # Font weights selected for both aesthetics and legibility
    FONT_WEIGHT = {
        "light": "300",
        "normal": "400",
        "medium": "500",
        "semibold": "600",
        "bold": "700",
    }
    
    @classmethod
    def style(cls, size: str, weight: str = "normal", height: str = "normal", 
              spacing: str = "normal", family: str = "system") -> Dict[str, str]:
        """
        Generate a complete typography style.
        
        Creates a harmonious combination of font properties
        for consistent application across the interface.
        """
        return {
            "font-family": cls.FONT_FAMILY[family],
            "font-size": cls.FONT_SIZE[size],
            "font-weight": cls.FONT_WEIGHT[weight],
            "line-height": cls.LINE_HEIGHT[height],
            "letter-spacing": cls.LETTER_SPACING[spacing],
        }


class Spacing:
    """
    A mathematically precise spacing system.
    
    Based on an 8-point grid for visual harmony across
    all dimensions, with 4-point accommodations for fine detail work.
    """
    # Base 4px system - mathematically sound with screen pixels
    UNIT = {
        "0": "0",
        "0.5": "0.125rem",  # 2px
        "1": "0.25rem",     # 4px
        "2": "0.5rem",      # 8px
        "3": "0.75rem",     # 12px
        "4": "1rem",        # 16px
        "5": "1.25rem",     # 20px
        "6": "1.5rem",      # 24px
        "8": "2rem",        # 32px
        "10": "2.5rem",     # 40px
        "12": "3rem",       # 48px
        "16": "4rem",       # 64px
        "20": "5rem",       # 80px
        "24": "6rem",       # 96px
        "32": "8rem",       # 128px
    }
    
    @classmethod
    def inset(cls, space: str) -> Dict[str, str]:
        """Create uniform inset spacing."""
        return {
            "padding": cls.UNIT[space]
        }
    
    @classmethod
    def stack(cls, space: str) -> Dict[str, str]:
        """Create vertical stack spacing between elements."""
        return {
            "margin-bottom": cls.UNIT[space]
        }
    
    @classmethod
    def inline(cls, space: str) -> Dict[str, str]:
        """Create horizontal inline spacing between elements."""
        return {
            "margin-right": cls.UNIT[space]
        }


class Shadows:
    """
    Thoughtfully crafted shadow system.
    
    Each shadow has been meticulously adjusted for realistic
    depth perception, with precise offset, blur, and opacity values.
    """
    ELEVATIONS = {
        "xs": "0 1px 2px rgba(15, 23, 42, 0.05)",
        "sm": "0 1px 3px rgba(15, 23, 42, 0.1), 0 1px 2px rgba(15, 23, 42, 0.06)",
        "md": "0 4px 6px -1px rgba(15, 23, 42, 0.1), 0 2px 4px -1px rgba(15, 23, 42, 0.06)",
        "lg": "0 10px 15px -3px rgba(15, 23, 42, 0.1), 0 4px 6px -2px rgba(15, 23, 42, 0.05)",
        "xl": "0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 10px 10px -5px rgba(15, 23, 42, 0.04)",
        "2xl": "0 25px 50px -12px rgba(15, 23, 42, 0.25)",
        "inner": "inset 0 2px 4px rgba(15, 23, 42, 0.06)",
        "focus": "0 0 0 3px rgba(99, 102, 241, 0.45)",
    }


class BorderRadius:
    """
    Unified radius system for interface elements.
    
    Provides a consistent visual language across all components,
    with mathematically derived increments.
    """
    RADIUS = {
        "none": "0",
        "sm": "0.125rem",     # 2px
        "default": "0.25rem", # 4px
        "md": "0.375rem",     # 6px
        "lg": "0.5rem",       # 8px
        "xl": "0.75rem",      # 12px
        "2xl": "1rem",        # 16px
        "3xl": "1.5rem",      # 24px
        "full": "9999px",     # Circle
    }


class Animation:
    """
    Precisely timed animation system.
    
    Carefully calibrated timing functions that reflect the
    natural world and create a sense of physicality.
    """
    DURATION = {
        "75": "75ms",
        "100": "100ms",
        "150": "150ms",
        "200": "200ms",
        "300": "300ms",
        "500": "500ms",
        "700": "700ms",
        "1000": "1000ms",
    }
    
    EASING = {
        "default": "cubic-bezier(0.4, 0, 0.2, 1)",  # Smooth, natural movement
        "in": "cubic-bezier(0.4, 0, 1, 1)",         # Acceleration from zero
        "out": "cubic-bezier(0, 0, 0.2, 1)",        # Deceleration to zero
        "in-out": "cubic-bezier(0.4, 0, 0.2, 1)",   # Acceleration until halfway, then deceleration
    }
    
    @classmethod
    def transition(cls, property_name: str, duration: str = "200", 
                   easing: str = "default") -> str:
        """Generate a transition with perfect timing."""
        return f"{property_name} {cls.DURATION[duration]} {cls.EASING[easing]}"


class ComponentState(Enum):
    """Enumerated component states for consistent interaction patterns."""
    DEFAULT = "default"
    HOVER = "hover"
    ACTIVE = "active"
    FOCUS = "focus"
    DISABLED = "disabled"


class InteractionPatterns:
    """
    Consistent interaction behaviors.
    
    Defines how elements respond to user input, with precise
    timing and visual feedback.
    """
    BUTTON = {
        ComponentState.DEFAULT: {
            "background": Colors.INDIGO_500,
            "color": Colors.SLATE_100,
            "border": "none",
            "shadow": Shadows.ELEVATIONS["sm"],
            "transform": "translateY(0)",
        },
        ComponentState.HOVER: {
            "background": Colors.INDIGO_600,
            "transform": "translateY(-1px)",
            "shadow": Shadows.ELEVATIONS["md"],
        },
        ComponentState.ACTIVE: {
            "background": Colors.INDIGO_600,
            "transform": "translateY(1px)",
            "shadow": Shadows.ELEVATIONS["xs"],
        },
        ComponentState.FOCUS: {
            "shadow": Shadows.ELEVATIONS["focus"],
        },
        ComponentState.DISABLED: {
            "background": Colors.SLATE_300,
            "color": Colors.SLATE_500,
            "shadow": "none",
            "cursor": "not-allowed",
        }
    }
    
    INPUT = {
        ComponentState.DEFAULT: {
            "border": f"1px solid {Colors.SLATE_300}",
            "background": Colors.SLATE_100,
            "color": Colors.SLATE_700,
        },
        ComponentState.HOVER: {
            "border": f"1px solid {Colors.SLATE_500}",
        },
        ComponentState.FOCUS: {
            "border": f"1px solid {Colors.INDIGO_500}",
            "shadow": Shadows.ELEVATIONS["focus"],
        },
        ComponentState.DISABLED: {
            "background": Colors.SLATE_200,
            "color": Colors.SLATE_500,
            "cursor": "not-allowed",
        }
    }


class ZIndex:
    """
    Layering system for components.
    
    Ensures consistent stacking of interface elements
    with logical numerical increments.
    """
    INDICES = {
        "negative": "-1",       # Below content
        "base": "0",            # Default layer
        "above": "10",          # Above content
        "dropdown": "20",       # Dropdown menus
        "sticky": "30",         # Sticky elements
        "fixed": "40",          # Fixed elements
        "modal": "50",          # Modal dialogs
        "popover": "60",        # Popovers
        "tooltip": "70",        # Tooltips
    }


class Layout:
    """
    Layout primitives for composition.
    
    Provides building blocks for crafting interfaces
    with consistent spatial relationships.
    """
    CONTAINER_WIDTH = {
        "xs": "20rem",      # 320px
        "sm": "24rem",      # 384px
        "md": "28rem",      # 448px
        "lg": "32rem",      # 512px
        "xl": "36rem",      # 576px
        "2xl": "42rem",     # 672px
        "3xl": "48rem",     # 768px
        "4xl": "56rem",     # 896px
        "5xl": "64rem",     # 1024px
        "6xl": "72rem",     # 1152px
        "7xl": "80rem",     # 1280px
        "full": "100%",
    }
    
    ASPECT_RATIO = {
        "square": "1 / 1",
        "video": "16 / 9",
        "golden": "1.618 / 1",
    }


class Breakpoints:
    """
    Responsive design breakpoints.
    
    Carefully selected to accommodate common device sizes
    while maintaining visual consistency.
    """
    POINTS = {
        "sm": "640px",
        "md": "768px",
        "lg": "1024px",
        "xl": "1280px",
        "2xl": "1536px",
    }