#!/usr/bin/env python3
"""
HANA Performance Dashboard for MCTS Service

This module implements a monitoring dashboard for SAP HANA performance
metrics related to the MCTS service. It uses Smart Data Integration to
analyze performance patterns and provide optimization recommendations.
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.app.db.hana_connector import hana_manager
from api.app.core.config import get_settings

# Initialize settings and logger
settings = get_settings()
logger = logging.getLogger("mctx.dashboard")


class HANAPerformanceMonitor:
    """Monitor for SAP HANA performance metrics related to MCTS service."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        # Initialize HANA connector
        hana_manager.initialize()
        
        # Performance metrics cache
        self.metrics_cache = {
            "last_refresh": 0,
            "cache_ttl": 60,  # seconds
            "performance_data": None,
            "daily_stats": None,
            "optimization_stats": None,
        }
    
    def get_performance_metrics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get performance metrics from HANA.
        
        Args:
            force_refresh: Whether to force a refresh of cached data.
            
        Returns:
            Dictionary of performance metrics.
        """
        current_time = time.time()
        if (not force_refresh and 
            self.metrics_cache["performance_data"] is not None and
            current_time - self.metrics_cache["last_refresh"] < self.metrics_cache["cache_ttl"]):
            return self.metrics_cache["performance_data"]
        
        try:
            # Query to get performance metrics
            with hana_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Query for execution times
                    cursor.execute("""
                    SELECT 
                        SEARCH_TYPE, 
                        AVG(DURATION_MS) as AVG_DURATION,
                        STDDEV(DURATION_MS) as STDDEV_DURATION,
                        MIN(DURATION_MS) as MIN_DURATION,
                        MAX(DURATION_MS) as MAX_DURATION,
                        COUNT(*) as EXECUTION_COUNT
                    FROM MCTX_SEARCH_HISTORY
                    WHERE TIMESTAMP > ADD_DAYS(CURRENT_TIMESTAMP, -30)
                    GROUP BY SEARCH_TYPE
                    """)
                    
                    execution_stats = []
                    for row in cursor.fetchall():
                        execution_stats.append({
                            "search_type": row[0],
                            "avg_duration": row[1],
                            "stddev_duration": row[2],
                            "min_duration": row[3],
                            "max_duration": row[4],
                            "execution_count": row[5]
                        })
                    
                    # Query for HANA system metrics
                    cursor.execute("""
                    SELECT 
                        HOST, 
                        SERVICE_NAME, 
                        ROUND(AVG(TOTAL_MEMORY_USED_SIZE)/1024/1024, 2) as AVG_MEM_USED_MB,
                        ROUND(MAX(TOTAL_MEMORY_USED_SIZE)/1024/1024, 2) as MAX_MEM_USED_MB,
                        ROUND(AVG(TOTAL_CPU_USAGE), 2) as AVG_CPU,
                        ROUND(MAX(TOTAL_CPU_USAGE), 2) as MAX_CPU
                    FROM M_SERVICE_STATISTICS 
                    WHERE STATISTICS_TIME > ADD_DAYS(CURRENT_TIMESTAMP, -1)
                    GROUP BY HOST, SERVICE_NAME
                    """)
                    
                    system_stats = []
                    for row in cursor.fetchall():
                        system_stats.append({
                            "host": row[0],
                            "service_name": row[1],
                            "avg_mem_used_mb": row[2],
                            "max_mem_used_mb": row[3],
                            "avg_cpu": row[4],
                            "max_cpu": row[5]
                        })
                    
                    # Query for optimization metrics
                    cursor.execute("""
                    SELECT 
                        DATE_KEY,
                        SEARCH_TYPE,
                        TOTAL_SEARCHES,
                        AVG_DURATION_MS,
                        T4_OPTIMIZED_COUNT,
                        DISTRIBUTED_COUNT,
                        AVG_DEVICES
                    FROM MCTX_SEARCH_STATISTICS
                    WHERE DATE_KEY > ADD_DAYS(CURRENT_DATE, -30)
                    ORDER BY DATE_KEY DESC
                    """)
                    
                    optimization_stats = []
                    for row in cursor.fetchall():
                        optimization_stats.append({
                            "date": row[0],
                            "search_type": row[1],
                            "total_searches": row[2],
                            "avg_duration_ms": row[3],
                            "t4_optimized_count": row[4],
                            "distributed_count": row[5],
                            "avg_devices": row[6]
                        })
            
            # Combine all metrics
            performance_data = {
                "execution_stats": execution_stats,
                "system_stats": system_stats,
                "optimization_stats": optimization_stats
            }
            
            # Update cache
            self.metrics_cache["performance_data"] = performance_data
            self.metrics_cache["last_refresh"] = current_time
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            # Return cached data if available
            if self.metrics_cache["performance_data"] is not None:
                return self.metrics_cache["performance_data"]
            # Return empty data otherwise
            return {
                "execution_stats": [],
                "system_stats": [],
                "optimization_stats": []
            }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations based on performance metrics.
        
        Returns:
            List of optimization recommendations.
        """
        recommendations = []
        performance_data = self.get_performance_metrics()
        
        # Check for excessive query times
        for stat in performance_data.get("execution_stats", []):
            if stat.get("avg_duration", 0) > 5000:  # More than 5 seconds
                recommendations.append({
                    "type": "performance",
                    "severity": "high",
                    "message": f"High average execution time ({stat['avg_duration']:.2f}ms) for {stat['search_type']} searches",
                    "recommendation": "Consider increasing the use of T4 optimizations or distributed computing"
                })
        
        # Check for memory usage
        for stat in performance_data.get("system_stats", []):
            if stat.get("max_mem_used_mb", 0) > 8000:  # More than 8GB
                recommendations.append({
                    "type": "memory",
                    "severity": "medium",
                    "message": f"High memory usage ({stat['max_mem_used_mb']}MB) on {stat['host']}",
                    "recommendation": "Consider memory optimization or scaling up HANA instances"
                })
        
        # Check for optimization usage
        opt_stats = performance_data.get("optimization_stats", [])
        if opt_stats:
            total_searches = sum(stat.get("total_searches", 0) for stat in opt_stats)
            t4_searches = sum(stat.get("t4_optimized_count", 0) for stat in opt_stats)
            distributed_searches = sum(stat.get("distributed_count", 0) for stat in opt_stats)
            
            if total_searches > 0:
                t4_percentage = (t4_searches / total_searches) * 100
                distributed_percentage = (distributed_searches / total_searches) * 100
                
                if t4_percentage < 20:
                    recommendations.append({
                        "type": "optimization",
                        "severity": "medium",
                        "message": f"Low T4 optimization usage ({t4_percentage:.1f}%)",
                        "recommendation": "Increase use of T4 optimizations for better performance"
                    })
                
                if distributed_percentage < 10 and total_searches > 1000:
                    recommendations.append({
                        "type": "optimization",
                        "severity": "low",
                        "message": f"Low distributed search usage ({distributed_percentage:.1f}%)",
                        "recommendation": "Consider using distributed search for large-scale workloads"
                    })
        
        return recommendations


def create_dashboard():
    """Create and configure the Dash dashboard application."""
    monitor = HANAPerformanceMonitor()
    app = Dash(__name__, title="HANA Performance Dashboard")
    
    app.layout = html.Div([
        html.H1("SAP HANA Performance Dashboard for MCTS Service"),
        
        html.Div([
            html.Button("Refresh Data", id="refresh-button", className="refresh-button"),
            html.Div(id="last-refresh-time")
        ], className="header-controls"),
        
        html.Div([
            html.H2("Performance Metrics"),
            
            html.Div([
                html.Div([
                    html.H3("Execution Times by Search Type"),
                    dcc.Graph(id="execution-times-graph")
                ], className="graph-container"),
                
                html.Div([
                    html.H3("System Resource Usage"),
                    dcc.Graph(id="system-usage-graph")
                ], className="graph-container")
            ], className="graph-row"),
            
            html.Div([
                html.Div([
                    html.H3("Optimization Usage Trends"),
                    dcc.Graph(id="optimization-trends-graph")
                ], className="graph-container"),
                
                html.Div([
                    html.H3("Search Type Distribution"),
                    dcc.Graph(id="search-type-pie")
                ], className="graph-container")
            ], className="graph-row")
        ]),
        
        html.Div([
            html.H2("Optimization Recommendations"),
            html.Div(id="recommendations-container", className="recommendations")
        ]),
        
        # Hidden div for storing the data
        html.Div(id="performance-data", style={"display": "none"}),
        
        # Refresh interval
        dcc.Interval(
            id="interval-component",
            interval=60 * 1000,  # in milliseconds (1 minute)
            n_intervals=0
        )
    ])
    
    @app.callback(
        [Output("performance-data", "children"),
         Output("last-refresh-time", "children")],
        [Input("refresh-button", "n_clicks"),
         Input("interval-component", "n_intervals")]
    )
    def update_data(n_clicks, n_intervals):
        """Update the performance data."""
        data = monitor.get_performance_metrics(force_refresh=True)
        refresh_time = time.strftime("%H:%M:%S")
        return json.dumps(data), f"Last refreshed: {refresh_time}"
    
    @app.callback(
        [Output("execution-times-graph", "figure"),
         Output("system-usage-graph", "figure"),
         Output("optimization-trends-graph", "figure"),
         Output("search-type-pie", "figure"),
         Output("recommendations-container", "children")],
        [Input("performance-data", "children")]
    )
    def update_graphs(json_data):
        """Update all graphs based on the performance data."""
        if not json_data:
            # Return empty figures if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, html.Div("No recommendations available")
        
        # Parse the JSON data
        data = json.loads(json_data)
        
        # Create execution times graph
        exec_df = pd.DataFrame(data.get("execution_stats", []))
        if not exec_df.empty:
            exec_fig = px.bar(
                exec_df, 
                x="search_type", 
                y="avg_duration",
                error_y="stddev_duration",
                color="search_type",
                labels={"search_type": "Search Type", "avg_duration": "Average Duration (ms)"},
                title="Average Execution Time by Search Type"
            )
        else:
            exec_fig = go.Figure()
            exec_fig.update_layout(title="No execution data available")
        
        # Create system usage graph
        sys_df = pd.DataFrame(data.get("system_stats", []))
        if not sys_df.empty:
            sys_fig = px.bar(
                sys_df,
                x="service_name",
                y=["avg_cpu", "max_cpu"],
                barmode="group",
                labels={
                    "service_name": "Service", 
                    "value": "CPU Usage (%)",
                    "variable": "Metric"
                },
                title="CPU Usage by Service"
            )
        else:
            sys_fig = go.Figure()
            sys_fig.update_layout(title="No system data available")
        
        # Create optimization trends graph
        opt_df = pd.DataFrame(data.get("optimization_stats", []))
        if not opt_df.empty:
            # Convert date strings to datetime
            opt_df["date"] = pd.to_datetime(opt_df["date"])
            # Group by date
            grouped_df = opt_df.groupby("date").agg({
                "total_searches": "sum",
                "t4_optimized_count": "sum",
                "distributed_count": "sum"
            }).reset_index()
            
            opt_fig = px.line(
                grouped_df,
                x="date",
                y=["total_searches", "t4_optimized_count", "distributed_count"],
                labels={
                    "date": "Date",
                    "value": "Count",
                    "variable": "Metric"
                },
                title="Optimization Usage Trends"
            )
        else:
            opt_fig = go.Figure()
            opt_fig.update_layout(title="No optimization data available")
        
        # Create search type pie chart
        if not exec_df.empty:
            pie_fig = px.pie(
                exec_df,
                names="search_type",
                values="execution_count",
                title="Search Type Distribution"
            )
        else:
            pie_fig = go.Figure()
            pie_fig.update_layout(title="No search type data available")
        
        # Get recommendations
        recommendations = monitor.get_optimization_recommendations()
        
        # Create recommendation elements
        rec_elements = []
        for rec in recommendations:
            severity_class = f"recommendation-{rec['severity']}"
            rec_elements.append(html.Div([
                html.H4(rec["message"]),
                html.P(rec["recommendation"]),
                html.Span(rec["type"], className="recommendation-type")
            ], className=f"recommendation {severity_class}"))
        
        if not rec_elements:
            rec_elements = [html.Div("No recommendations at this time.", className="no-recommendations")]
        
        return exec_fig, sys_fig, opt_fig, pie_fig, rec_elements
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HANA Performance Dashboard")
    parser.add_argument("--host", default="localhost", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    app = create_dashboard()
    app.run_server(host=args.host, port=args.port, debug=args.debug)