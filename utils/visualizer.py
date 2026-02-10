"""
Visualizer Utility
Creates visualizations for Streamlit dashboard
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class Visualizer:
    """Creates visualizations for video analysis results"""
    
    @staticmethod
    def create_quality_gauge(score: float, title: str = "Video Quality Score") -> go.Figure:
        """
        Create a gauge chart for quality score
        
        Args:
            score: Quality score (0-1)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Determine color based on score
        if score >= 0.75:
            color = "green"
        elif score >= 0.5:
            color = "lightgreen"
        elif score >= 0.25:
            color = "orange"
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            number={'suffix': "", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.25], 'color': '#ffcccc'},
                    {'range': [0.25, 0.5], 'color': '#ffffcc'},
                    {'range': [0.5, 0.75], 'color': '#ccffcc'},
                    {'range': [0.75, 1], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        
        return fig
    
    @staticmethod
    def create_temporal_plot(
        frame_indices: List[int],
        values: List[float],
        title: str,
        ylabel: str,
        threshold: Optional[float] = None
    ) -> go.Figure:
        """
        Create line plot for temporal analysis
        
        Args:
            frame_indices: Frame numbers
            values: Values to plot
            title: Chart title
            ylabel: Y-axis label
            threshold: Optional threshold line
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=frame_indices,
            y=values,
            mode='lines',
            name=ylabel,
            line=dict(color='blue', width=2)
        ))
        
        # Add threshold line if provided
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold}",
                annotation_position="right"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Frame Number",
            yaxis_title=ylabel,
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_issue_timeline(
        total_frames: int,
        issue_frames: List[int],
        issue_type: str
    ) -> go.Figure:
        """
        Create timeline visualization for detected issues
        
        Args:
            total_frames: Total number of frames
            issue_frames: Frame indices with issues
            issue_type: Type of issue
            
        Returns:
            Plotly figure
        """
        # Create binary array (0 = no issue, 1 = issue)
        timeline = np.zeros(total_frames)
        timeline[issue_frames] = 1
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(total_frames)),
            y=timeline,
            mode='markers',
            marker=dict(
                color=timeline,
                colorscale=[[0, 'green'], [1, 'red']],
                size=8,
                line=dict(width=0)
            ),
            name=issue_type,
            hovertemplate='Frame: %{x}<br>Issue: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{issue_type} Detection Timeline",
            xaxis_title="Frame Number",
            yaxis_title="Issue Detected",
            yaxis=dict(tickvals=[0, 1], ticktext=['No', 'Yes']),
            height=300,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_histogram(
        values: List[float],
        title: str,
        xlabel: str,
        bins: int = 50
    ) -> go.Figure:
        """
        Create histogram
        
        Args:
            values: Values to plot
            title: Chart title
            xlabel: X-axis label
            bins: Number of bins
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=bins,
            marker_color='blue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_metrics_table(metrics: Dict) -> pd.DataFrame:
        """
        Create metrics table for display
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Pandas DataFrame
        """
        data = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            data.append({
                "Metric": key.replace("_", " ").title(),
                "Value": formatted_value
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_comparison_bar(
        categories: List[str],
        values: List[float],
        title: str,
        ylabel: str
    ) -> go.Figure:
        """
        Create bar chart for comparisons
        
        Args:
            categories: Category names
            values: Values for each category
            title: Chart title
            ylabel: Y-axis label
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color='lightblue',
            text=values,
            texttemplate='%{text:.2f}',
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title=ylabel,
            height=400,
            showlegend=False
        )
        
        return fig
