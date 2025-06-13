import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class ChartGenerator:
    """Generate interactive charts for stock analysis"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_price_chart(self, data, symbol):
        """Create interactive price chart with candlesticks"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} Stock Price', 'Technical Indicators'),
                row_width=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.color_palette['success'],
                    decreasing_line_color=self.color_palette['danger']
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            if 'MA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MA_20'],
                        line=dict(color=self.color_palette['primary'], width=1),
                        name='MA 20',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            if 'MA_50' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MA_50'],
                        line=dict(color=self.color_palette['secondary'], width=1),
                        name='MA 50',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Add RSI if available
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        line=dict(color=self.color_palette['info'], width=2),
                        name='RSI',
                    ),
                    row=2, col=1
                )
                
                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Stock Analysis',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                showlegend=True,
                height=600,
                hovermode='x unified'
            )
            
            # Update x-axis
            fig.update_xaxes(
                rangeslider_visible=False,
                showspikes=True,
                spikecolor="grey",
                spikesnap="cursor",
                spikemode="across"
            )
            
            # Update y-axis
            fig.update_yaxes(
                showspikes=True,
                spikecolor="grey",
                spikesnap="cursor",
                spikemode="across"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating price chart: {str(e)}")
            return self.create_empty_chart("Error creating price chart")
    
    def create_volume_chart(self, data, symbol):
        """Create volume chart"""
        try:
            fig = go.Figure()
            
            # Volume bars with color based on price change
            colors = []
            for i in range(len(data)):
                if i == 0:
                    colors.append(self.color_palette['primary'])
                else:
                    if data['Close'].iloc[i] >= data['Close'].iloc[i-1]:
                        colors.append(self.color_palette['success'])
                    else:
                        colors.append(self.color_palette['danger'])
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                )
            )
            
            fig.update_layout(
                title=f'{symbol} Trading Volume',
                xaxis_title='Date',
                yaxis_title='Volume',
                template='plotly_white',
                showlegend=False,
                height=300,
                hovermode='x'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating volume chart: {str(e)}")
            return self.create_empty_chart("Error creating volume chart")
    
    def create_csv_chart(self, data):
        """Create chart from CSV data"""
        try:
            fig = go.Figure()
            
            # Try to identify appropriate columns for charting
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            date_cols = []
            
            # Look for date columns
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    date_cols.append(col)
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                x_col = date_cols[0]
                
                # Plot up to 5 numeric columns
                colors = [self.color_palette['primary'], self.color_palette['secondary'], 
                         self.color_palette['success'], self.color_palette['danger'], 
                         self.color_palette['warning']]
                
                for i, col in enumerate(numeric_cols[:5]):
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_col],
                            y=data[col],
                            mode='lines',
                            name=col,
                            line=dict(color=colors[i % len(colors)], width=2)
                        )
                    )
                
                fig.update_layout(
                    title='CSV Data Visualization',
                    xaxis_title=x_col,
                    yaxis_title='Value',
                    template='plotly_white',
                    showlegend=True,
                    height=500,
                    hovermode='x unified'
                )
                
            else:
                # Fallback: create a simple bar chart of first numeric column
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig.add_trace(
                        go.Bar(
                            x=data.index[:50],  # Limit to first 50 rows
                            y=data[col][:50],
                            name=col,
                            marker_color=self.color_palette['primary']
                        )
                    )
                    
                    fig.update_layout(
                        title=f'CSV Data: {col}',
                        xaxis_title='Index',
                        yaxis_title=col,
                        template='plotly_white',
                        showlegend=False,
                        height=400
                    )
            
            return fig
            
        except Exception as e:
            print(f"Error creating CSV chart: {str(e)}")
            return self.create_empty_chart("Error creating chart from CSV data")
    
    def create_correlation_heatmap(self, data):
        """Create correlation heatmap for numeric columns"""
        try:
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            
            if len(numeric_data.columns) < 2:
                return self.create_empty_chart("Not enough numeric columns for correlation")
            
            correlation_matrix = numeric_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Correlation Heatmap',
                template='plotly_white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {str(e)}")
            return self.create_empty_chart("Error creating correlation heatmap")
    
    def create_empty_chart(self, message):
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            showlegend=False
        )
        return fig
    
    def create_performance_metrics_chart(self, data):
        """Create performance metrics visualization"""
        try:
            if 'Daily_Return' not in data.columns:
                return self.create_empty_chart("No return data available")
            
            returns = data['Daily_Return'].dropna()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Returns Distribution', 'Cumulative Returns', 
                               'Rolling Volatility', 'Drawdown'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Returns histogram
            fig.add_trace(
                go.Histogram(
                    x=returns * 100,
                    name='Daily Returns (%)',
                    nbinsx=50,
                    marker_color=self.color_palette['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Cumulative returns
            cumulative_returns = (1 + returns).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns * 100,
                    mode='lines',
                    name='Cumulative Returns (%)',
                    line=dict(color=self.color_palette['success'], width=2)
                ),
                row=1, col=2
            )
            
            # Rolling volatility
            if 'Volatility' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Volatility'] * 100,
                        mode='lines',
                        name='30-Day Volatility (%)',
                        line=dict(color=self.color_palette['warning'], width=2)
                    ),
                    row=2, col=1
                )
            
            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown (%)',
                    line=dict(color=self.color_palette['danger'], width=2),
                    fill='tonexty'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Performance Metrics',
                template='plotly_white',
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance metrics chart: {str(e)}")
            return self.create_empty_chart("Error creating performance metrics")
