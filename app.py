import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback_context, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_cytoscape as cyto
import os
import json
import numpy as np
from datetime import datetime, timedelta
import base64
import io
# from scipy import stats  # Optional for advanced statistical analysis

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def load_data():
    """Load the pre-processed RFM clustered data from Delta table."""
    try:
        from databricks import sql
        from databricks.sdk.core import Config
        import os
        
        # Ensure environment variable is set correctly
        assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set as environment variable."
        
        cfg = Config()
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
            credentials_provider=lambda: cfg.authenticate
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM retail_analytics.ml.customer_rfm_description_and_recommendation")
                data = cursor.fetchall_arrow().to_pandas()
                
        print(f"Data loaded successfully: {data.shape[0]} customers, {len(data['Cluster'].unique())} clusters")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def load_lineage_data():
    """Load data lineage information from Delta table."""
    try:
        from databricks import sql
        from databricks.sdk.core import Config
        import os
        
        # Ensure environment variable is set correctly
        assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set as environment variable."
        
        cfg = Config()
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
            credentials_provider=lambda: cfg.authenticate
        ) as connection:
            with connection.cursor() as cursor:
                lineage_query = """
                SELECT source_type, source_path, source_table_full_name, target_type, target_table_full_name
                FROM retail_analytics.dataops.data_lineage 
                WHERE (source_table_full_name LIKE '%.dlt.%' OR source_table_full_name LIKE '%.ml.%'
                OR target_table_full_name LIKE '%.dlt.%' OR target_table_full_name LIKE '%.ml.%')
                AND source_type in ('STREAMING_TABLE', 'TABLE', 'VIEW', 'PATH')
                """
                cursor.execute(lineage_query)
                lineage_data = cursor.fetchall_arrow().to_pandas()
                
        print(f"Lineage data loaded successfully: {lineage_data.shape[0]} relationships")
        return lineage_data
    except Exception as e:
        print(f"Error loading lineage data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def load_quality_data():
    """Load data quality monitoring tables."""
    try:
        from databricks import sql
        from databricks.sdk.core import Config
        import os
        
        # Ensure environment variable is set correctly
        assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set as environment variable."
        
        cfg = Config()
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
            credentials_provider=lambda: cfg.authenticate
        ) as connection:
            with connection.cursor() as cursor:
                # Load bronze-silver comparison
                cursor.execute("SELECT * FROM retail_analytics.dlt.bronze_silver_dq_comparison ORDER BY processing_date DESC")
                dq_comparison = cursor.fetchall_arrow().to_pandas()
                
                # Load daily counts
                cursor.execute("SELECT * FROM retail_analytics.dlt.dlt_daily_counts ORDER BY invoicedate DESC")
                daily_counts = cursor.fetchall_arrow().to_pandas()
                
                # Load bronze quality metrics
                cursor.execute("SELECT * FROM retail_analytics.dlt.bronze_daily_quality ORDER BY processing_date DESC")
                bronze_quality = cursor.fetchall_arrow().to_pandas()
                
        print(f"Quality data loaded: {len(dq_comparison)} comparison records, {len(daily_counts)} daily counts, {len(bronze_quality)} bronze quality records")
        return dq_comparison, daily_counts, bronze_quality
    except Exception as e:
        print(f"Error loading quality data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Initialize global variables for data
data = None
lineage_data = None
dq_comparison = None
daily_counts = None
bronze_quality = None

def refresh_all_data():
    """Refresh all data from warehouse."""
    global data, lineage_data, dq_comparison, daily_counts, bronze_quality
    
    print("🔄 Refreshing data from warehouse...")
    data = load_data()
    lineage_data = load_lineage_data()
    dq_comparison, daily_counts, bronze_quality = load_quality_data()
    
    if data is None:
        print("Error: Could not load RFM data from Databricks. Please check your DATABRICKS_WAREHOUSE_ID environment variable.")
        exit()
    
    print("✅ Data refresh completed successfully")

# Load the data from Delta tables on startup
refresh_all_data()

def get_node_type_and_layer(node_name):
    """Determine node type and layer for positioning."""
    if pd.isna(node_name):
        return 'unknown', 8
    
    node_str = str(node_name)
    
    # S3 paths - layer 0
    if 's3://' in node_str:
        return 's3_path', 0
    
    # Quality and comparison tables - check FIRST to override bronze/silver/gold
    elif '.dlt.' in node_str and ('quality' in node_str.lower() or 'comparison' in node_str.lower()):
        return 'other_table', 4
    
    # DLT Bronze - layer 1
    elif '.dlt.' in node_str and 'bronze' in node_str.lower():
        return 'dlt_bronze', 1
    
    # DLT Silver - layer 2
    elif '.dlt.' in node_str and 'silver' in node_str.lower():
        return 'dlt_silver', 2
    
    # DLT Gold - layer 3
    elif '.dlt.' in node_str and 'gold' in node_str.lower():
        return 'dlt_gold', 3
    
    # Other DLT tables (daily counts, etc.) - layer 4
    elif '.dlt.' in node_str and 'daily' in node_str.lower():
        return 'other_table', 4
    
    # Other DLT tables (non-medallion) - layer 4
    elif '.dlt.' in node_str:
        return 'other_table', 4
    
    # ML Tables - spread across layers 5-7
    elif '.ml.' in node_str:
        # Basic clustering tables - layer 5
        if 'kmeans_clustered' in node_str.lower() or 'rfm_gold' in node_str.lower():
            return 'ml_table', 5
        # Description and recommendation tables - layer 6
        elif 'description' in node_str.lower() or 'recommendation' in node_str.lower():
            return 'ml_table', 6
        # Summary tables - layer 7
        elif 'summary' in node_str.lower():
            return 'ml_table', 7
        # Default ML tables - layer 5
        else:
            return 'ml_table', 5
    
    # Other tables (non-medallion) - layer 4
    elif '.' in node_str and not any(x in node_str.lower() for x in ['bronze', 'silver', 'gold']):
        return 'other_table', 4
    
    # Default - layer 8
    else:
        return 'default', 8

def get_display_name(node):
    """Create readable display name for nodes showing full schema.table names."""
    if pd.isna(node):
        return 'Unknown'
    
    node_str = str(node)
    
    if 's3://' in node_str:
        return 'S3 Raw Data\n(External Source)'
    elif '.' in node_str:
        parts = node_str.split('.')
        if len(parts) >= 3:
            catalog = parts[0]  # retail_analytics
            schema = parts[1]   # dlt, ml, etc.
            table = parts[2]    # actual table name
            
            # Show full schema.table name with catalog context
            return f"{schema}.{table}\n({catalog} catalog)"
        else:
            return node_str
    else:
        return node_str

def create_architecture_flow():
    """Create data architecture flow visualization using Cytoscape."""
    
    # Define the architecture flow nodes - simplified and generalized
    architecture_nodes = [
        # Data Ingestion
        {'id': 'data_ingestion', 'label': 'Data Ingestion\nAutomated Pipeline\nGitHub → S3 → Delta', 'type': 'source', 'layer': 0},
        
        # Medallion Architecture
        {'id': 'bronze_layer', 'label': '🥉 Bronze Layer\nRaw Data\nFull Fidelity', 'type': 'bronze', 'layer': 1},
        {'id': 'silver_layer', 'label': '🥈 Silver Layer\nCleaned Data\nBusiness Rules', 'type': 'silver', 'layer': 2},
        {'id': 'gold_layer', 'label': '🥇 Gold Layer\nAnalytics Ready\nRFM Metrics', 'type': 'gold', 'layer': 3},
        
        # Analytics & Applications
        {'id': 'ml_layer', 'label': 'ML Analytics\nCustomer Clustering\nSegmentation', 'type': 'ml', 'layer': 4},
        {'id': 'application_layer', 'label': 'Applications\nDashboard & Insights\nBusiness Intelligence', 'type': 'application', 'layer': 5},
        
        # Cross-cutting Concerns
        {'id': 'data_governance', 'label': 'Data Governance\nQuality & Monitoring\nObservability', 'type': 'quality', 'layer': 2}
    ]
    
    # Define the flow connections - simplified architectural flow
    architecture_edges = [
        # Main Data Flow
        {'source': 'data_ingestion', 'target': 'bronze_layer', 'label': 'Raw Data'},
        {'source': 'bronze_layer', 'target': 'silver_layer', 'label': 'Cleanse & Validate'},
        {'source': 'silver_layer', 'target': 'gold_layer', 'label': 'Aggregate & Transform'},
        {'source': 'gold_layer', 'target': 'ml_layer', 'label': 'Analytics'},
        {'source': 'ml_layer', 'target': 'application_layer', 'label': 'Insights'},
        
        # Governance Flow
        {'source': 'bronze_layer', 'target': 'data_governance', 'label': 'Monitor'},
        {'source': 'silver_layer', 'target': 'data_governance', 'label': 'Validate'},
        {'source': 'gold_layer', 'target': 'data_governance', 'label': 'Track'}
    ]
    
    # Create Cytoscape elements
    elements = []
    
    # Add nodes with positions for left-to-right flow
    for node in architecture_nodes:
        x_pos = node['layer'] * 300  # Optimized spacing for fewer nodes
        
        # Strategic Y positioning for simplified architecture
        if node['type'] in ['quality']:
            y_pos = 150  # Data governance below main flow
        else:
            y_pos = 0  # Main pipeline flow at center
            
        elements.append({
            'data': {
                'id': node['id'],
                'label': node['label'],
                'type': node['type'],
                'layer': node['layer']
            },
            'position': {'x': x_pos, 'y': y_pos}
        })
    
    # Add edges
    for edge in architecture_edges:
        elements.append({
            'data': {
                'id': f"{edge['source']}-{edge['target']}",
                'source': edge['source'],
                'target': edge['target'],
                'label': edge['label']
            }
        })
    
    # Define comprehensive stylesheet
    stylesheet = [
        # Base node style
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '12px',
                'font-weight': 'bold',
                'color': 'white',
                'text-wrap': 'wrap',
                'text-max-width': '140px',
                'width': '160px',
                'height': '80px',
                'shape': 'round-rectangle',
                'border-width': '3px',
                'border-color': 'white',
                'border-opacity': 0.9
            }
        },
        
        # Node types with medallion architecture and ML highlighted, others in grey shades
        {
            'selector': 'node[type = "source"]',
            'style': {'background-color': '#95A5A6'}  # Light grey for data sources
        },
        {
            'selector': 'node[type = "automation"]',
            'style': {'background-color': '#7F8C8D'}  # Medium grey for automation
        },
        {
            'selector': 'node[type = "storage"]',
            'style': {'background-color': '#BDC3C7'}  # Light grey for storage
        },
        {
            'selector': 'node[type = "ingestion"]',
            'style': {'background-color': '#85929E'}  # Medium grey for ingestion
        },
        {
            'selector': 'node[type = "bronze"]',
            'style': {'background-color': '#CD7F32', 'color': 'white'}  # Bronze - HIGHLIGHTED
        },
        {
            'selector': 'node[type = "silver"]',
            'style': {'background-color': '#C0C0C0', 'color': 'black'}  # Silver - HIGHLIGHTED
        },
        {
            'selector': 'node[type = "gold"]',
            'style': {'background-color': '#FFD700', 'color': 'black'}  # Gold - HIGHLIGHTED
        },
        {
            'selector': 'node[type = "ml"]',
            'style': {'background-color': '#9B59B6'}  # Purple for ML - HIGHLIGHTED
        },
        {
            'selector': 'node[type = "ml_output"]',
            'style': {'background-color': '#8E44AD'}  # Darker purple for ML output - HIGHLIGHTED
        },
        {
            'selector': 'node[type = "application"]',
            'style': {'background-color': '#AAB7B8'}  # Light grey for applications
        },
        {
            'selector': 'node[type = "quality"]',
            'style': {'background-color': '#909497'}  # Medium grey for quality
        },
        {
            'selector': 'node[type = "monitoring"]',
            'style': {'background-color': '#A6ACAF'}  # Light grey for monitoring
        },
        
        # Edge styles
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'straight',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#666',
                'line-color': '#666',
                'width': 3,
                'arrow-scale': 1.5,
                'font-size': '10px',
                'color': '#333',
                'text-background-color': 'white',
                'text-background-opacity': 0.8,
                'text-background-padding': '2px'
            }
        },
        
        # Hover effects
        {
            'selector': 'node:selected',
            'style': {
                'border-color': '#FF6B6B',
                'border-width': '5px'
            }
        }
    ]
    
    return html.Div([
        cyto.Cytoscape(
            id='architecture-flow-cytoscape',
            elements=elements,
            layout={
                'name': 'preset',
                'padding': 30,  # Reduced padding for mobile
                'animate': True,
                'animationDuration': 500
            },
            style={
                'width': '100%',
                'height': '500px',
                'border': '1px solid #ddd',
                'border-radius': '5px'
            },
            stylesheet=stylesheet,
            responsive=True,
            minZoom=0.1,  # Allow more zoom out for mobile
            maxZoom=3.0,  # Allow more zoom in for mobile
            wheelSensitivity=0.1
        ),
        # Mobile-friendly controls and info
        html.Div([
            dbc.Alert([
                html.I(className="fas fa-mobile-alt me-2"),
                html.Strong("Mobile Tip: "),
                "Pinch to zoom, drag to pan. Tap nodes for details."
            ], color="info", className="d-md-none mt-2 mb-0", style={"fontSize": "0.85rem"})
        ])
    ])

def create_cytoscape_lineage():
    """Create data lineage visualization using Cytoscape."""
    if lineage_data.empty:
        return html.Div([
            html.H4("No lineage data available", className="text-center text-muted"),
            html.P("Connect your data sources to see the pipeline visualization", className="text-center")
        ], style={'height': '400px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'})
    
    # Get all unique nodes
    all_nodes = set()
    edges = []
    
    for _, row in lineage_data.iterrows():
        source = row['source_table_full_name'] if pd.notna(row['source_table_full_name']) else row['source_path']
        target = row['target_table_full_name']
        
        if pd.notna(source):
            all_nodes.add(source)
        if pd.notna(target):
            all_nodes.add(target)
        
        if pd.notna(source) and pd.notna(target):
            edges.append({
                'source': str(source),
                'target': str(target),
                'source_type': row['source_type']
            })
    
    # Create Cytoscape elements with positioned layout
    elements = []
    
    # Calculate positions for left-to-right layout with no arrow overlap
    layer_nodes = {}
    for node in all_nodes:
        node_type, layer = get_node_type_and_layer(node)
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)
    
    # Add nodes with calculated positions
    for node in all_nodes:
        node_type, layer = get_node_type_and_layer(node)
        display_name = get_display_name(node)
        
        # Calculate position for left-to-right flow with better spacing
        x_pos = layer * 400  # Increased horizontal spacing to 400px
        nodes_in_layer = layer_nodes[layer]
        layer_index = nodes_in_layer.index(node)
        
        # Center nodes in layer and add extra spacing
        y_offset = (layer_index - (len(nodes_in_layer) - 1) / 2) * 180  # Increased vertical spacing to 180px
        y_pos = y_offset
        
        elements.append({
            'data': {
                'id': str(node),
                'label': display_name,
                'type': node_type,
                'layer': layer,
                'full_path': str(node)
            },
            'position': {'x': x_pos, 'y': y_pos}
        })
    
    # Add edges
    for edge in edges:
        elements.append({
            'data': {
                'id': f"{edge['source']}-{edge['target']}",
                'source': edge['source'],
                'target': edge['target'],
                'edge_type': edge['source_type'],
                'label': edge['source_type']
            }
        })
    
    # Define comprehensive stylesheet with larger fonts and table-like nodes
    stylesheet = [
        # Base node style - table-like appearance with bigger fonts
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '14px',  # Increased font size
                'font-weight': 'bold',
                'color': 'black',
                'text-wrap': 'wrap',
                'text-max-width': '140px',  # Increased width for longer names
                'width': '150px',    # Increased node width
                'height': '80px',    # Increased node height
                'shape': 'round-rectangle',  # Table-like appearance
                'border-width': '3px',  # Thicker border
                'border-color': 'white',
                'border-opacity': 0.9
            }
        },
        
        # S3 Sources
        {
            'selector': 'node[type = "s3_path"]',
            'style': {
                'background-color': '#F39C12',  # Orange
                'width': '160px',
                'height': '70px'
            }
        },
        
        # DLT Bronze
        {
            'selector': 'node[type = "dlt_bronze"]',
            'style': {
                'background-color': '#CD7F32'  # Bronze color
            }
        },
        
        # DLT Silver  
        {
            'selector': 'node[type = "dlt_silver"]',
            'style': {
                'background-color': '#C0C0C0'  # Silver color
            }
        },
        
        # DLT Gold
        {
            'selector': 'node[type = "dlt_gold"]',
            'style': {
                'background-color': '#FFD700',  # Gold color
                'color': 'black'  # Black text on gold
            }
        },
        
        # ML Tables - Purple as requested
        {
            'selector': 'node[type = "ml_table"]',
            'style': {
                'background-color': '#9B59B6'  # Purple
            }
        },
        
        # Other tables (non-medallion) - Light teal for better readability with black text
        {
            'selector': 'node[type = "other_table"]',
            'style': {
                'background-color': '#48C9B0'  # Light teal - good contrast with black text
            }
        },
        
        # Default tables
        {
            'selector': 'node[type = "default"]',
            'style': {
                'background-color': '#95A5A6'  # Gray
            }
        },
        
        # Base edge style
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'straight',  # Straight lines to avoid overlapping
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#666',
                'line-color': '#666',
                'width': 3,  # Thicker edges
                'arrow-scale': 1.5,
                'font-size': '12px',  # Larger edge labels
                'text-rotation': 'autorotate'
            }
        },
        
        # Edge types with better visibility
        {
            'selector': 'edge[edge_type = "PATH"]',
            'style': {
                'line-color': '#F39C12',
                'target-arrow-color': '#F39C12',
                'width': 4,
                'line-style': 'solid'
            }
        },
        {
            'selector': 'edge[edge_type = "STREAMING_TABLE"]',
            'style': {
                'line-color': '#E74C3C',
                'target-arrow-color': '#E74C3C',
                'width': 4,
                'line-style': 'dashed'
            }
        },
        {
            'selector': 'edge[edge_type = "TABLE"]',
            'style': {
                'line-color': '#3498DB',
                'target-arrow-color': '#3498DB',
                'width': 3
            }
        },
        {
            'selector': 'edge[edge_type = "VIEW"]',
            'style': {
                'line-color': '#2ECC71',
                'target-arrow-color': '#2ECC71',
                'width': 3,
                'line-style': 'dotted'
            }
        },
        
        # Enhanced hover effects
        {
            'selector': 'node:selected',
            'style': {
                'border-color': '#FF6B6B',
                'border-width': '5px',
                'background-color': '#FF6B6B'
            }
        },
        {
            'selector': 'edge:selected',
            'style': {
                'line-color': '#FF6B6B',
                'target-arrow-color': '#FF6B6B',
                'width': 5
            }
        }
    ]
    
    return cyto.Cytoscape(
        id='lineage-cytoscape',
        elements=elements,
        layout={
            'name': 'preset',  # Use preset positions for left-to-right flow
            'padding': 50,
            'animate': True,
            'animationDuration': 500
        },
        style={
            'width': '100%',
            'height': '800px',  # Increased height for better visibility
            'border': '1px solid #ddd',
            'border-radius': '5px'
        },
        stylesheet=stylesheet,
        responsive=True,
        minZoom=0.2,
        maxZoom=2.0,
        wheelSensitivity=0.1
    )

def create_record_count_trends():
    """Create record count trends chart."""
    if daily_counts.empty:
        return html.Div("No daily counts data available")
    
    # Get date range for title
    date_range = f"{daily_counts['invoicedate'].min()} to {daily_counts['invoicedate'].max()}"
    
    fig = go.Figure()
    
    # Filter for bronze and silver layers
    bronze_data = daily_counts[daily_counts['table_layer'].str.contains('bronze', case=False, na=False)]
    silver_data = daily_counts[daily_counts['table_layer'].str.contains('silver', case=False, na=False)]
    
    if not bronze_data.empty:
        fig.add_trace(go.Scatter(
            x=bronze_data['invoicedate'],
            y=bronze_data['record_counts'],
            mode='lines+markers',
            name='Bronze Layer',
            line=dict(color='#CD7F32', width=3),
            marker=dict(size=8)
        ))
    
    if not silver_data.empty:
        fig.add_trace(go.Scatter(
            x=silver_data['invoicedate'],
            y=silver_data['record_counts'],
            mode='lines+markers',
            name='Silver Layer',
            line=dict(color='#C0C0C0', width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f'Daily Record Counts: Bronze vs Silver ({date_range})',
        xaxis_title='Date',
        yaxis_title='Record Count',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_quality_score_trend():
    """Create quality score trend chart."""
    if bronze_quality.empty:
        return html.Div("No quality data available")
    
    # Get date range for title
    date_range = f"{bronze_quality['processing_date'].min()} to {bronze_quality['processing_date'].max()}"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=bronze_quality['processing_date'],
        y=bronze_quality['rescued_data_quality_score'],
        mode='lines+markers',
        name='Quality Score',
        line=dict(color='#2ECC71', width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor='rgba(46, 204, 113, 0.1)'
    ))
    
    fig.update_layout(
        title=f'Data Quality Score Trend ({date_range})',
        xaxis_title='Date',
        yaxis_title='Quality Score (%)',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_data_issues_breakdown():
    """Create data quality issues breakdown."""
    if bronze_quality.empty:
        return html.Div("No quality data available")
    
    # Get latest quality metrics
    latest_quality = bronze_quality.iloc[0] if not bronze_quality.empty else None
    
    if latest_quality is None:
        return html.Div("No quality data available")
    
    latest_date = latest_quality['processing_date']
    
    # Create pie chart of issues
    issues = {
        'Null Customer IDs': latest_quality.get('null_customer_ids', 0),
        'Bad Quantities': latest_quality.get('bad_quantities', 0),
        'Bad Prices': latest_quality.get('bad_price', 0),
        'Null Invoices': latest_quality.get('null_invoices', 0),
        'Null Stock Codes': latest_quality.get('null_stock_codes', 0)
    }
    
    # Filter out zero values
    non_zero_issues = {k: v for k, v in issues.items() if v > 0}
    
    if not non_zero_issues:
        return html.Div([
            html.H5("✅ No Data Quality Issues", className="text-success text-center"),
            html.P(f"All data quality checks passed for {latest_date}!", className="text-center")
        ])
    
    fig = px.pie(
        values=list(non_zero_issues.values()),
        names=list(non_zero_issues.keys()),
        title=f'Data Quality Issues Breakdown ({latest_date})'
    )
    
    fig.update_layout(height=400)
    return fig

def create_data_contract_content(tab_value):
    """Create data contract content based on selected tab."""
    
    if tab_value == "bronze-schema":
        # Bronze Layer Schema Table
        bronze_schema = [
            {"Field": "Invoice", "Type": "STRING", "Description": "Invoice number", "Constraints": "Raw format from source"},
            {"Field": "StockCode", "Type": "STRING", "Description": "Product code", "Constraints": "Original product identifier"},
            {"Field": "Description", "Type": "STRING", "Description": "Product description", "Constraints": "May contain nulls"},
            {"Field": "Quantity", "Type": "INTEGER", "Description": "Quantity purchased", "Constraints": "Raw values including negatives"},
            {"Field": "Price", "Type": "DECIMAL(10,2)", "Description": "Unit price", "Constraints": "Original pricing data"},
            {"Field": "CustomerID", "Type": "STRING", "Description": "Customer identifier", "Constraints": "May contain nulls"},
            {"Field": "Country", "Type": "STRING", "Description": "Customer country", "Constraints": "Geographic information"},
            {"Field": "InvoiceDate", "Type": "STRING", "Description": "Transaction date (raw)", "Constraints": "Unparsed date string"},
            {"Field": "ingestion_timestamp", "Type": "TIMESTAMP", "Description": "Pipeline processing time", "Constraints": "System generated"},
            {"Field": "source_file", "Type": "STRING", "Description": "Source file path", "Constraints": "Metadata for lineage"},
            {"Field": "processing_date", "Type": "DATE", "Description": "Processing date", "Constraints": "ETL batch date"}
        ]
        
        return html.Div([
            html.H5("🥉 Bronze Layer Schema (retail_transactions_bronze)", className="mb-3"),
            html.P("Raw data landing zone with full fidelity preservation from S3 sources.", className="text-muted mb-3"),
            dag.AgGrid(
                columnDefs=[
                    {"headerName": "Field Name", "field": "Field", "width": 180, "pinned": "left"},
                    {"headerName": "Data Type", "field": "Type", "width": 150},
                    {"headerName": "Description", "field": "Description", "width": 250},
                    {"headerName": "Constraints", "field": "Constraints", "width": 200}
                ],
                rowData=bronze_schema,
                defaultColDef={"sortable": True, "resizable": True},
                style={'height': '400px'}
            )
        ])
    
    elif tab_value == "silver-schema":
        # Silver Layer Schema Table
        silver_schema = [
            {"Field": "InvoiceNo", "Type": "STRING", "Description": "Cleaned invoice number", "Constraints": "NOT NULL, length 6-7 characters"},
            {"Field": "StockCode", "Type": "STRING", "Description": "Product code", "Constraints": "NOT NULL, validated format"},
            {"Field": "Description", "Type": "STRING", "Description": "Product description", "Constraints": "Nulls allowed"},
            {"Field": "Quantity", "Type": "INTEGER", "Description": "Quantity purchased", "Constraints": "NOT NULL, >0 for valid transactions"},
            {"Field": "UnitPrice", "Type": "DECIMAL(10,2)", "Description": "Unit price", "Constraints": ">=0, nulls allowed"},
            {"Field": "CustomerID", "Type": "STRING", "Description": "Customer identifier", "Constraints": "Nulls allowed, validated format"},
            {"Field": "Country", "Type": "STRING", "Description": "Customer country", "Constraints": "Standardized country names"},
            {"Field": "InvoiceDate", "Type": "DATE", "Description": "Parsed transaction date", "Constraints": "NOT NULL, valid date format"},
            {"Field": "IsCancellation", "Type": "BOOLEAN", "Description": "Cancellation flag", "Constraints": "TRUE if invoice starts with 'C'"},
            {"Field": "TotalPrice", "Type": "DECIMAL(12,2)", "Description": "Calculated total", "Constraints": "Quantity × UnitPrice"},
            {"Field": "Year", "Type": "INTEGER", "Description": "Extracted year", "Constraints": "From InvoiceDate"},
            {"Field": "Month", "Type": "INTEGER", "Description": "Extracted month", "Constraints": "1-12"},
            {"Field": "DayOfWeek", "Type": "INTEGER", "Description": "Day of week", "Constraints": "1-7 (Monday=1)"},
            {"Field": "SurrogateKey", "Type": "STRING", "Description": "Unique identifier", "Constraints": "InvoiceNo_StockCode_Quantity"},
            {"Field": "ingestion_timestamp", "Type": "TIMESTAMP", "Description": "From Bronze layer", "Constraints": "Inherited metadata"},
            {"Field": "source_file", "Type": "STRING", "Description": "From Bronze layer", "Constraints": "Lineage preservation"},
            {"Field": "processing_date", "Type": "DATE", "Description": "From Bronze layer", "Constraints": "ETL batch tracking"}
        ]
        
        return html.Div([
            html.H5("🥈 Silver Layer Schema (retail_transactions_silver)", className="mb-3"),
            html.P("Cleaned, validated, and enriched data ready for analytics with comprehensive business rules applied.", className="text-muted mb-3"),
            dag.AgGrid(
                columnDefs=[
                    {"headerName": "Field Name", "field": "Field", "width": 180, "pinned": "left"},
                    {"headerName": "Data Type", "field": "Type", "width": 150},
                    {"headerName": "Description", "field": "Description", "width": 250},
                    {"headerName": "Constraints", "field": "Constraints", "width": 300}
                ],
                rowData=silver_schema,
                defaultColDef={"sortable": True, "resizable": True},
                style={'height': '500px'}
            )
        ])
    
    elif tab_value == "gold-schema":
        # Gold Layer Schema Table
        gold_schema = [
            {"Field": "CustomerID", "Type": "STRING", "Description": "Customer identifier", "Constraints": "Primary key, NOT NULL"},
            {"Field": "MaxInvoiceDate", "Type": "DATE", "Description": "Last purchase date", "Constraints": "Most recent transaction date"},
            {"Field": "Recency", "Type": "INTEGER", "Description": "Days since last purchase", "Constraints": "Calculated from MaxInvoiceDate"},
            {"Field": "Frequency", "Type": "INTEGER", "Description": "Number of transactions", "Constraints": "Count of distinct invoices"},
            {"Field": "Monetary", "Type": "DECIMAL", "Description": "Total spend amount", "Constraints": "Sum of all TotalPrice values"}
        ]
        
        return html.Div([
            html.H5("🥇 Gold Layer Schema (customer_rfm_gold)", className="mb-3"),
            html.P("Business-ready aggregated metrics optimized for RFM analysis and machine learning applications.", className="text-muted mb-3"),
            dag.AgGrid(
                columnDefs=[
                    {"headerName": "Field Name", "field": "Field", "width": 180, "pinned": "left"},
                    {"headerName": "Data Type", "field": "Type", "width": 150},
                    {"headerName": "Description", "field": "Description", "width": 250},
                    {"headerName": "Constraints", "field": "Constraints", "width": 300}
                ],
                rowData=gold_schema,
                defaultColDef={"sortable": True, "resizable": True},
                style={'height': '250px'}
            ),
            html.Hr(),
            html.H6("📊 RFM Metrics Calculation", className="mb-2"),
            html.Ul([
                html.Li([html.Strong("Recency: "), "Days between customer's last purchase and analysis date"]),
                html.Li([html.Strong("Frequency: "), "Total number of unique transactions per customer"]),
                html.Li([html.Strong("Monetary: "), "Total revenue generated by customer across all transactions"])
            ])
        ])
    
    elif tab_value == "quality-rules":
        # Quality Rules Table
        quality_rules = [
            {"Layer": "Bronze", "Rule": "valid_file_ingestion", "Description": "Successful file ingestion from S3", "Action": "Log and alert on failure"},
            {"Layer": "Silver", "Rule": "valid_invoice_no", "Description": "Invoice length 6-7 characters, not null", "Action": "Reject invalid records"},
            {"Layer": "Silver", "Rule": "valid_stock_code", "Description": "Stock code must be present", "Action": "Reject null stock codes"},
            {"Layer": "Silver", "Rule": "valid_quantity", "Description": "Quantity > 0 and not null", "Action": "Filter out invalid quantities"},
            {"Layer": "Silver", "Rule": "valid_unit_price", "Description": "Unit price >= 0", "Action": "Allow nulls, reject negatives"},
            {"Layer": "Silver", "Rule": "valid_invoice_date", "Description": "Valid date format required", "Action": "Reject unparseable dates"},
            {"Layer": "Silver", "Rule": "customer_id_format", "Description": "Customer ID format validation", "Action": "Allow nulls for B2C transactions"},
            {"Layer": "Gold", "Rule": "customer_presence", "Description": "Customer ID required for RFM", "Action": "Exclude null customers from RFM"},
            {"Layer": "Gold", "Rule": "positive_monetary", "Description": "Total monetary value > 0", "Action": "Include only paying customers"},
            {"Layer": "Gold", "Rule": "valid_frequency", "Description": "Frequency >= 1", "Action": "At least one transaction required"}
        ]
        
        return html.Div([
            html.H5("✅ Data Quality Rules & Expectations", className="mb-3"),
            html.P("Comprehensive validation rules applied across all pipeline layers to ensure data integrity.", className="text-muted mb-3"),
            dag.AgGrid(
                columnDefs=[
                    {"headerName": "Layer", "field": "Layer", "width": 100},
                    {"headerName": "Rule Name", "field": "Rule", "width": 200},
                    {"headerName": "Description", "field": "Description", "width": 300},
                    {"headerName": "Action", "field": "Action", "width": 200}
                ],
                rowData=quality_rules,
                defaultColDef={"sortable": True, "resizable": True},
                style={'height': '400px'}
            ),
            html.Hr(),
            html.H6("📈 Quality Monitoring", className="mb-2"),
            html.P("Quality scores are calculated based on the percentage of records passing all validation rules.", className="small text-muted")
        ])
    
    elif tab_value == "data-filters":
        # Data Filters Table
        data_filters = [
            {"Stage": "Bronze → Silver", "Filter": "Cancellation Exclusion", "Logic": "Invoice NOT LIKE 'C%'", "Reason": "Remove cancelled transactions"},
            {"Stage": "Bronze → Silver", "Filter": "Stock Code Pattern", "Logic": "StockCode REGEXP '^[0-9]{5}$|^PADS$'", "Reason": "Valid product codes only"},
            {"Stage": "Bronze → Silver", "Filter": "Positive Quantities", "Logic": "Quantity > 0", "Reason": "Exclude returns and errors"},
            {"Stage": "Bronze → Silver", "Filter": "Valid Prices", "Logic": "UnitPrice >= 0", "Reason": "Non-negative pricing"},
            {"Stage": "Silver → Gold", "Filter": "Customer Presence", "Logic": "CustomerID IS NOT NULL", "Reason": "RFM requires customer identity"},
            {"Stage": "Silver → Gold", "Filter": "Transaction Validity", "Logic": "IsCancellation = FALSE", "Reason": "Valid purchases only"},
            {"Stage": "Gold → ML", "Filter": "Minimum Frequency", "Logic": "Frequency >= 1", "Reason": "At least one purchase required"},
            {"Stage": "Gold → ML", "Filter": "Positive Monetary", "Logic": "Monetary > 0", "Reason": "Revenue-generating customers"},
            {"Stage": "ML Pipeline", "Filter": "Outlier Detection", "Logic": "IQR-based filtering", "Reason": "Separate extreme customers"},
            {"Stage": "ML Pipeline", "Filter": "Cluster Validation", "Logic": "Valid cluster assignment", "Reason": "Successful segmentation"}
        ]
        
        return html.Div([
            html.H5("🔧 Data Filters & Business Rules", className="mb-3"),
            html.P("Applied filters and transformations to ensure clean, business-relevant data throughout the pipeline.", className="text-muted mb-3"),
            dag.AgGrid(
                columnDefs=[
                    {"headerName": "Pipeline Stage", "field": "Stage", "width": 150},
                    {"headerName": "Filter Name", "field": "Filter", "width": 180},
                    {"headerName": "Logic", "field": "Logic", "width": 250},
                    {"headerName": "Business Reason", "field": "Reason", "width": 200}
                ],
                rowData=data_filters,
                defaultColDef={"sortable": True, "resizable": True},
                style={'height': '400px'}
            ),
            html.Hr(),
            html.H6("🎯 Filter Impact", className="mb-2"),
            html.P("These filters ensure data quality while preserving business logic. Cancelled transactions and invalid records are excluded to maintain analytical accuracy.", className="small text-muted")
        ])
    
    return html.Div("Select a tab to view data contract details.")

def create_bronze_silver_comparison(selected_date=None):
    """Create bronze vs silver comparison table."""
    if dq_comparison.empty:
        return html.Div("No comparison data available")
    
    # Use selected date or default to latest
    if selected_date is None:
        selected_date = str(dq_comparison['processing_date'].max())
    
    # Convert processing_date to string for comparison if needed
    dq_comparison_str = dq_comparison.copy()
    dq_comparison_str['processing_date'] = dq_comparison_str['processing_date'].astype(str)
    
    comparison_data = dq_comparison_str[dq_comparison_str['processing_date'] == selected_date]
    
    if comparison_data.empty:
        return html.Div(f"No data found for {selected_date}")
    
    return dag.AgGrid(
        id='comparison-grid',
        columnDefs=[
            {"headerName": "Layer", "field": "table_layer", "width": 180},
            {"headerName": "Total Records", "field": "total_records", "width": 120, "type": "numericColumn"},
            {"headerName": "Null Invoice", "field": "null_invoice", "width": 100, "type": "numericColumn"},
            {"headerName": "Bad Invoice", "field": "bad_invoice", "width": 100, "type": "numericColumn"},
            {"headerName": "Null Stock Code", "field": "null_stock_code", "width": 120, "type": "numericColumn"},
            {"headerName": "Bad Quantity", "field": "bad_quantity", "width": 110, "type": "numericColumn"},
            {"headerName": "Null Customer ID", "field": "null_customer_id", "width": 130, "type": "numericColumn"}
        ],
        rowData=comparison_data.to_dict('records'),
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        style={'height': '200px'}
    )

def get_cluster_summary():
    """Generate cluster summary statistics."""
    cluster_summary = data.groupby(['Cluster', 'Cluster_Description']).agg({
        'CustomerID': 'count',
        'Monetary': ['mean', 'median', 'sum'],
        'Frequency': ['mean', 'median'],
        'Recency': ['mean', 'median']
    }).round(2)
    
    cluster_summary.columns = ['Count', 'Avg_Monetary', 'Med_Monetary', 'Total_Monetary',
                              'Avg_Frequency', 'Med_Frequency', 'Avg_Recency', 'Med_Recency']
    cluster_summary = cluster_summary.reset_index()
    
    return cluster_summary

def create_3d_scatter(selected_clusters=None):
    """Create 3D scatter plot of RFM clusters."""
    filtered_data = data[data['Cluster'].isin(selected_clusters)] if selected_clusters else data
    
    # Use centralized color mapping for consistency across all charts
    color_discrete_map = get_cluster_color_map()
    
    # Create cluster description mapping
    cluster_descriptions = data[['Cluster', 'Cluster_Description']].drop_duplicates().set_index('Cluster')['Cluster_Description']
    
    # Convert Cluster to string to ensure consistent color mapping
    filtered_data = filtered_data.copy()
    filtered_data['Cluster_str'] = filtered_data['Cluster'].astype(str)
    
    # Create string-based color map
    color_discrete_map_str = {str(k): v for k, v in color_discrete_map.items()}
    
    fig = px.scatter_3d(
        filtered_data, 
        x='Monetary', 
        y='Frequency', 
        z='Recency',
        color='Cluster_str',
        color_discrete_map=color_discrete_map_str,
        title='Customer RFM Segmentation - 3D View',
        labels={
            'Monetary': 'Monetary Value ($)',
            'Frequency': 'Purchase Frequency',
            'Recency': 'Days Since Last Purchase',
            'Cluster_str': 'Cluster'
        },
        hover_data=['CustomerID', 'Cluster_Description']
    )
    
    # Update legend names to show descriptions with error handling
    def update_trace_name(trace):
        try:
            cluster_id = int(trace.name)
            description = cluster_descriptions.get(cluster_id, 'Unknown')
            trace.update(name=f"Cluster {cluster_id}: {description}")
        except (ValueError, TypeError):
            # If trace name can't be converted to int, keep original name
            pass
    
    fig.for_each_trace(update_trace_name)
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Monetary Value ($)',
            yaxis_title='Purchase Frequency', 
            zaxis_title='Days Since Last Purchase'
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def create_cluster_distribution():
    """Create cluster distribution chart."""
    cluster_counts = data.groupby(['Cluster', 'Cluster_Description']).size().reset_index(name='Count')
    
    # Use centralized color mapping for consistency across all charts
    color_discrete_map = get_cluster_color_map()
    
    # Convert Cluster to string to ensure consistent color mapping
    cluster_counts = cluster_counts.copy()
    cluster_counts['Cluster_str'] = cluster_counts['Cluster'].astype(str)
    
    # Create string-based color map
    color_discrete_map_str = {str(k): v for k, v in color_discrete_map.items()}
    
    fig = px.bar(
        cluster_counts,
        x='Cluster_Description',
        y='Count',
        color='Cluster_str',
        color_discrete_map=color_discrete_map_str,
        title='Customer Distribution by Segment',
        labels={'Cluster_Description': 'Customer Segment', 'Count': 'Number of Customers', 'Cluster_str': 'Cluster'},
    )
    
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(tickangle=45)
    return fig

def create_monetary_analysis():
    """Create separate monetary value analysis plots for each segment."""
    import math
    
    # Use centralized color mapping for consistency across all charts
    color_discrete_map = get_cluster_color_map()
    
    # Get cluster descriptions if available
    cluster_descriptions = {}
    if 'Cluster_Description' in data.columns:
        cluster_descriptions = data[['Cluster', 'Cluster_Description']].drop_duplicates().set_index('Cluster')['Cluster_Description'].to_dict()
    
    # Calculate number of rows and columns for subplots
    n_clusters = len(unique_clusters)
    n_cols = min(3, n_clusters)  # Max 3 columns
    n_rows = math.ceil(n_clusters / n_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[cluster_descriptions.get(cluster, f'Cluster {cluster}') for cluster in unique_clusters],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Add box plot for each cluster
    for i, cluster in enumerate(unique_clusters):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        cluster_data = data[data['Cluster'] == cluster]['Monetary']
        
        # Add box plot
        fig.add_trace(
            go.Box(
                y=cluster_data,
                name=f'Cluster {cluster}',
                marker_color=color_discrete_map[cluster],
                showlegend=False,
                boxpoints='outliers',  # Show outlier points
                jitter=0.3,
                pointpos=-1.8
            ),
            row=row, col=col
        )
        
        # Add statistics annotation
        stats_text = f"Count: {len(cluster_data)}<br>" + \
                    f"Median: ${cluster_data.median():,.0f}<br>" + \
                    f"Mean: ${cluster_data.mean():,.0f}<br>" + \
                    f"Max: ${cluster_data.max():,.0f}"
        
        fig.add_annotation(
            text=stats_text,
            xref=f"x{i+1}", yref=f"y{i+1}",
            x=0.5, y=0.95,
            xanchor='center', yanchor='top',
            showarrow=False,
            font=dict(size=10, color='gray'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Monetary Value Distribution by Segment',
            font=dict(size=16, family="Segoe UI")
        ),
        height=200 * n_rows + 100,  # Dynamic height based on number of rows
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI", size=10)
    )
    
    # Update all y-axes to show currency format
    for i in range(1, n_clusters + 1):
        fig.update_yaxes(
            title_text="Monetary Value ($)" if i <= n_cols else "",  # Only show y-axis title on first row
            tickformat='$,.0f',
            row=(i-1)//n_cols + 1, 
            col=(i-1)%n_cols + 1
        )
    
    # Remove x-axis labels since we have subplot titles
    fig.update_xaxes(showticklabels=False)
    
    return fig

def create_frequency_analysis():
    """Create separate frequency analysis plots for each segment."""
    import math
    
    # Use centralized color mapping for consistency across all charts
    color_discrete_map = get_cluster_color_map()
    
    # Get cluster descriptions if available
    cluster_descriptions = {}
    if 'Cluster_Description' in data.columns:
        cluster_descriptions = data[['Cluster', 'Cluster_Description']].drop_duplicates().set_index('Cluster')['Cluster_Description'].to_dict()
    
    # Calculate number of rows and columns for subplots
    n_clusters = len(unique_clusters)
    n_cols = min(3, n_clusters)  # Max 3 columns
    n_rows = math.ceil(n_clusters / n_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[cluster_descriptions.get(cluster, f'Cluster {cluster}') for cluster in unique_clusters],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Add histogram for each cluster
    for i, cluster in enumerate(unique_clusters):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        cluster_data = data[data['Cluster'] == cluster]['Frequency']
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=cluster_data,
                name=f'Cluster {cluster}',
                marker_color=color_discrete_map[cluster],
                showlegend=False,
                opacity=0.7,
                nbinsx=min(20, len(cluster_data.unique()))  # Adaptive bin count
            ),
            row=row, col=col
        )
        
        # Add statistics annotation
        stats_text = f"Count: {len(cluster_data)}<br>" + \
                    f"Median: {cluster_data.median():.1f}<br>" + \
                    f"Mean: {cluster_data.mean():.1f}<br>" + \
                    f"Max: {cluster_data.max()}"
        
        fig.add_annotation(
            text=stats_text,
            xref=f"x{i+1}", yref=f"y{i+1}",
            x=0.95, y=0.95,
            xanchor='right', yanchor='top',
            showarrow=False,
            font=dict(size=10, color='gray'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Purchase Frequency Distribution by Segment',
            font=dict(size=16, family="Segoe UI")
        ),
        height=200 * n_rows + 100,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI", size=10)
    )
    
    # Update all axes
    for i in range(1, n_clusters + 1):
        fig.update_xaxes(
            title_text="Number of Orders" if i > (n_rows-1)*n_cols else "",  # Only show x-axis title on last row
            row=(i-1)//n_cols + 1, 
            col=(i-1)%n_cols + 1
        )
        fig.update_yaxes(
            title_text="Customer Count" if i <= n_cols else "",  # Only show y-axis title on first row
            row=(i-1)//n_cols + 1, 
            col=(i-1)%n_cols + 1
        )
    
    return fig

def create_recency_analysis():
    """Create separate recency analysis plots for each segment."""
    import math
    
    # Use centralized color mapping for consistency across all charts
    color_discrete_map = get_cluster_color_map()
    
    # Get cluster descriptions if available
    cluster_descriptions = {}
    if 'Cluster_Description' in data.columns:
        cluster_descriptions = data[['Cluster', 'Cluster_Description']].drop_duplicates().set_index('Cluster')['Cluster_Description'].to_dict()
    
    # Calculate number of rows and columns for subplots
    n_clusters = len(unique_clusters)
    n_cols = min(3, n_clusters)  # Max 3 columns
    n_rows = math.ceil(n_clusters / n_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[cluster_descriptions.get(cluster, f'Cluster {cluster}') for cluster in unique_clusters],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Add box plot for each cluster
    for i, cluster in enumerate(unique_clusters):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        cluster_data = data[data['Cluster'] == cluster]['Recency']
        
        # Add box plot
        fig.add_trace(
            go.Box(
                y=cluster_data,
                name=f'Cluster {cluster}',
                marker_color=color_discrete_map[cluster],
                showlegend=False,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=row, col=col
        )
        
        # Add statistics annotation
        stats_text = f"Count: {len(cluster_data)}<br>" + \
                    f"Median: {cluster_data.median():.0f} days<br>" + \
                    f"Mean: {cluster_data.mean():.0f} days<br>" + \
                    f"Min: {cluster_data.min()} days"
        
        fig.add_annotation(
            text=stats_text,
            xref=f"x{i+1}", yref=f"y{i+1}",
            x=0.5, y=0.95,
            xanchor='center', yanchor='top',
            showarrow=False,
            font=dict(size=10, color='gray'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Days Since Last Purchase by Segment',
            font=dict(size=16, family="Segoe UI")
        ),
        height=200 * n_rows + 100,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI", size=10)
    )
    
    # Update all y-axes
    for i in range(1, n_clusters + 1):
        fig.update_yaxes(
            title_text="Days Since Last Purchase" if i <= n_cols else "",
            row=(i-1)//n_cols + 1, 
            col=(i-1)%n_cols + 1
        )
    
    # Remove x-axis labels since we have subplot titles
    fig.update_xaxes(showticklabels=False)
    
    return fig

# Executive Dashboard page with key metrics and insights
def create_dashboard_page():
    if data is None:
        return html.Div([
            dbc.Alert("No data available. Please refresh to load data.", color="warning")
        ])
    

    # Helper function to safely convert to numeric
    def safe_numeric_operation(series, operation='mean'):
        """Safely perform numeric operations on series that might contain Decimal types."""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.isna().all():
                return 0
            
            if operation == 'mean':
                return float(numeric_series.mean())
            elif operation == 'median':
                return float(numeric_series.median())
            elif operation == 'min':
                return float(numeric_series.min())
            elif operation == 'max':
                return float(numeric_series.max())
            else:
                return float(numeric_series.mean())
        except Exception:
            return 0
    
    # Debug: Check available columns and segment info
    print(f"Available columns: {list(data.columns)}")
    
    # Calculate key metrics - use flexible column names
    total_customers = len(data)
    
    # Try different possible column names first
    cluster_col = None
    for col in ['Cluster', 'cluster', 'segment_id', 'cluster_id']:
        if col in data.columns:
            cluster_col = col
            break
    
    total_clusters = len(data[cluster_col].unique()) if cluster_col else 0
    
    # Try different possible monetary column names
    monetary_col = None
    for col in ['Monetary', 'monetary', 'total_spend', 'revenue', 'TotalSpend']:
        if col in data.columns:
            monetary_col = col
            break
    
    # Try different possible recency column names
    recency_col = None
    for col in ['Recency', 'recency', 'days_since_last_purchase']:
        if col in data.columns:
            recency_col = col
            break
    
    # Try different possible frequency column names
    frequency_col = None
    for col in ['Frequency', 'frequency', 'order_count', 'purchase_count']:
        if col in data.columns:
            frequency_col = col
            break
    
    # Debug segment information
    if 'Cluster_Description' in data.columns:
        print(f"Available Cluster Descriptions: {data['Cluster_Description'].unique()[:5]}")
    if 'Segment' in data.columns:
        print(f"Available Segments: {data['Segment'].unique()[:5]}")
    if 'Cluster' in data.columns:
        print(f"Available Clusters: {data['Cluster'].unique()[:5]}")
    
    # Debug revenue data
    if monetary_col:
        revenue_debug = data.groupby(data.columns[data.columns.str.contains('Cluster|Segment', case=False)][0] if any(data.columns.str.contains('Cluster|Segment', case=False)) else data.columns[0])[monetary_col].sum()
        print(f"Revenue by segment (debug): {revenue_debug.to_dict()}")
    
    # Use safe numeric operations to calculate averages
    avg_monetary = safe_numeric_operation(data[monetary_col]) if monetary_col else 0
    avg_recency = safe_numeric_operation(data[recency_col]) if recency_col else 0
    avg_frequency = safe_numeric_operation(data[frequency_col]) if frequency_col else 0
    
    # Try different possible segment column names
    segment_col = None
    for col in ['Segment', 'segment', 'customer_segment', 'segment_name', 'SegmentName']:
        if col in data.columns:
            segment_col = col
            break
    
    # Use actual segment names from data source, prioritize description columns
    display_col = None
    group_col = None
    
    # Check for description columns first (these contain business-friendly names)
    for desc_col in ['Cluster_Description', 'Segment_Description', 'SegmentDescription']:
        if desc_col in data.columns:
            display_col = desc_col
            # Find the corresponding ID column
            if 'Cluster_Description' in desc_col and 'Cluster' in data.columns:
                group_col = 'Cluster'
            elif 'Segment' in desc_col and segment_col:
                group_col = segment_col
            break
    
    # If no description column, use the raw segment/cluster column
    if not display_col:
        if segment_col:
            display_col = segment_col
            group_col = segment_col
        elif cluster_col:
            display_col = cluster_col
            group_col = cluster_col
    
    
    if display_col and group_col:
        # Get ALL segments (not just top 5) for consistent display
        if display_col != group_col:
            # Use description column for display, but group by ID for accuracy
            segment_map = data[[group_col, display_col]].drop_duplicates().set_index(group_col)[display_col]
            raw_counts = data[group_col].value_counts()  # Get ALL segments
            all_segments_counts = pd.Series({
                segment_map.get(cluster, str(cluster)): count 
                for cluster, count in raw_counts.items()
            })
            # Get top 8 for display consistency between charts
            top_segments = all_segments_counts.head(8)
        else:
            all_segments_counts = data[display_col].value_counts()  # Get ALL segments
            top_segments = all_segments_counts.head(8)
        
        # Revenue by segment using actual segment names with stacking for same descriptions
        if monetary_col:
            if display_col != group_col:
                # Create detailed revenue data for stacked chart
                revenue_data = data.groupby([group_col, display_col])[monetary_col].sum().reset_index()
                revenue_data['segment_name'] = revenue_data[display_col]
                revenue_data['cluster_id'] = revenue_data[group_col].astype(str)
                
                # Get aggregated revenue by segment description for ordering
                segment_totals = revenue_data.groupby('segment_name')[monetary_col].sum().sort_values(ascending=False)
                
                # Use top 8 segments by customer count, but show their revenue breakdown
                selected_segments = top_segments.index[:8]
                revenue_chart_data = revenue_data[revenue_data['segment_name'].isin(selected_segments)]
                
                # Use centralized color mapping for consistency
                cluster_color_map = get_cluster_color_map()
                segment_color_map = get_segment_color_map()
            else:
                # If display_col same as group_col, no stacking needed
                revenue_data = data.groupby(display_col)[monetary_col].sum().reset_index()
                revenue_data['segment_name'] = revenue_data[display_col]
                revenue_data['cluster_id'] = revenue_data[display_col].astype(str)
                
                selected_segments = top_segments.index[:8]
                revenue_chart_data = revenue_data[revenue_data['segment_name'].isin(selected_segments)]
                
                # Use centralized color mapping for consistency
                cluster_color_map = get_cluster_color_map()
                segment_color_map = get_segment_color_map()
        else:
            revenue_chart_data = pd.DataFrame()
            cluster_color_map = {}
            segment_color_map = {}
            selected_segments = []
    
    # Debug output for chart consistency
    if len(top_segments) > 0 and 'revenue_chart_data' in locals() and not revenue_chart_data.empty:
        print(f"Pie chart segments: {list(top_segments.index)}")
        print(f"Revenue chart segments: {list(revenue_chart_data['segment_name'].unique())}")
        missing_in_revenue = set(top_segments.index) - set(revenue_chart_data['segment_name'].unique())
        if missing_in_revenue:
            print(f"⚠️  Segments missing from revenue chart: {missing_in_revenue}")
        # Show stacking info
        stacking_info = revenue_chart_data.groupby('segment_name')['cluster_id'].count()
        segments_with_multiple_clusters = stacking_info[stacking_info > 1]
        if len(segments_with_multiple_clusters) > 0:
            print(f"📊 Segments with multiple clusters (stacked): {segments_with_multiple_clusters.to_dict()}")
    else:
        top_segments = pd.Series(dtype=int)
        revenue_chart_data = pd.DataFrame()
        cluster_color_map = {}
        segment_color_map = {}
        selected_segments = []
    
    return [
        # Page Header
        html.Div([
            html.H1("Executive Dashboard", className="page-title"),
            html.P("Key metrics and insights for strategic decision making", className="page-subtitle")
        ], className="page-header"),
        
        # Key Metrics Cards
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-users metric-icon", style={"color": "#20b2aa"}),
                        html.H2(f"{total_customers:,}", className="metric-value"),
                        html.P("Total Customers", className="metric-label")
                    ])
                ], className="metric-card")
            ], xl=3, lg=6, md=6, sm=12, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-layer-group metric-icon", style={"color": "#5f9ea0"}),
                        html.H2(str(total_clusters), className="metric-value"),
                        html.P("Customer Segments", className="metric-label")
                    ])
                ], className="metric-card")
            ], xl=3, lg=6, md=6, sm=12, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-dollar-sign metric-icon", style={"color": "#4682b4"}),
                        html.H2(f"${avg_monetary:,.0f}", className="metric-value"),
                        html.P("Avg. Customer Value", className="metric-label")
                    ])
                ], className="metric-card")
            ], xl=3, lg=6, md=6, sm=12, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-clock metric-icon", style={"color": "#008b8b"}),
                        html.H2(f"{avg_recency:.0f}", className="metric-value"),
                        html.P("Avg. Days Since Purchase", className="metric-label")
                    ])
                ], className="metric-card")
            ], xl=3, lg=6, md=6, sm=12, className="mb-4")
        ]),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Customer Base Composition", className="mb-3"),
                    html.P("Understanding your customer segments for targeted strategies", className="text-muted mb-3"),
                    dcc.Graph(
                        figure=px.pie(
                            values=top_segments.values if len(top_segments) > 0 else [1],
                            names=top_segments.index if len(top_segments) > 0 else ['No Data'],
                            title="Customer Distribution by Business Segment",
                            color=top_segments.index if len(top_segments) > 0 else ['No Data'],
                            color_discrete_map=segment_color_map if len(top_segments) > 0 else {}
                        ).update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Segoe UI", size=12),
                            showlegend=True,
                            legend=dict(orientation="v", yanchor="middle", y=0.5)
                        )
                    ) if len(top_segments) > 0 else dcc.Graph(
                        figure=go.Figure().add_annotation(
                            text="No segment data available",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False
                        ).update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                    )
                ], className="chart-container")
            ], xl=6, lg=12, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H4("Revenue Performance by Segment", className="mb-3"),
                    html.P("Identify your most valuable customer groups", className="text-muted mb-3"),
                    dcc.Graph(
                        figure=px.bar(
                            revenue_chart_data if 'revenue_chart_data' in locals() and not revenue_chart_data.empty else pd.DataFrame(),
                            x='segment_name',
                            y=monetary_col,
                            color='segment_name',
                            title="Revenue Generation by Customer Segment",
                            color_discrete_map=segment_color_map if 'segment_color_map' in locals() else {},
                            category_orders={'segment_name': list(selected_segments) if 'selected_segments' in locals() else []}
                        ).update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Segoe UI", size=12),
                            xaxis_title="Customer Segment",
                            yaxis_title="Total Revenue ($)",
                            legend_title="Cluster ID",
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            )
                        ).update_xaxes(tickangle=45).update_traces(
                            hovertemplate='<b>%{x}</b><br>' +
                                        'Cluster: %{fullData.name}<br>' +
                                        'Revenue: $%{y:,.0f}<extra></extra>'
                        ) if 'revenue_chart_data' in locals() and not revenue_chart_data.empty else go.Figure().add_annotation(
                            text="No revenue data available",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False
                        ).update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                    )
                ], className="chart-container")
            ], xl=6, lg=12, className="mb-4")
        ]),
        
        # Business Insights and Recommendations
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Business Insights & Recommendations", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5([html.I(className="fas fa-clock me-2", style={"color": "#20b2aa"}), "Customer Engagement"], className="mb-3"),
                                html.P(f"Average time since last purchase: {avg_recency:.1f} days" if recency_col else "N/A", className="mb-2"),
                                html.P([
                                    html.Strong("Recommendation: "),
                                    "Focus on win-back campaigns for customers inactive >30 days" if recency_col and avg_recency > 30 else "Maintain engagement with recent active customers"
                                ], className="small text-muted")
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.H5([html.I(className="fas fa-shopping-cart me-2", style={"color": "#5f9ea0"}), "Purchase Behavior"], className="mb-3"),
                                html.P(f"Average orders per customer: {avg_frequency:.1f}" if frequency_col else "N/A", className="mb-2"),
                                html.P([
                                    html.Strong("Top Opportunity: "),
                                    f"Encourage repeat purchases - {((data[frequency_col] == 1).sum() / len(data) * 100):.1f}% are one-time buyers" if frequency_col and not data[frequency_col].isna().all() else "Analyze purchase patterns"
                                ], className="small text-muted")
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.H5([html.I(className="fas fa-dollar-sign me-2", style={"color": "#4682b4"}), "Revenue Potential"], className="mb-3"),
                                html.P(f"Average customer value: ${avg_monetary:,.0f}" if monetary_col else "N/A", className="mb-2"),
                                html.P([
                                    html.Strong("Strategy: "),
                                    f"Focus on high-value segments - top 20% customers represent significant revenue opportunity" if monetary_col and not data[monetary_col].isna().all() else "Identify high-value customers"
                                ], className="small text-muted")
                            ])
                        ], md=4)
                    ])
                ], className="chart-container")
            ], width=12)
        ]),
        
        # Top Segment Insights
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Key Customer Segments", className="mb-3"),
                    html.Div([
                        html.Div([
                            html.H6(f"🏆 {str(top_segments.index[0])[:30]}..." if len(top_segments) > 0 and len(str(top_segments.index[0])) > 30 else f"🏆 {top_segments.index[0]}" if len(top_segments) > 0 else "🏆 N/A", className="mb-2"),
                            html.P(f"{top_segments.iloc[0]:,} customers" if len(top_segments) > 0 else "N/A", className="mb-1"),
                            html.Small("Largest segment - prioritize retention", className="text-muted")
                        ], className="p-3 border rounded me-3 mb-3", style={"min-width": "200px", "background-color": "#f8f9fa"}),
                        
                        html.Div([
                            html.H6(
                                f"💰 {str(revenue_chart_data.groupby('segment_name')[monetary_col].sum().sort_values(ascending=False).index[0])[:30]}..." 
                                if 'revenue_chart_data' in locals() and not revenue_chart_data.empty and len(str(revenue_chart_data.groupby('segment_name')[monetary_col].sum().sort_values(ascending=False).index[0])) > 30 
                                else f"💰 {revenue_chart_data.groupby('segment_name')[monetary_col].sum().sort_values(ascending=False).index[0]}" 
                                if 'revenue_chart_data' in locals() and not revenue_chart_data.empty 
                                else "💰 N/A", 
                                className="mb-2"
                            ),
                            html.P(
                                f"${revenue_chart_data.groupby('segment_name')[monetary_col].sum().sort_values(ascending=False).iloc[0]:,.0f} revenue" 
                                if 'revenue_chart_data' in locals() and not revenue_chart_data.empty 
                                else "N/A", 
                                className="mb-1"
                            ),
                            html.Small("Highest revenue - protect at all costs", className="text-muted")
                        ], className="p-3 border rounded me-3 mb-3", style={"min-width": "200px", "background-color": "#e0f7fa"}),
                        
                        html.Div([
                            html.H6("📈 Growth Opportunity", className="mb-2"),
                            html.P(
                                f"{((pd.to_numeric(data[recency_col], errors='coerce') > safe_numeric_operation(data[recency_col], 'median')) & (pd.to_numeric(data[monetary_col], errors='coerce') > safe_numeric_operation(data[monetary_col], 'median'))).sum():,} customers" 
                                if recency_col and monetary_col 
                                else "N/A", 
                                className="mb-1"
                            ),
                            html.Small("High-value but dormant - reactivation target", className="text-muted")
                        ], className="p-3 border rounded mb-3", style={"min-width": "200px", "background-color": "#e0f2f1"})
                    ], className="d-flex flex-wrap")
                ], className="chart-container")
            ], width=12)
        ]),
        
    ]

# Enhanced segments page with detailed segment analysis
def create_segments_page():
    if data is None:
        return html.Div([
            dbc.Alert("No data available. Please refresh to load data.", color="warning")
        ])
    
    # Find available columns dynamically
    segment_col = None
    for col in ['Segment', 'segment', 'customer_segment', 'segment_name', 'SegmentName', 'Cluster', 'cluster']:
        if col in data.columns:
            segment_col = col
            break
    
    customer_col = None
    for col in ['CustomerID', 'customer_id', 'CustomerId', 'ID', 'id']:
        if col in data.columns:
            customer_col = col
            break
    
    monetary_col = None
    for col in ['Monetary', 'monetary', 'total_spend', 'revenue', 'TotalSpend']:
        if col in data.columns:
            monetary_col = col
            break
    
    frequency_col = None
    for col in ['Frequency', 'frequency', 'order_count', 'purchase_count']:
        if col in data.columns:
            frequency_col = col
            break
    
    recency_col = None
    for col in ['Recency', 'recency', 'days_since_last_purchase']:
        if col in data.columns:
            recency_col = col
            break
    
    if not segment_col:
        return html.Div([
            dbc.Alert("No segment/cluster column found in data. Available columns: " + ", ".join(data.columns), color="warning")
        ])
    
    # Segment analysis with available columns
    agg_dict = {}
    if customer_col:
        agg_dict[customer_col] = 'count'
    if monetary_col:
        agg_dict[monetary_col] = ['mean', 'sum', 'std']
    if frequency_col:
        agg_dict[frequency_col] = ['mean', 'std']
    if recency_col:
        agg_dict[recency_col] = ['mean', 'std']
    
    if not agg_dict:
        return html.Div([
            dbc.Alert("No analyzable columns found in data.", color="warning")
        ])
    
    segment_analysis = data.groupby(segment_col).agg(agg_dict).round(2)
    
    # Flatten column names
    new_columns = []
    for col in segment_analysis.columns:
        if isinstance(col, tuple):
            if col[1] == 'count':
                new_columns.append('Customer_Count')
            elif col[1] == 'mean' and 'monetary' in col[0].lower():
                new_columns.append('Avg_Monetary')
            elif col[1] == 'sum' and 'monetary' in col[0].lower():
                new_columns.append('Total_Revenue')
            elif col[1] == 'std' and 'monetary' in col[0].lower():
                new_columns.append('Monetary_Std')
            elif col[1] == 'mean' and 'frequency' in col[0].lower():
                new_columns.append('Avg_Frequency')
            elif col[1] == 'std' and 'frequency' in col[0].lower():
                new_columns.append('Frequency_Std')
            elif col[1] == 'mean' and 'recency' in col[0].lower():
                new_columns.append('Avg_Recency')
            elif col[1] == 'std' and 'recency' in col[0].lower():
                new_columns.append('Recency_Std')
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(str(col))
    
    segment_analysis.columns = new_columns
    segment_analysis = segment_analysis.reset_index()
    
    # Add cluster description and recommendation if available
    description_cols = ['Cluster_Description', 'SegmentDescription', 'Description']
    recommendation_cols = ['Cluster_Recommendation', 'SegmentRecommendation', 'Recommendation', 'Strategy']
    
    # Find description column
    desc_col = None
    for col in description_cols:
        if col in data.columns:
            desc_col = col
            break
    
    # Find recommendation column  
    rec_col = None
    for col in recommendation_cols:
        if col in data.columns:
            rec_col = col
            break
    
    # Add description and recommendation to the analysis
    if desc_col or rec_col:
        # Get unique segment descriptions and recommendations
        segment_info = data[[segment_col] + [col for col in [desc_col, rec_col] if col]].drop_duplicates()
        segment_info = segment_info.set_index(segment_col)
        
        # Merge with segment analysis
        if desc_col:
            segment_analysis['Description'] = segment_analysis[segment_col].map(segment_info[desc_col])
        if rec_col:
            segment_analysis['Recommendation'] = segment_analysis[segment_col].map(segment_info[rec_col])
    
    return [
        # Page Header
        html.Div([
            html.H1("Customer Segments Analysis", className="page-title"),
            html.P("Detailed analysis of customer segments and their characteristics", className="page-subtitle")
        ], className="page-header"),
        
        # Segment Filters
        html.Div([
            html.H5("Segment Filters", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Segments:", className="form-label"),
                    dcc.Dropdown(
                        id="segment-filter",
                        options=[],  # Will be populated by callback
                        value=[],    # Will be populated by callback
                        multi=True,
                        placeholder="Select segments to analyze...",
                        className="mb-3"
                    ) if segment_col else html.P("No segment data available")
                ], md=6),
                dbc.Col([
                    html.Label("Minimum Customer Count:", className="form-label"),
                    dcc.Slider(
                        id="min-customers-slider",
                        min=1,
                        max=data[segment_col].value_counts().max() if segment_col else 100,
                        value=1,
                        marks={i: str(i) for i in range(0, (data[segment_col].value_counts().max() if segment_col else 100)+1, max(1, (data[segment_col].value_counts().max() if segment_col else 100)//10))},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=6)
            ])
        ], className="filter-panel"),
        
        # Segment Comparison Table with Descriptions and Recommendations
        html.Div([
            html.Div([
                html.H4("Detailed Segment Analysis", className="mb-3", style={"display": "inline-block"}),
                html.Div(id="filter-status", className="ms-3", style={"display": "inline-block", "color": "#6c757d"})
            ], className="d-flex align-items-center mb-3"),
            html.Div([
                html.P("Comprehensive analysis including customer metrics", className="text-muted mb-1"),
                html.Div([
                    html.Span([html.I(className="fas fa-info-circle me-1", style={"color": "#20b2aa"}), "Descriptions"], className="badge bg-light text-dark me-2") if any(col in data.columns for col in ['Cluster_Description', 'SegmentDescription', 'Description']) else "",
                    html.Span([html.I(className="fas fa-lightbulb me-1", style={"color": "#5f9ea0"}), "Recommendations"], className="badge bg-light text-dark me-2") if any(col in data.columns for col in ['Cluster_Recommendation', 'SegmentRecommendation', 'Recommendation', 'Strategy']) else "",
                    html.Span([html.I(className="fas fa-chart-bar me-1", style={"color": "#4682b4"}), "Metrics"], className="badge bg-light text-dark")
                ], className="mb-3")
            ]),
            dag.AgGrid(
                id="segment-comparison-grid",
                columnDefs=[
                    {"field": segment_col, "headerName": "Segment", "pinned": "left", "width": 150},
                    {"field": "Customer_Count", "headerName": "Customers", "type": "numericColumn", "width": 120},
                    {"field": "Avg_Monetary", "headerName": "Avg Revenue", "type": "numericColumn", "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"}, "width": 130},
                    {"field": "Total_Revenue", "headerName": "Total Revenue", "type": "numericColumn", "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"}, "width": 140},
                    {"field": "Avg_Frequency", "headerName": "Avg Orders", "type": "numericColumn", "width": 120},
                    {"field": "Avg_Recency", "headerName": "Days Since Purchase", "type": "numericColumn", "width": 150},
                ] + ([{"field": "Description", "headerName": "Description", "width": 300, "wrapText": True, "autoHeight": True}] if 'Description' in segment_analysis.columns else []) + ([{"field": "Recommendation", "headerName": "Strategy Recommendation", "width": 400, "wrapText": True, "autoHeight": True}] if 'Recommendation' in segment_analysis.columns else []),
                rowData=segment_analysis.to_dict('records') if not segment_analysis.empty else [],
                defaultColDef={"sortable": True, "filter": True, "resizable": True, "wrapText": True},
                dashGridOptions={
                    "pagination": True, 
                    "paginationPageSize": 8,
                    "domLayout": "autoHeight",
                    "suppressHorizontalScroll": False,
                    "columnTypes": {
                        "textColumn": {"wrapText": True, "autoHeight": True}
                    }
                },
                className="ag-theme-alpine",
                style={"height": "500px", "width": "100%"}
            )
        ], className="chart-container")
    ]

# About page with project information
def create_about_page():
    return [
        # Page Header
        html.Div([
            html.H1("About This Project", className="page-title"),
            html.P("Learn about the data engineering pipeline and RFM analysis methodology", className="page-subtitle")
        ], className="page-header"),
        
        # Project Overview
        html.Div([
            html.H4("Project Overview", className="mb-3"),
            html.P("This is a comprehensive data engineering project that demonstrates the complete data lifecycle from raw data ingestion to machine learning application deployment. The project implements customer segmentation using RFM analysis and K-means clustering for retail transaction data."),
            
            html.H5("Key Features", className="mt-4 mb-3"),
            html.Ul([
                html.Li("End-to-end data pipeline with automated daily processing"),
                html.Li("Medallion architecture (Bronze, Silver, Gold) using Delta Lake"),
                html.Li("RFM customer segmentation with intelligent clustering"),
                html.Li("Real-time data quality monitoring"),
                html.Li("Interactive analytics dashboard"),
                html.Li("Comprehensive data lineage tracking")
            ]),
            
            html.H5("Technology Stack", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Data Platform", className="text-muted"),
                        html.P("Databricks, Delta Lake, Apache Spark")
                    ])
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Storage", className="text-muted"),
                        html.P("AWS S3, Delta Tables")
                    ])
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Analytics", className="text-muted"),
                        html.P("Python, Pandas, Plotly, Dash")
                    ])
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.H6("ML & Clustering", className="text-muted"),
                        html.P("scikit-learn, K-means")
                    ])
                ], md=3)
            ])
        ], className="chart-container")
    ]

# Landing page layout (redirects to about page)
def create_landing_page():
    return create_about_page()
    return [
        # Hero Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("End-to-End Data Engineering Pipeline", className="display-4 text-center mb-3"),
                    html.H2("Retail Customer Analytics", className="text-center text-muted mb-4"),
                    html.P("A comprehensive data engineering project demonstrating the complete data lifecycle from raw data ingestion to machine learning application deployment", 
                           className="lead text-center mb-4"),
                    # Latest update indicator
                    dbc.Alert([
                        html.I(className="fas fa-database me-2"),
                        html.Strong("Live Pipeline Status: "),
                        f"Latest batch processed {data.iloc[0]['UpdateDate'] if data is not None and not data.empty and 'UpdateDate' in data.columns else 'N/A'}",
                        html.Span(" | ", className="mx-2"),
                        f"{len(data):,} customers analyzed" if data is not None and not data.empty else "Data loading...",
                        html.Br(),
                        html.Small([
                            html.I(className="fas fa-clock me-1"),
                            "Automated daily ETL pipeline • Real-time Delta Live Tables processing"
                        ], className="text-muted")
                    ], color="info", className="text-center"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P("💼 Created by", className="text-center text-muted mb-1"),
                            html.H4("Jonathan Musni", className="text-center mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button([
                                        html.I(className="fab fa-github me-2"),
                                        "GitHub"
                                    ], href="https://github.com/jemusni07", target="_blank", color="dark", outline=True, className="me-2 mb-2")
                                ], width="auto"),
                                dbc.Col([
                                    dbc.Button([
                                        html.I(className="fab fa-linkedin me-2"),
                                        "LinkedIn"
                                    ], href="https://www.linkedin.com/in/musni-jonathan/", target="_blank", color="primary", outline=True, className="me-2 mb-2")
                                ], width="auto"),
                                dbc.Col([
                                    dbc.Button([
                                        html.I(className="fab fa-medium me-2"),
                                        "Medium"
                                    ], href="https://medium.com/@musni.jonathan7", target="_blank", color="success", outline=True, className="me-2 mb-2")
                                ], width="auto"),
                                dbc.Col([
                                    dbc.Button([
                                        html.I(className="fas fa-globe me-2"),
                                        "Website"
                                    ], href="https://jonathanmusni.com/", target="_blank", color="info", outline=True, className="mb-2")
                                ], width="auto")
                            ], justify="center")
                        ], width=12)
                    ])
                ], className="text-center py-5", style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'color': 'white', 'border-radius': '10px'})
            ], width=12)
        ], className="mb-5"),
        
        # Project Purpose Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H3("🎯 Project Purpose & Goals", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.H5("Create a Low-Cost, Production-Grade Data Engineering Solution", className="text-primary mb-3"),
                        dbc.Alert([
                            html.I(className="fas fa-dollar-sign me-2"),
                            html.Strong("Cost Breakdown: "),
                            "This entire production pipeline runs on ",
                            html.Strong("Databricks Community Edition (FREE)"),
                            " with only minimal AWS S3 storage costs (~$1-2/month). Demonstrates enterprise-grade capabilities at near-zero cost!"
                        ], color="success", className="mb-3"),
                        html.P("This project demonstrates how to build a comprehensive data engineering pipeline that is:", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Ul([
                                    html.Li([html.Strong("Production-Ready: "), "Implements industry best practices with proper data quality, monitoring, and error handling"]),
                                    html.Li([html.Strong("Cost-Effective: "), "Built entirely on Databricks Community Edition (FREE) + minimal AWS S3 costs"]),
                                    html.Li([html.Strong("Scalable: "), "Built on Databricks and AWS infrastructure that can handle growing data volumes"]),
                                    html.Li([html.Strong("Real-Time: "), "Daily automated updates simulate production batch processing workflows"])
                                ])
                            ], width=6),
                            dbc.Col([
                                html.Ul([
                                    html.Li([html.Strong("End-to-End: "), "Complete pipeline from raw data to deployed web application"]),
                                    html.Li([html.Strong("ML-Powered: "), "Advanced customer segmentation using unsupervised learning techniques"]),
                                    html.Li([html.Strong("Observable: "), "Comprehensive monitoring, lineage tracking, and quality assurance"]),
                                    html.Li([html.Strong("Business-Focused: "), "Delivers actionable customer insights with marketing recommendations"])
                                ])
                            ], width=6)
                        ]),
                        html.Hr(),
                        html.Div([
                            html.H6("📦 Project Repository", className="mb-2"),
                            dbc.Button([
                                html.I(className="fab fa-github me-2"),
                                "View Complete Project on GitHub"
                            ], href="https://github.com/jemusni07/retail-pipeline", target="_blank", color="dark", size="lg")
                        ], className="text-center")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Architecture Overview Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H3("🏗️ End-to-End Architecture Overview", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Complete data flow from source to application, demonstrating modern data engineering practices:", className="mb-4"),
                        
                        # Data Flow Steps
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("📊 Data Source", className="text-primary mb-2"),
                                        html.P([html.Strong("UCI Repository"), " → Historical retail transactions (2009-2011)"]),
                                        html.P([html.Strong("GitHub Actions"), " → Automated daily file uploads"]),
                                        html.P([html.Strong("AWS S3"), " → Raw data lake storage"]),
                                        html.Hr(),
                                        html.P([html.Em("Note: Historical dates shifted to current period (2025-2026) to simulate real-time daily retail transactions.")], className="text-muted small")
                                    ])
                                ], color="light", className="h-100")
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("⚙️ Processing Layer", className="text-success mb-2"),
                                        html.P([html.Strong("Delta Live Tables"), " → Streaming ETL pipeline"]),
                                        html.P([html.Strong("Medallion Architecture"), " → Bronze → Silver → Gold"]),
                                        html.P([html.Strong("Data Quality"), " → Comprehensive validation & monitoring"])
                                    ])
                                ], color="light", className="h-100")
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("🧠 Analytics Layer", className="text-danger mb-2"),
                                        html.P([html.Strong("RFM Analysis"), " → Customer behavioral metrics"]),
                                        html.P([html.Strong("K-means Clustering"), " → Intelligent customer segmentation"]),
                                        html.P([html.Strong("Web Dashboard"), " → Real-time business insights"])
                                    ])
                                ], color="light", className="h-100")
                            ], width=4)
                        ], className="mb-4"),
                        
                        # Technology Stack
                        html.Hr(),
                        html.H5("🛠️ Technology Stack", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Data Platform", className="text-info mb-2"),
                                html.Ul([
                                    html.Li([html.Strong("Databricks Community Edition"), " (FREE)"]),
                                    html.Li("Delta Live Tables (ETL)"),
                                    html.Li("Unity Catalog (Governance)"),
                                    html.Li("Apache Spark (Processing)")
                                ])
                            ], width=3),
                            dbc.Col([
                                html.H6("Cloud Infrastructure", className="text-warning mb-2"),
                                html.Ul([
                                    html.Li([html.Strong("AWS S3"), " (~$1-2/month)"]),
                                    html.Li("Delta Lake (ACID Storage)"),
                                    html.Li([html.Strong("GitHub Actions"), " (FREE)"]),
                                    html.Li([html.Strong("Render"), " (FREE tier)"])
                                ])
                            ], width=3),
                            dbc.Col([
                                html.H6("Analytics & ML", className="text-success mb-2"),
                                html.Ul([
                                    html.Li("Python & SQL"),
                                    html.Li("scikit-learn (Clustering)"),
                                    html.Li("Plotly Dash (Visualization)"),
                                    html.Li("Pandas (Data Manipulation)")
                                ])
                            ], width=3),
                            dbc.Col([
                                html.H6("Quality & Monitoring", className="text-danger mb-2"),
                                html.Ul([
                                    html.Li("Data Expectations"),
                                    html.Li("Pipeline Lineage"),
                                    html.Li("Quality Scoring"),
                                    html.Li("Error Handling")
                                ])
                            ], width=3)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Key Features Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H3("✨ Key Features & Capabilities", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("🎯 Business Intelligence", className="text-primary mb-2"),
                                html.Ul([
                                    html.Li("Customer RFM segmentation analysis"),
                                    html.Li("15+ business-friendly customer segments"),
                                    html.Li("Actionable marketing recommendations"),
                                    html.Li("Interactive 3D customer visualization"),
                                    html.Li("Real-time dashboard with daily updates")
                                ])
                            ], width=6),
                            dbc.Col([
                                html.H5("🔧 Data Engineering", className="text-success mb-2"),
                                html.Ul([
                                    html.Li("Automated daily batch processing"),
                                    html.Li("Comprehensive data quality monitoring"),
                                    html.Li("End-to-end data lineage tracking"),
                                    html.Li("Schema evolution and data contracts"),
                                    html.Li("Production-grade error handling")
                                ])
                            ], width=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H5("🤖 Machine Learning", className="text-warning mb-2"),
                                html.Ul([
                                    html.Li("Intelligent K-means clustering with outlier detection"),
                                    html.Li("Automated optimal cluster selection"),
                                    html.Li("Statistical validation with silhouette analysis"),
                                    html.Li("Business-friendly segment naming"),
                                    html.Li("Materialized ML results for applications")
                                ])
                            ], width=6),
                            dbc.Col([
                                html.H5("📊 Data Visualization", className="text-info mb-2"),
                                html.Ul([
                                    html.Li("Interactive customer segment explorer"),
                                    html.Li("Data quality monitoring dashboards"),
                                    html.Li("Pipeline lineage visualization"),
                                    html.Li("Architecture flow diagrams"),
                                    html.Li("Comprehensive data contract documentation")
                                ])
                            ], width=6)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Getting Started Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H3("🚀 Explore the Dashboard", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Navigate through the different sections to explore the complete data engineering pipeline:", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("📈 RFM Analytics", className="text-primary mb-2"),
                                        html.P("Explore customer segmentation results, 3D visualizations, and business recommendations", className="small")
                                    ])
                                ], color="primary", outline=True)
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("✅ Data Quality", className="text-success mb-2"),
                                        html.P("Monitor data quality metrics, validation rules, and comprehensive data contracts", className="small")
                                    ])
                                ], color="success", outline=True)
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("🔗 Data Lineage", className="text-info mb-2"),
                                        html.P("Visualize end-to-end data flow and dependencies across the entire pipeline", className="small")
                                    ])
                                ], color="info", outline=True)
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("🏗️ Architecture", className="text-warning mb-2"),
                                        html.P("Understand the technical architecture, design patterns, and implementation details", className="small")
                                    ])
                                ], color="warning", outline=True)
                            ], width=3)
                        ])
                    ])
                ])
            ], width=12)
        ])
    ]

# Analytics page layout
def create_analytics_page():
    return [
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Customer RFM Segmentation Dashboard", className="text-center mb-4"),
                html.P("Analysis of pre-segmented customer data based on RFM clustering", 
                       className="text-center text-muted mb-4")
            ], width=12)
        ]),
        
        # Key metrics row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(data):,}", className="card-title text-primary"),
                        html.P("Total Customers", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(data['Cluster'].unique())}", className="card-title text-success"),
                        html.P("Customer Segments", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${data['Monetary'].mean():.2f}", className="card-title text-warning"),
                        html.P("Avg Monetary Value", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${data['Monetary'].sum():,.0f}", className="card-title text-info"),
                        html.P("Total Revenue", className="card-text")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Cluster selection row
        dbc.Row([
            dbc.Col([
                html.Label("Select Customer Segments to Visualize:", className="fw-bold"),
                dcc.Dropdown(
                    id='cluster-selector',
                    options=[{'label': desc, 'value': cluster} 
                            for cluster, desc in data[['Cluster', 'Cluster_Description']].drop_duplicates().values],
                    value=list(data['Cluster'].unique()),  # All clusters selected by default
                    multi=True,
                    placeholder="Select segments to display..."
                )
            ], width=12)
        ], className="mb-4"),
        
        # Main visualizations
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='3d-scatter', figure=create_3d_scatter())
            ], width=8),
            dbc.Col([
                dcc.Graph(id='cluster-distribution', figure=create_cluster_distribution())
            ], width=4)
        ], className="mb-4"),
        
        # Analysis tabs
        dbc.Row([
            dbc.Col([
                dcc.Tabs(id="analysis-tabs", value="summary", children=[
                    dcc.Tab(label="Segment Summary", value="summary"),
                    dcc.Tab(label="Monetary Analysis", value="monetary"),
                    dcc.Tab(label="Frequency Analysis", value="frequency"),
                    dcc.Tab(label="Recency Analysis", value="recency"),
                    dcc.Tab(label="Customer Details", value="details"),
                    dcc.Tab(label="Recommendations", value="recommendations")
                ]),
                html.Div(id="tab-content")
            ], width=12)
        ])
    ]

# Data Quality page layout
def create_quality_page():
    return [
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Data Quality Monitoring", className="text-center mb-4"),
                html.P("Monitor data quality across bronze and silver layers", 
                       className="text-center text-muted mb-4")
            ], width=12)
        ]),
        
        # Quality metrics summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{bronze_quality.iloc[0]['rescued_data_quality_score']:.1f}%" if not bronze_quality.empty else "N/A", 
                               className="card-title text-success"),
                        html.P("Bronze Ingestion Quality", className="card-text"),
                        html.Small(f"Rescued data score ({bronze_quality.iloc[0]['processing_date']})" if not bronze_quality.empty else "", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{bronze_quality.iloc[0]['total_records']:,}" if not bronze_quality.empty else "N/A", 
                               className="card-title text-primary"),
                        html.P("Records Processed", className="card-text"),
                        html.Small(f"Latest batch ({bronze_quality.iloc[0]['processing_date']})" if not bronze_quality.empty else "", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{bronze_quality.iloc[0]['null_customer_ids']:,}" if not bronze_quality.empty else "N/A", 
                               className="card-title text-warning"),
                        html.P("Null Customer IDs", className="card-text"),
                        html.Small(f"Latest batch ({bronze_quality.iloc[0]['processing_date']})" if not bronze_quality.empty else "", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{bronze_quality.iloc[0]['bad_quantities']:,}" if not bronze_quality.empty else "N/A", 
                               className="card-title text-danger"),
                        html.P("Bad Quantities", className="card-text"),
                        html.Small(f"Latest batch ({bronze_quality.iloc[0]['processing_date']})" if not bronze_quality.empty else "", className="text-muted")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Charts row
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_record_count_trends())
            ], width=12)
        ], className="mb-4"),
        
        # Issues breakdown and comparison
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_data_issues_breakdown())
            ], width=6),
            dbc.Col([
                html.H5("Bronze vs Silver Layer Comparison", className="mb-3"),
                html.Label("Select Date:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id='comparison-date-selector',
                    options=[{'label': str(date), 'value': str(date)} 
                            for date in sorted(dq_comparison['processing_date'].unique(), reverse=True)] if not dq_comparison.empty else [],
                    value=str(dq_comparison['processing_date'].max()) if not dq_comparison.empty else None,
                    className="mb-3"
                ),
                html.Div(id="comparison-table-container")
            ], width=6)
        ], className="mb-4"),
        
        # Data Contract Details Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📋 Data Contract Specifications", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Detailed schema definitions and data quality expectations across all pipeline layers.", className="mb-4"),
                        
                        # Data Contract Tabs
                        dcc.Tabs(id="data-contract-tabs", value="bronze-schema", children=[
                            dcc.Tab(label="Bronze Layer Schema", value="bronze-schema"),
                            dcc.Tab(label="Silver Layer Schema", value="silver-schema"),
                            dcc.Tab(label="Gold Layer Schema", value="gold-schema"),
                            dcc.Tab(label="Quality Rules", value="quality-rules"),
                            dcc.Tab(label="Data Filters", value="data-filters")
                        ]),
                        html.Div(id="data-contract-content", className="mt-4")
                    ])
                ])
            ], width=12)
        ])
    ]

# Data Architecture page layout
def create_architecture_page():
    return [
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Data Architecture Overview", className="text-center mb-4"),
                html.P("Comprehensive explanation of data engineering architectures used in this retail analytics pipeline", 
                       className="text-center text-muted mb-4"),
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    "For detailed technical specifications, refer to the ",
                    html.A("project README", href="https://github.com/jemusni07/rfm-dashboard/blob/main/README.md", target="_blank", className="alert-link"),
                    " file."
                ], color="info", className="mb-4")
            ], width=12)
        ]),
        
        # Data Flow Visualization
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📊 End-to-End Data Flow Visualization", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Interactive visualization showing how data flows through each architectural layer from source to dashboard.", className="mb-3"),
                        create_architecture_flow(),
                        html.Div([
                            html.P("💡 Tip: Click and drag to pan, use mouse wheel to zoom, and click on nodes to select them.", className="small text-muted mt-2")
                        ]),
                        html.Hr(),
                        html.H6("🎨 Color Legend", className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.H6("🌟 Core Architecture", className="small fw-bold text-primary mb-2 architecture-legend"),
                                html.Div([
                                    html.Span("🟤", style={'fontSize': '16px', 'marginRight': '8px'}),
                                    html.Span("Bronze Layer", className="small fw-bold")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("⚪", style={'fontSize': '16px', 'marginRight': '8px'}),
                                    html.Span("Silver Layer", className="small fw-bold")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("🟡", style={'fontSize': '16px', 'marginRight': '8px'}),
                                    html.Span("Gold Layer", className="small fw-bold")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("🟣", style={'fontSize': '16px', 'marginRight': '8px'}),
                                    html.Span("ML Analytics", className="small fw-bold")
                                ])
                            ], width=12, md=6),
                            dbc.Col([
                                html.H6("🔘 Supporting Layers", className="small fw-bold text-muted mb-2 architecture-legend"),
                                html.Div([
                                    html.Span("⚫", style={'fontSize': '16px', 'marginRight': '8px'}),
                                    html.Span("Data Ingestion", className="small text-muted")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("⚫", style={'fontSize': '16px', 'marginRight': '8px'}),
                                    html.Span("Applications", className="small text-muted")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("⚫", style={'fontSize': '16px', 'marginRight': '8px'}),
                                    html.Span("Data Governance", className="small text-muted")
                                ])
                            ], width=12, md=6)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Architecture flow node information display
        dbc.Row([
            dbc.Col([
                html.Div(id="architecture-flow-info")
            ], width=12)
        ], className="mb-4"),
        
        # Medallion Architecture Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🏛️ Medallion Architecture (Bronze-Silver-Gold)", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("A multi-layered data architecture pattern that progressively refines raw data into analytics-ready datasets.", className="lead mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H6("🥉 Bronze Layer - Raw Data Ingestion", className="text-warning mb-2"),
                                html.Ul([
                                    html.Li("Raw data landing zone with full fidelity from S3 sources"),
                                    html.Li("CloudFiles streaming for real-time ingestion"),
                                    html.Li("Metadata capture and basic filtering"),
                                    html.Li("Original CSV structure preserved with pipeline metadata")
                                ], className="mb-3"),
                                html.P([
                                    html.Strong("Implementation: "), 
                                    html.Code("dlt_scripts/01_bronze_layer.py", className="text-muted")
                                ], className="small")
                            ], width=12, lg=4, className="mb-3 mb-lg-0"),
                            dbc.Col([
                                html.H6("🥈 Silver Layer - Cleaned & Validated", className="text-secondary mb-2"),
                                html.Ul([
                                    html.Li("Clean, validated, and enriched data for analytics"),
                                    html.Li("Data quality expectations and type casting"),
                                    html.Li("Feature engineering (cancellation flags, totals)"),
                                    html.Li("Standardized schema with business rules applied")
                                ], className="mb-3"),
                                html.P([
                                    html.Strong("Implementation: "), 
                                    html.Code("dlt_scripts/02_silver_layer.py", className="text-muted")
                                ], className="small")
                            ], width=12, lg=4, className="mb-3 mb-lg-0"),
                            dbc.Col([
                                html.H6("🥇 Gold Layer - Business Analytics", className="text-warning mb-2"),
                                html.Ul([
                                    html.Li("Aggregated metrics for business intelligence"),
                                    html.Li("RFM calculation (Recency, Frequency, Monetary)"),
                                    html.Li("Customer-level aggregations"),
                                    html.Li("Analytics-ready datasets for ML applications")
                                ], className="mb-3"),
                                html.P([
                                    html.Strong("Implementation: "), 
                                    html.Code("dlt_scripts/05_customer_rfm_gold.sql", className="text-muted")
                                ], className="small")
                            ], width=12, lg=4, className="mb-3 mb-lg-0")
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Data Pipeline Architecture
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🔄 End-to-End Data Pipeline Architecture", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Complete data lifecycle from raw ingestion to machine learning application deployment.", className="lead mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H6("📥 Data Ingestion Layer", className="text-primary mb-2"),
                                html.Ul([
                                    html.Li("Automated daily batch processing from S3"),
                                    html.Li("GitHub Actions for scheduled uploads"),
                                    html.Li("CloudFiles streaming for real-time processing"),
                                    html.Li("External data source integration")
                                ])
                            ], width=12, md=6, className="mb-3 mb-md-0"),
                            dbc.Col([
                                html.H6("🔧 Data Processing Layer", className="text-success mb-2"),
                                html.Ul([
                                    html.Li("Multi-layered ETL pipeline using Delta Live Tables"),
                                    html.Li("Data quality expectations and validation"),
                                    html.Li("Automated schema evolution and data lineage"),
                                    html.Li("Error handling and data rescue capabilities")
                                ])
                            ], width=12, md=6, className="mb-3 mb-md-0")
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H6("💾 Data Storage Layer", className="text-info mb-2"),
                                html.Ul([
                                    html.Li("Delta Lake with ACID transactions"),
                                    html.Li("Time travel and versioning capabilities"),
                                    html.Li("Optimized storage with auto-compaction"),
                                    html.Li("Unity Catalog for governance and discovery")
                                ])
                            ], width=12, md=6, className="mb-3 mb-md-0"),
                            dbc.Col([
                                html.H6("📊 Analytics & Application Layer", className="text-danger mb-2"),
                                html.Ul([
                                    html.Li("RFM customer segmentation analysis"),
                                    html.Li("K-means clustering with intelligent outlier handling"),
                                    html.Li("Materialized customer segments for business use"),
                                    html.Li("Interactive dashboard for real-time insights")
                                ])
                            ], width=12, md=6, className="mb-3 mb-md-0")
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Machine Learning Architecture
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🧠 Machine Learning Architecture", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Intelligent customer segmentation using unsupervised learning with advanced outlier management.", className="lead mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H6("🎯 Clustering Strategy", className="mb-2"),
                                html.Ul([
                                    html.Li("Automated K-selection using silhouette score (60%) + elbow method (40%)"),
                                    html.Li("3D analysis on scaled Recency, Frequency, and Monetary dimensions"),
                                    html.Li("StandardScaler normalization for fair clustering"),
                                    html.Li("Business-friendly segment classifications")
                                ], className="mb-3")
                            ], width=6),
                            dbc.Col([
                                html.H6("🔍 Outlier Management", className="mb-2"),
                                html.Ul([
                                    html.Li("IQR-based outlier detection before clustering"),
                                    html.Li("Separate outlier clusters (-1, -2, -3) for extreme customers"),
                                    html.Li("Preserves main segment accuracy while identifying VIP/risk customers"),
                                    html.Li("Specialized business treatment for outlier segments")
                                ], className="mb-3")
                            ], width=6)
                        ]),
                        
                        html.P([
                            html.Strong("Implementation: "), 
                            html.Code("customer_segmentation_kmeans_clustering/RFM data clustering.ipynb", className="text-muted")
                        ], className="small")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Data Quality & Monitoring Architecture
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📈 Data Quality & Monitoring Architecture", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Comprehensive monitoring and quality assurance across all pipeline layers.", className="lead mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H6("✅ Quality Expectations", className="text-success mb-2"),
                                html.Ul([
                                    html.Li("Invoice validation (6-7 characters, not null)"),
                                    html.Li("Stock code presence validation"),
                                    html.Li("Quantity and price range validation"),
                                    html.Li("Date format and business rule validation")
                                ])
                            ], width=4),
                            dbc.Col([
                                html.H6("📊 Monitoring Scripts", className="text-info mb-2"),
                                html.Ul([
                                    html.Li([html.Code("bronze_dq.sql", className="small"), " - Ingestion quality tracking"]),
                                    html.Li([html.Code("dlt_daily_counts.sql", className="small"), " - Processing metrics"]),
                                    html.Li([html.Code("bronze_silver_dq_comparison.sql", className="small"), " - Layer validation"]),
                                    html.Li("Data lineage tracking and observability")
                                ])
                            ], width=4),
                            dbc.Col([
                                html.H6("🔧 Data Filters", className="text-warning mb-2"),
                                html.Ul([
                                    html.Li("Excludes cancellation transactions (Invoice starting with 'C')"),
                                    html.Li("Stock code pattern validation"),
                                    html.Li("Customer ID presence requirement for RFM"),
                                    html.Li("Invalid quantity and price removal")
                                ])
                            ], width=4)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Technology Stack Summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🛠️ Technology Stack", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Data Platform", className="mb-2"),
                                html.Ul([
                                    html.Li("Databricks (Unified Analytics Platform)"),
                                    html.Li("Delta Live Tables (Pipeline Framework)"),
                                    html.Li("Unity Catalog (Data Governance)"),
                                    html.Li("Databricks Workflows (Orchestration)")
                                ])
                            ], width=3),
                            dbc.Col([
                                html.H6("Storage & Processing", className="mb-2"),
                                html.Ul([
                                    html.Li("AWS S3 (Raw file storage)"),
                                    html.Li("Delta Lake (Processed data)"),
                                    html.Li("Apache Spark (Distributed processing)"),
                                    html.Li("CloudFiles (Streaming ingestion)")
                                ])
                            ], width=3),
                            dbc.Col([
                                html.H6("Analytics & ML", className="mb-2"),
                                html.Ul([
                                    html.Li("scikit-learn (K-means clustering)"),
                                    html.Li("Databricks Notebooks & Dashboards"),
                                    html.Li("Plotly Dash (Interactive dashboard)"),
                                    html.Li("Python & SQL (Analytics languages)")
                                ])
                            ], width=3),
                            dbc.Col([
                                html.H6("Automation & DevOps", className="mb-2"),
                                html.Ul([
                                    html.Li("GitHub Actions (CI/CD)"),
                                    html.Li("Render (Dashboard deployment)"),
                                    html.Li("Environment-based configuration"),
                                    html.Li("Automated testing and validation")
                                ])
                            ], width=3)
                        ])
                    ])
                ])
            ], width=12)
        ])
    ]

# Lineage page layout
def create_lineage_page():
    return [
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Data Lineage Dashboard", className="text-center mb-4"),
                html.P("Interactive visualization of your retail analytics data pipeline in Unity Catalog", 
                       className="text-center text-muted mb-4")
            ], width=12)
        ]),
        
        # Enhanced summary metrics with pipeline insights
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(lineage_data)}", className="card-title text-primary"),
                        html.P("Data Relationships", className="card-text"),
                        html.Small("Total lineage connections", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(lineage_data[lineage_data['source_type'] == 'STREAMING_TABLE']) if not lineage_data.empty else 0}", 
                               className="card-title text-danger"),
                        html.P("Streaming Tables", className="card-text"),
                        html.Small("Real-time data flows", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(lineage_data[lineage_data['source_type'] == 'TABLE']) if not lineage_data.empty else 0}", 
                               className="card-title text-info"),
                        html.P("Data Tables", className="card-text"),
                        html.Small("Stored datasets", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(lineage_data[lineage_data['source_type'] == 'PATH']) if not lineage_data.empty else 0}", 
                               className="card-title text-warning"),
                        html.P("S3 Data Sources", className="card-text"),
                        html.Small("External data inputs", className="text-muted")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Pipeline flow description with Unity Catalog context
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("📊 Unity Catalog Data Pipeline Flow", className="alert-heading"),
                    html.P("Raw S3 Data → DLT Bronze → DLT Silver → DLT Gold → ML Analytics → Customer Insights", className="mb-2"),
                    html.P("📚 All tables are organized within the retail_analytics catalog in Unity Catalog", className="mb-0 small text-info"),
                    html.Hr(),
                    html.P("This horizontal visualization shows how data flows through your retail analytics platform, with full schema.table names displayed for each Unity Catalog asset.", className="mb-0 small")
                ], color="light", className="mb-4")
            ], width=12)
        ]),
        
        # Enhanced network visualization with Cytoscape
        dbc.Row([
            dbc.Col([
                create_cytoscape_lineage()
            ], width=12)
        ], className="mb-4"),
        
        # Interactive controls and enhanced legend
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎯 Interactive Controls"),
                    dbc.CardBody([
                        html.P("• Click and drag to pan around the diagram", className="mb-1"),
                        html.P("• Use mouse wheel to zoom in/out", className="mb-1"),
                        html.P("• Click on nodes to select and view detailed information below", className="mb-1"),
                        html.P("• Double-click to fit diagram to view", className="mb-0")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🏷️ Color Legend"),
                    dbc.CardBody([
                        html.Div([
                            html.Span("🟠", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("S3 Sources", style={'fontWeight': 'bold', 'color': '#F39C12'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🟤", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("DLT Bronze Tables", style={'fontWeight': 'bold', 'color': '#CD7F32'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("⚪", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("DLT Silver Tables", style={'fontWeight': 'bold', 'color': '#C0C0C0'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🟡", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("DLT Gold Tables", style={'fontWeight': 'bold', 'color': '#FFD700'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🟣", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("ML Schema Tables", style={'fontWeight': 'bold', 'color': '#9B59B6'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🟢", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("Other Tables", style={'fontWeight': 'bold', 'color': '#48C9B0'})
                        ])
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Node information display
        dbc.Row([
            dbc.Col([
                html.Div(id="node-info-display")
            ], width=12)
        ])
    ]

# Centralized color mapping functions for consistent colors across all charts
def get_cluster_color_map():
    """Get consistent color mapping for clusters across all charts."""
    if data is None:
        return {}
    
    # Teal-based color palette optimized for readability
    color_palette = ['#20b2aa', '#5f9ea0', '#4682b4', '#008b8b', '#2e8b57', '#6495ed', '#40e0d0', '#48d1cc', '#00ced1', '#87ceeb']
    
    # Always use cluster IDs as the base for color assignment
    unique_clusters = sorted(data['Cluster'].unique())
    cluster_color_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
    
    return cluster_color_map

def get_segment_color_map():
    """Get color mapping for segments that maps back to cluster colors."""
    if data is None:
        return {}
    
    cluster_colors = get_cluster_color_map()
    segment_color_map = {}
    
    # Create a mapping from cluster descriptions back to cluster IDs
    if 'Cluster_Description' in data.columns:
        # Map each unique cluster description to its cluster's color
        cluster_to_desc = data[['Cluster', 'Cluster_Description']].drop_duplicates()
        for _, row in cluster_to_desc.iterrows():
            cluster_id = row['Cluster']
            description = row['Cluster_Description']
            segment_color_map[description] = cluster_colors.get(cluster_id, '#20b2aa')
    
    # Handle other segment column possibilities
    if 'Segment' in data.columns:
        segment_to_cluster = data[['Cluster', 'Segment']].drop_duplicates()
        for _, row in segment_to_cluster.iterrows():
            cluster_id = row['Cluster']
            segment = row['Segment']
            segment_color_map[segment] = cluster_colors.get(cluster_id, '#20b2aa')
    
    # Also map raw cluster identifiers for compatibility
    for cluster_id, color in cluster_colors.items():
        segment_color_map[cluster_id] = color
        segment_color_map[str(cluster_id)] = color
        segment_color_map[f'Cluster {cluster_id}'] = color
    
    return segment_color_map

# Initialize the Dash app with modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
server = app.server  # For deployment

# Custom CSS for modern design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                margin: 0;
                padding: 0;
            }
            .sidebar {
                background: linear-gradient(135deg, #20b2aa 0%, #008b8b 100%);
                min-height: 100vh;
                padding: 0;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }
            .sidebar-header {
                padding: 2rem 1rem;
                text-align: center;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                margin-bottom: 1rem;
            }
            .sidebar-brand {
                color: white;
                font-size: 1.5rem;
                font-weight: bold;
                text-decoration: none;
                display: block;
                margin-bottom: 0.5rem;
            }
            .sidebar-subtitle {
                color: rgba(255,255,255,0.8);
                font-size: 0.9rem;
                margin: 0;
            }
            .nav-item {
                margin-bottom: 0.5rem;
            }
            .nav-link {
                color: rgba(255,255,255,0.9) !important;
                padding: 0.75rem 1.5rem;
                border-radius: 0.5rem;
                margin: 0 1rem;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                text-decoration: none;
            }
            .nav-link:hover {
                background-color: rgba(255,255,255,0.1);
                color: white !important;
                transform: translateX(5px);
            }
            .nav-link.active {
                background-color: rgba(255,255,255,0.2);
                color: white !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }
            .nav-link i {
                margin-right: 0.75rem;
                width: 1.25rem;
                text-align: center;
            }
            .main-content {
                padding: 2rem;
                background-color: #f8f9fa;
                min-height: 100vh;
            }
            .metric-card {
                background: white;
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(0,0,0,0.05);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 25px rgba(0,0,0,0.12);
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #2c3e50;
                margin: 0;
                line-height: 1.2;
            }
            .metric-label {
                color: #6c757d;
                font-size: 0.9rem;
                margin-top: 0.5rem;
                margin-bottom: 0;
            }
            .metric-icon {
                font-size: 2rem;
                margin-bottom: 1rem;
            }
            .page-header {
                background: white;
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(0,0,0,0.05);
            }
            .page-title {
                color: #2c3e50;
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            .page-subtitle {
                color: #6c757d;
                font-size: 1.1rem;
                margin: 0;
            }
            .chart-container {
                background: white;
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(0,0,0,0.05);
                margin-bottom: 2rem;
            }
            .refresh-button {
                background: linear-gradient(135deg, #20b2aa 0%, #008b8b 100%);
                border: none;
                border-radius: 0.5rem;
                color: white;
                padding: 0.5rem 1rem;
                font-size: 0.9rem;
                transition: all 0.3s ease;
            }
            .refresh-button:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 15px rgba(32, 178, 170, 0.4);
            }
            .loading-spinner {
                display: inline-block;
                width: 1rem;
                height: 1rem;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 0.5rem;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .status-success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
                border-radius: 0.5rem;
                padding: 0.75rem 1rem;
                margin-bottom: 1rem;
            }
            .status-error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                border-radius: 0.5rem;
                padding: 0.75rem 1rem;
                margin-bottom: 1rem;
            }
            .filter-panel {
                background: white;
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(0,0,0,0.05);
                margin-bottom: 2rem;
            }
            /* Responsive Design */
            @media (max-width: 1200px) {
                .sidebar {
                    width: 250px;
                }
                .main-content {
                    margin-left: 250px;
                }
            }
            
            @media (max-width: 992px) {
                .sidebar {
                    position: fixed;
                    top: 0;
                    left: -250px;
                    width: 250px;
                    transition: left 0.3s ease;
                    z-index: 1000;
                }
                .sidebar.open {
                    left: 0;
                }
                .main-content {
                    margin-left: 0 !important;
                    padding: 1rem;
                }
                .page-header {
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                }
                .page-title {
                    font-size: 1.75rem;
                }
                .chart-container {
                    padding: 1rem;
                    margin-bottom: 1.5rem;
                }
            }
            
            @media (max-width: 768px) {
                .main-content {
                    padding: 0.75rem;
                    padding-top: 4rem; /* Account for mobile menu button */
                }
                .metric-card {
                    margin-bottom: 1rem;
                    padding: 1rem;
                }
                .metric-value {
                    font-size: 2rem;
                }
                .page-header {
                    padding: 1rem;
                    margin-bottom: 1rem;
                }
                .page-title {
                    font-size: 1.5rem;
                }
                .chart-container {
                    padding: 0.75rem;
                    margin-bottom: 1rem;
                }
                .filter-panel {
                    padding: 1rem;
                    margin-bottom: 1rem;
                }
                /* Responsive tables */
                .ag-theme-alpine {
                    font-size: 12px;
                }
                /* Better touch targets */
                .nav-link {
                    padding: 1rem 1.5rem;
                    font-size: 1rem;
                    min-height: 44px; /* iOS touch target recommendation */
                }
                .refresh-button {
                    padding: 0.75rem 1rem;
                    font-size: 1rem;
                    min-height: 44px;
                }
                /* Mobile menu improvements */
                #mobile-menu-toggle {
                    min-width: 44px;
                    min-height: 44px;
                }
            }
            
            @media (max-width: 576px) {
                .main-content {
                    padding: 0.5rem;
                    padding-top: 4rem;
                }
                .metric-card {
                    padding: 0.75rem;
                }
                .metric-value {
                    font-size: 1.75rem;
                }
                .page-header {
                    padding: 0.75rem;
                }
                .page-title {
                    font-size: 1.25rem;
                }
                .chart-container {
                    padding: 0.5rem;
                }
                .filter-panel {
                    padding: 0.75rem;
                }
                /* Stack metric cards vertically on small screens */
                .row .col-lg-3, .row .col-md-4, .row .col-sm-6 {
                    margin-bottom: 0.75rem;
                }
                /* Mobile-specific architecture adjustments */
                #architecture-flow-cytoscape {
                    height: 350px !important;
                    font-size: 10px !important;
                }
                /* Make text in architecture diagram smaller on mobile */
                .architecture-legend {
                    font-size: 0.75rem;
                }
                .card-header h4 {
                    font-size: 1.1rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the modern app layout with sidebar navigation
app.layout = html.Div([
    dcc.Store(id='active-tab-store', data='dashboard'),
    dcc.Store(id='data-refresh-store', data={'timestamp': datetime.now().isoformat()}),
    
    # Mobile menu toggle (hidden on desktop)
    html.Div([
        dbc.Button(
            html.I(className="fas fa-bars"),
            id="mobile-menu-toggle",
            className="d-md-none",
            style={"position": "fixed", "top": "1rem", "left": "1rem", "z-index": "1001", "background": "linear-gradient(135deg, #20b2aa 0%, #008b8b 100%)", "border": "none"}
        )
    ]),
    
    # Sidebar Navigation
    html.Div([
        # Sidebar Header
        html.Div([
            html.A("RFM Analytics", className="sidebar-brand", href="#"),
            html.P("Customer Intelligence Dashboard", className="sidebar-subtitle")
        ], className="sidebar-header"),
        
        # Navigation Menu
        dbc.Nav([
            dbc.NavItem(dbc.NavLink([
                html.I(className="fas fa-tachometer-alt"),
                " Executive Dashboard"
            ], href="#", id="dashboard-tab", active=True, className="nav-link")),
            
            dbc.NavItem(dbc.NavLink([
                html.I(className="fas fa-chart-scatter"),
                " RFM Analytics"
            ], href="#", id="analytics-tab", active=False, className="nav-link")),
            
            dbc.NavItem(dbc.NavLink([
                html.I(className="fas fa-users"),
                " Customer Segments"
            ], href="#", id="segments-tab", active=False, className="nav-link")),
            
            dbc.NavItem(dbc.NavLink([
                html.I(className="fas fa-shield-alt"),
                " Data Quality"
            ], href="#", id="quality-tab", active=False, className="nav-link")),
            
            dbc.NavItem(dbc.NavLink([
                html.I(className="fas fa-project-diagram"),
                " Data Lineage"
            ], href="#", id="lineage-tab", active=False, className="nav-link")),
            
            dbc.NavItem(dbc.NavLink([
                html.I(className="fas fa-layer-group"),
                " Architecture"
            ], href="#", id="architecture-tab", active=False, className="nav-link")),
            
            dbc.NavItem(dbc.NavLink([
                html.I(className="fas fa-info-circle"),
                " About Project"
            ], href="#", id="about-tab", active=False, className="nav-link"))
        ], vertical=True, className="flex-column"),
        
        # Sidebar Footer with Refresh Button
        html.Div([
            dbc.Button([
                html.I(className="fas fa-sync-alt", id="refresh-icon"),
                " Refresh Data"
            ], id="refresh-button", className="refresh-button w-100", size="sm"),
            
            html.Div(id="refresh-status", className="mt-2"),
            
            html.Hr(style={"border-color": "rgba(255,255,255,0.1)", "margin": "1rem 0"}),
            
            html.Div([
                html.P([
                    html.I(className="fas fa-database", style={"margin-right": "0.5rem"}),
                    "Last Updated: ",
                    html.Span(id="last-updated", children=datetime.now().strftime("%H:%M:%S"))
                ], className="small text-light mb-2"),
                
                html.P([
                    html.I(className="fas fa-chart-line", style={"margin-right": "0.5rem"}),
                    "Total Customers: ",
                    html.Span(id="total-customers", children=str(len(data)) if data is not None else "0")
                ], className="small text-light mb-0")
            ], style={"padding": "0 1rem"})
        ], style={"position": "absolute", "bottom": "2rem", "left": "0", "right": "0", "padding": "0 1rem"})
    ], className="sidebar", id="sidebar", style={"position": "fixed", "left": "0", "top": "0", "width": "280px", "height": "100vh", "z-index": "1000"}),
    
    # Main Content Area
    html.Div([
        html.Div(id="page-content", children=[])
    ], className="main-content", style={"margin-left": "280px", "min-height": "100vh"})
])

# Enhanced callback for navigation and refresh
@app.callback(
    [Output('refresh-status', 'children'),
     Output('refresh-icon', 'className'),
     Output('last-updated', 'children'),
     Output('total-customers', 'children'),
     Output('dashboard-tab', 'active'),
     Output('analytics-tab', 'active'),
     Output('segments-tab', 'active'),
     Output('quality-tab', 'active'),
     Output('lineage-tab', 'active'),
     Output('architecture-tab', 'active'),
     Output('about-tab', 'active'),
     Output('active-tab-store', 'data'),
     Output('page-content', 'children')],
    [Input('refresh-button', 'n_clicks'),
     Input('dashboard-tab', 'n_clicks'),
     Input('analytics-tab', 'n_clicks'),
     Input('segments-tab', 'n_clicks'),
     Input('quality-tab', 'n_clicks'),
     Input('lineage-tab', 'n_clicks'),
     Input('architecture-tab', 'n_clicks'),
     Input('about-tab', 'n_clicks')],
    [State('active-tab-store', 'data')]
)
def update_page(refresh_clicks, dashboard_clicks, analytics_clicks, segments_clicks, quality_clicks, lineage_clicks, architecture_clicks, about_clicks, current_tab):
    ctx = callback_context
    
    refresh_status = ""
    refresh_icon = "fas fa-sync-alt"
    last_updated = datetime.now().strftime("%H:%M:%S")
    total_customers = str(len(data)) if data is not None else "0"
    
    # Handle refresh button click
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'refresh-button.n_clicks':
        refresh_all_data()
        refresh_status = html.Div([
            html.I(className="fas fa-check-circle", style={"margin-right": "0.5rem"}),
            "Data refreshed successfully!"
        ], className="status-success")
        refresh_icon = "fas fa-sync-alt"
        last_updated = datetime.now().strftime("%H:%M:%S")
        total_customers = str(len(data)) if data is not None else "0"
        # Keep current tab after refresh
        button_id = current_tab if current_tab else 'dashboard-tab'
    else:
        # Get which button was clicked
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        else:
            button_id = 'dashboard-tab'  # Default to dashboard
    
    # Reset all tab states
    dashboard_active = False
    analytics_active = False
    segments_active = False
    quality_active = False
    lineage_active = False
    architecture_active = False
    about_active = False
    
    # Set active tab and create content
    if button_id == 'dashboard-tab':
        dashboard_active = True
        page_content = create_dashboard_page()
        active_tab = 'dashboard-tab'
    elif button_id == 'analytics-tab':
        analytics_active = True
        page_content = create_analytics_page()
        active_tab = 'analytics-tab'
    elif button_id == 'segments-tab':
        segments_active = True
        page_content = create_segments_page()
        active_tab = 'segments-tab'
    elif button_id == 'quality-tab':
        quality_active = True
        page_content = create_quality_page()
        active_tab = 'quality-tab'
    elif button_id == 'lineage-tab':
        lineage_active = True
        page_content = create_lineage_page()
        active_tab = 'lineage-tab'
    elif button_id == 'architecture-tab':
        architecture_active = True
        page_content = create_architecture_page()
        active_tab = 'architecture-tab'
    elif button_id == 'about-tab':
        about_active = True
        page_content = create_about_page()
        active_tab = 'about-tab'
    else:
        dashboard_active = True
        page_content = create_dashboard_page()
        active_tab = 'dashboard-tab'
    
    return (
        refresh_status, refresh_icon, last_updated, total_customers,
        dashboard_active, analytics_active, segments_active, 
        quality_active, lineage_active, architecture_active, about_active,
        active_tab, page_content
    )

# Mobile menu toggle callback
@app.callback(
    Output('sidebar', 'style'),
    [Input('mobile-menu-toggle', 'n_clicks')],
    [State('sidebar', 'style')]
)
def toggle_mobile_menu(n_clicks, current_style):
    if not n_clicks:
        return {"position": "fixed", "left": "0", "top": "0", "width": "280px", "height": "100vh", "z-index": "1000"}
    
    # Toggle the sidebar visibility on mobile
    if current_style and 'left' in current_style:
        if current_style['left'] == '0':
            return {"position": "fixed", "left": "-280px", "top": "0", "width": "280px", "height": "100vh", "z-index": "1000"}
        else:
            return {"position": "fixed", "left": "0", "top": "0", "width": "280px", "height": "100vh", "z-index": "1000"}
    
    return {"position": "fixed", "left": "0", "top": "0", "width": "280px", "height": "100vh", "z-index": "1000"}

# Segment filter callback
@app.callback(
    Output('segment-comparison-grid', 'rowData'),
    [Input('segment-filter', 'value'),
     Input('min-customers-slider', 'value')]
)
def update_segment_filters(selected_segments, min_customers):
    if data is None:
        return []
    
    # Find segment column
    segment_col = None
    for col in ['Segment', 'segment', 'customer_segment', 'segment_name', 'SegmentName', 'Cluster', 'cluster']:
        if col in data.columns:
            segment_col = col
            break
    
    if not segment_col or not selected_segments:
        return []
    
    # Filter data based on selections
    filtered_data = data[data[segment_col].isin(selected_segments)]
    
    # Find other columns
    customer_col = next((col for col in ['CustomerID', 'customer_id', 'CustomerId', 'ID', 'id'] if col in data.columns), None)
    monetary_col = next((col for col in ['Monetary', 'monetary', 'total_spend', 'revenue', 'TotalSpend'] if col in data.columns), None)
    frequency_col = next((col for col in ['Frequency', 'frequency', 'order_count', 'purchase_count'] if col in data.columns), None)
    recency_col = next((col for col in ['Recency', 'recency', 'days_since_last_purchase'] if col in data.columns), None)
    
    # Create aggregation dictionary
    agg_dict = {}
    if customer_col:
        agg_dict[customer_col] = 'count'
    if monetary_col:
        agg_dict[monetary_col] = ['mean', 'sum', 'std']
    if frequency_col:
        agg_dict[frequency_col] = ['mean', 'std']
    if recency_col:
        agg_dict[recency_col] = ['mean', 'std']
    
    if not agg_dict:
        return []
    
    # Perform aggregation
    segment_analysis = filtered_data.groupby(segment_col).agg(agg_dict).round(2)
    
    # Flatten column names
    new_columns = []
    for col in segment_analysis.columns:
        if isinstance(col, tuple):
            if col[1] == 'count':
                new_columns.append('Customer_Count')
            elif col[1] == 'mean' and monetary_col and monetary_col.lower() in col[0].lower():
                new_columns.append('Avg_Monetary')
            elif col[1] == 'sum' and monetary_col and monetary_col.lower() in col[0].lower():
                new_columns.append('Total_Revenue')
            elif col[1] == 'std' and monetary_col and monetary_col.lower() in col[0].lower():
                new_columns.append('Monetary_Std')
            elif col[1] == 'mean' and frequency_col and frequency_col.lower() in col[0].lower():
                new_columns.append('Avg_Frequency')
            elif col[1] == 'std' and frequency_col and frequency_col.lower() in col[0].lower():
                new_columns.append('Frequency_Std')
            elif col[1] == 'mean' and recency_col and recency_col.lower() in col[0].lower():
                new_columns.append('Avg_Recency')
            elif col[1] == 'std' and recency_col and recency_col.lower() in col[0].lower():
                new_columns.append('Recency_Std')
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(str(col))
    
    segment_analysis.columns = new_columns
    segment_analysis = segment_analysis.reset_index()
    
    # Add description and recommendation if available
    description_cols = ['Cluster_Description', 'SegmentDescription', 'Description']
    recommendation_cols = ['Cluster_Recommendation', 'SegmentRecommendation', 'Recommendation', 'Strategy']
    
    # Find description column
    desc_col = None
    for col in description_cols:
        if col in data.columns:
            desc_col = col
            break
    
    # Find recommendation column  
    rec_col = None
    for col in recommendation_cols:
        if col in data.columns:
            rec_col = col
            break
    
    # Add description and recommendation to the analysis
    if desc_col or rec_col:
        # Get unique segment descriptions and recommendations
        segment_info = data[[segment_col] + [col for col in [desc_col, rec_col] if col]].drop_duplicates()
        segment_info = segment_info.set_index(segment_col)
        
        # Merge with segment analysis
        if desc_col:
            segment_analysis['Description'] = segment_analysis[segment_col].map(segment_info[desc_col])
        if rec_col:
            segment_analysis['Recommendation'] = segment_analysis[segment_col].map(segment_info[rec_col])
    
    # Filter by minimum customer count
    if 'Customer_Count' in segment_analysis.columns:
        segment_analysis = segment_analysis[segment_analysis['Customer_Count'] >= min_customers]
    
    return segment_analysis.to_dict('records')

# Update segment filter options when page loads
@app.callback(
    [Output('segment-filter', 'options'),
     Output('segment-filter', 'value')],
    [Input('active-tab-store', 'data')]
)
def update_segment_filter_options(active_tab):
    if data is None or active_tab != 'segments-tab':
        return [], []
    
    # Find segment column
    segment_col = None
    for col in ['Segment', 'segment', 'customer_segment', 'segment_name', 'SegmentName', 'Cluster', 'cluster']:
        if col in data.columns:
            segment_col = col
            break
    
    if not segment_col:
        return [], []
    
    unique_segments = sorted(data[segment_col].unique())
    options = [{'label': str(segment), 'value': segment} for segment in unique_segments]
    
    return options, unique_segments

# Update min customers slider based on data
@app.callback(
    [Output('min-customers-slider', 'max'),
     Output('min-customers-slider', 'marks'),
     Output('min-customers-slider', 'value')],
    [Input('active-tab-store', 'data')]
)
def update_min_customers_slider(active_tab):
    if data is None or active_tab != 'segments-tab':
        return 100, {}, 1
    
    # Find segment column
    segment_col = None
    for col in ['Segment', 'segment', 'customer_segment', 'segment_name', 'SegmentName', 'Cluster', 'cluster']:
        if col in data.columns:
            segment_col = col
            break
    
    if not segment_col:
        return 100, {}, 1
    
    max_customers = data[segment_col].value_counts().max()
    step = max(1, max_customers // 10)
    marks = {i: str(i) for i in range(0, max_customers + 1, step)}
    
    return max_customers, marks, 1

# Update filter status
@app.callback(
    Output('filter-status', 'children'),
    [Input('segment-filter', 'value'),
     Input('min-customers-slider', 'value')]
)
def update_filter_status(selected_segments, min_customers):
    if not selected_segments:
        return "No segments selected"
    
    status_parts = []
    status_parts.append(f"{len(selected_segments)} segment(s) selected")
    
    if min_customers > 1:
        status_parts.append(f"min {min_customers} customers")
    
    return " | ".join(status_parts)

@app.callback(
    Output('3d-scatter', 'figure'),
    Input('cluster-selector', 'value')
)
def update_3d_scatter(selected_clusters):
    return create_3d_scatter(selected_clusters)

@app.callback(
    Output('tab-content', 'children'),
    Input('analysis-tabs', 'value')
)
def render_tab_content(active_tab):
    if active_tab == "summary":
        cluster_summary = get_cluster_summary()
        
        return dag.AgGrid(
            id='cluster-summary-grid',
            columnDefs=[
                {"headerName": "Cluster", "field": "Cluster", "width": 80},
                {"headerName": "Description", "field": "Cluster_Description", "width": 180},
                {"headerName": "Count", "field": "Count", "width": 80},
                {"headerName": "Avg Monetary", "field": "Avg_Monetary", "width": 120, "type": "numericColumn"},
                {"headerName": "Total Revenue", "field": "Total_Monetary", "width": 120, "type": "numericColumn"},
                {"headerName": "Avg Frequency", "field": "Avg_Frequency", "width": 120, "type": "numericColumn"},
                {"headerName": "Avg Recency", "field": "Avg_Recency", "width": 120, "type": "numericColumn"}
            ],
            rowData=cluster_summary.to_dict('records'),
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            style={'height': '400px'}
        )
    
    elif active_tab == "monetary":
        return dcc.Graph(figure=create_monetary_analysis())
    
    elif active_tab == "frequency":
        return dcc.Graph(figure=create_frequency_analysis())
    
    elif active_tab == "recency":
        return dcc.Graph(figure=create_recency_analysis())
    
    elif active_tab == "details":
        return dag.AgGrid(
            id='customer-details-grid',
            columnDefs=[
                {"headerName": "Customer ID", "field": "CustomerID", "width": 120},
                {"headerName": "Recency", "field": "Recency", "width": 100},
                {"headerName": "Frequency", "field": "Frequency", "width": 100},
                {"headerName": "Monetary", "field": "Monetary", "width": 120, "type": "numericColumn"},
                {"headerName": "Segment", "field": "Cluster_Description", "width": 180},
            ],
            rowData=data.to_dict('records'),
            defaultColDef={"resizable": True, "sortable": True, "filter": True},
            style={'height': '500px'},
            dashGridOptions={"pagination": True, "paginationPageSize": 20}
        )
    
    elif active_tab == "recommendations":
        recommendations_summary = data.groupby(['Cluster_Description', 'Cluster_Recommendation']).size().reset_index(name='Customer_Count')
        
        return html.Div([
            html.H5("Marketing Recommendations by Segment", className="mb-3"),
            *[
                dbc.Card([
                    dbc.CardHeader(html.H6(row['Cluster_Description'], className="mb-0")),
                    dbc.CardBody([
                        html.P(row['Cluster_Recommendation'], className="card-text"),
                        html.Small(f"Applies to {row['Customer_Count']} customers", className="text-muted")
                    ])
                ], className="mb-3")
                for _, row in recommendations_summary.iterrows()
            ]
        ])

# Callback for architecture flow node selection
@app.callback(
    Output('architecture-flow-info', 'children'),
    Input('architecture-flow-cytoscape', 'selectedNodeData')
)
def display_architecture_node_info(selected_nodes):
    if not selected_nodes:
        return dbc.Alert("Select a node in the architecture flow diagram to see detailed information", color="info", className="mt-3")
    
    node = selected_nodes[0]  # Take the first selected node
    node_id = node.get('id', 'Unknown')
    node_type = node.get('type', 'unknown')
    node_label = node.get('label', 'Unknown')
    
    # Create descriptions based on node type and id
    descriptions = {
        'data_ingestion': "Automated data ingestion pipeline that handles the complete flow from data sources to Delta Lake. Includes GitHub Actions for scheduling, S3 for raw storage, and Delta Live Tables for streaming ingestion with built-in quality controls.",
        'bronze_layer': "Raw data ingestion layer implementing medallion architecture principles. Preserves complete data fidelity from source systems while adding essential metadata, lineage tracking, and basic filtering for downstream processing.",
        'silver_layer': "Data cleansing and validation layer applying comprehensive business rules. Performs data quality validation, type casting, feature engineering, standardization, and applies business logic to prepare clean data for analytics.",
        'gold_layer': "Analytics-ready data layer containing business-optimized aggregations and metrics. Produces RFM customer behavioral analytics, aggregated metrics, and feature sets optimized for machine learning and business intelligence applications.",
        'ml_layer': "Advanced machine learning analytics layer implementing customer segmentation. Features K-means clustering with intelligent outlier detection, automated cluster optimization, statistical validation, and business-friendly segment generation.",
        'application_layer': "Interactive application and visualization layer delivering business insights. Built with Plotly Dash, provides real-time analytics dashboard, 3D customer visualizations, segment analysis, and comprehensive business intelligence capabilities.",
        'data_governance': "Cross-cutting data governance framework ensuring data quality and pipeline observability. Implements quality monitoring, validation rules, lineage tracking, performance metrics, and comprehensive observability across all pipeline layers."
    }
    
    description = descriptions.get(node_id, "Detailed information about this component is available in the project documentation.")
    
    # Create type-specific styling - highlighting medallion and ML layers
    type_colors = {
        'source': 'secondary',
        'automation': 'secondary', 
        'storage': 'secondary',
        'ingestion': 'secondary',
        'bronze': 'warning',     # Highlighted - Bronze
        'silver': 'light',       # Highlighted - Silver 
        'gold': 'warning',       # Highlighted - Gold
        'ml': 'primary',         # Highlighted - ML
        'ml_output': 'primary',  # Highlighted - ML Output
        'application': 'secondary',
        'quality': 'secondary',
        'monitoring': 'secondary'
    }
    
    color = type_colors.get(node_type, 'secondary')
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5(f"📋 {node_label.split()[0]} Details", className="mb-0")
        ]),
        dbc.CardBody([
            html.P([
                html.Strong("Component: "), 
                html.Span(node_label.replace('\n', ' - '))
            ]),
            html.P([
                html.Strong("Type: "), 
                html.Span(node_type.replace('_', ' ').title())
            ]),
            html.P([
                html.Strong("Description: "), 
                html.Span(description)
            ]),
            html.Hr(),
            html.P([
                html.Strong("💡 Implementation: "),
                "Refer to the project README for specific code files and technical details."
            ], className="small text-muted mb-0")
        ])
    ], className="mt-3")

# New callback for Cytoscape node selection
@app.callback(
    Output('node-info-display', 'children'),
    Input('lineage-cytoscape', 'selectedNodeData')
)
def display_node_info(selected_nodes):
    if not selected_nodes:
        return dbc.Alert("Select a node in the diagram above to see detailed information", color="info", className="mt-3")
    
    node = selected_nodes[0]  # Take the first selected node
    
    # Find related edges for this node
    related_edges = []
    if not lineage_data.empty:
        for _, row in lineage_data.iterrows():
            source = row['source_table_full_name'] if pd.notna(row['source_table_full_name']) else row['source_path']
            target = row['target_table_full_name']
            
            if str(source) == node['id'] or str(target) == node['id']:
                related_edges.append({
                    'source': source,
                    'target': target,
                    'source_type': row['source_type'],
                    'target_type': row['target_type']
                })
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5(f"📋 Unity Catalog Asset Details: {node.get('label', 'Unknown')}", className="mb-0")
        ]),
        dbc.CardBody([
            html.P([
                html.Strong("Full Path: "), 
                html.Code(node.get('full_path', 'N/A'))
            ]),
            html.P([
                html.Strong("Asset Type: "), 
                html.Span(node.get('type', 'unknown').replace('_', ' ').title())
            ]),
            html.P([
                html.Strong("Pipeline Layer: "), 
                html.Span(f"Layer {node.get('layer', 'Unknown')}")
            ]),
            html.Hr(),
            html.H6("Related Data Connections:"),
            html.Ul([
                html.Li(f"{edge['source']} → {edge['target']} (Source: {edge['source_type']}, Target: {edge['target_type']})")
                for edge in related_edges[:5]  # Show first 5 connections
            ]) if related_edges else html.P("No connections found", className="text-muted")
        ])
    ], className="mt-3")

# Callback for updating comparison table based on selected date
@app.callback(
    Output('comparison-table-container', 'children'),
    Input('comparison-date-selector', 'value')
)
def update_comparison_table(selected_date):
    if selected_date is None:
        return html.Div("No date selected")
    return create_bronze_silver_comparison(selected_date)

# Callback for data contract tabs
@app.callback(
    Output('data-contract-content', 'children'),
    Input('data-contract-tabs', 'value')
)
def render_data_contract_content(active_tab):
    return create_data_contract_content(active_tab)

# Performance optimization: Add caching for expensive computations
from functools import lru_cache

@lru_cache(maxsize=32)
def get_cached_segment_analysis():
    """Cache expensive segment analysis computations."""
    if data is None:
        return pd.DataFrame()
    
    return data.groupby('Segment').agg({
        'CustomerID': 'count',
        'Monetary': ['mean', 'sum', 'std'],
        'Frequency': ['mean', 'std'],
        'Recency': ['mean', 'std']
    }).round(2)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))