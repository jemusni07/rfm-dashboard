import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_cytoscape as cyto
import os
import json

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
    
    return cyto.Cytoscape(
        id='architecture-flow-cytoscape',
        elements=elements,
        layout={
            'name': 'preset',
            'padding': 50,
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
        minZoom=0.2,
        maxZoom=2.0,
        wheelSensitivity=0.1
    )

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
    
    # Define colors for clusters - create a comprehensive color map
    unique_clusters = sorted(data['Cluster'].unique())
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_discrete_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
    
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
    
    # Use the same color mapping as the 3D scatter
    unique_clusters = sorted(data['Cluster'].unique())
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_discrete_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
    
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
    """Create monetary value analysis by cluster."""
    # Use the same color mapping as other charts
    unique_clusters = sorted(data['Cluster'].unique())
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_discrete_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
    
    # Convert Cluster to string to ensure consistent color mapping
    data_copy = data.copy()
    data_copy['Cluster_str'] = data_copy['Cluster'].astype(str)
    
    # Create string-based color map
    color_discrete_map_str = {str(k): v for k, v in color_discrete_map.items()}
    
    fig = px.box(
        data_copy, 
        x='Cluster_Description', 
        y='Monetary', 
        color='Cluster_str',
        color_discrete_map=color_discrete_map_str,
        title='Monetary Value Distribution by Segment',
        labels={'Cluster_str': 'Cluster'}
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig

# Landing page layout
def create_landing_page():
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
                                html.H6("🌟 Core Architecture", className="small fw-bold text-primary mb-2"),
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
                            ], width=6),
                            dbc.Col([
                                html.H6("🔘 Supporting Layers", className="small fw-bold text-muted mb-2"),
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
                            ], width=6)
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
                            ], width=4),
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
                            ], width=4),
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
                            ], width=4)
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
                            ], width=6),
                            dbc.Col([
                                html.H6("🔧 Data Processing Layer", className="text-success mb-2"),
                                html.Ul([
                                    html.Li("Multi-layered ETL pipeline using Delta Live Tables"),
                                    html.Li("Data quality expectations and validation"),
                                    html.Li("Automated schema evolution and data lineage"),
                                    html.Li("Error handling and data rescue capabilities")
                                ])
                            ], width=6)
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
                            ], width=6),
                            dbc.Col([
                                html.H6("📊 Analytics & Application Layer", className="text-danger mb-2"),
                                html.Ul([
                                    html.Li("RFM customer segmentation analysis"),
                                    html.Li("K-means clustering with intelligent outlier handling"),
                                    html.Li("Materialized customer segments for business use"),
                                    html.Li("Interactive dashboard for real-time insights")
                                ])
                            ], width=6)
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

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server  # For deployment

# Define the app layout with navigation
app.layout = dbc.Container([
    # Navigation and refresh controls
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Project Overview", href="#", id="landing-tab", active=True)),
                dbc.NavItem(dbc.NavLink("RFM Analytics", href="#", id="analytics-tab", active=False)),
                dbc.NavItem(dbc.NavLink("Data Quality", href="#", id="quality-tab", active=False)),
                dbc.NavItem(dbc.NavLink("Data Lineage", href="#", id="lineage-tab", active=False)),
                dbc.NavItem(dbc.NavLink("Data Architecture", href="#", id="architecture-tab", active=False)),
            ], pills=True, className="mb-4")
        ], width=10),
        dbc.Col([
            dbc.Button(
                "🔄 Refresh Data", 
                id="refresh-button", 
                color="primary", 
                size="sm",
                className="mb-4"
            )
        ], width=2, className="text-end")
    ]),
    
    # Refresh status indicator
    dbc.Row([
        dbc.Col([
            html.Div(id="refresh-status")
        ], width=12)
    ]),
    
    # Page content - start with landing page
    html.Div(id="page-content", children=[])
], fluid=True)

# Callbacks
@app.callback(
    [Output('refresh-status', 'children'),
     Output('landing-tab', 'active'),
     Output('analytics-tab', 'active'),
     Output('quality-tab', 'active'),
     Output('lineage-tab', 'active'),
     Output('architecture-tab', 'active'),
     Output('page-content', 'children')],
    [Input('refresh-button', 'n_clicks'),
     Input('landing-tab', 'n_clicks'),
     Input('analytics-tab', 'n_clicks'),
     Input('quality-tab', 'n_clicks'),
     Input('lineage-tab', 'n_clicks'),
     Input('architecture-tab', 'n_clicks')]
)
def update_page(refresh_clicks, landing_clicks, analytics_clicks, quality_clicks, lineage_clicks, architecture_clicks):
    ctx = callback_context
    
    refresh_status = ""
    
    # Handle refresh button click
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'refresh-button.n_clicks':
        refresh_all_data()
        refresh_status = dbc.Alert("✅ Data refreshed successfully!", color="success", dismissable=True, duration=4000)
    
    # If no button has been clicked yet, show landing page
    if not ctx.triggered:
        return refresh_status, True, False, False, False, False, create_landing_page()
    
    # Get which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'landing-tab':
        return refresh_status, True, False, False, False, False, create_landing_page()
    elif button_id == 'analytics-tab':
        return refresh_status, False, True, False, False, False, create_analytics_page()
    elif button_id == 'quality-tab':
        return refresh_status, False, False, True, False, False, create_quality_page()
    elif button_id == 'lineage-tab':
        return refresh_status, False, False, False, True, False, create_lineage_page()
    elif button_id == 'architecture-tab':
        return refresh_status, False, False, False, False, True, create_architecture_page()
    elif button_id == 'refresh-button':
        # Keep current page active after refresh
        return refresh_status, True, False, False, False, False, create_landing_page()
    
    # Default to landing page
    return refresh_status, True, False, False, False, False, create_landing_page()

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
        ], color=color, inverse=True),
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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))