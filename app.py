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

# Load the data from Delta tables
data = load_data()
lineage_data = load_lineage_data()
dq_comparison, daily_counts, bronze_quality = load_quality_data()

if data is None:
    print("Error: Could not load RFM data from Databricks. Please check your DATABRICKS_WAREHOUSE_ID environment variable.")
    exit()

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
                        html.P("• Click on nodes or edges to select and highlight them", className="mb-1"),
                        html.P("• Hover over elements for detailed information", className="mb-0")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🏷️ Enhanced Legend"),
                    dbc.CardBody([
                        html.Div([
                            html.Span("🟠", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("S3 Sources", style={'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🟤", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("DLT Bronze Tables", style={'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("⚪", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("DLT Silver Tables", style={'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🟡", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("DLT Gold Tables", style={'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🟣", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("ML Schema Tables", style={'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("🔷", style={'fontSize': '20px', 'marginRight': '10px'}),
                            html.Span("Other Tables", style={'fontWeight': 'bold'})
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
    # Navigation
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("RFM Analytics", href="#", id="analytics-tab", active=True)),
                dbc.NavItem(dbc.NavLink("Data Quality", href="#", id="quality-tab", active=False)),
                dbc.NavItem(dbc.NavLink("Data Lineage", href="#", id="lineage-tab", active=False)),
            ], pills=True, className="mb-4")
        ], width=12)
    ]),
    
    # Page content - start with analytics page
    html.Div(id="page-content", children=create_analytics_page())
], fluid=True)

# Callbacks
@app.callback(
    [Output('analytics-tab', 'active'),
     Output('quality-tab', 'active'),
     Output('lineage-tab', 'active'),
     Output('page-content', 'children')],
    [Input('analytics-tab', 'n_clicks'),
     Input('quality-tab', 'n_clicks'),
     Input('lineage-tab', 'n_clicks')]
)
def update_page(analytics_clicks, quality_clicks, lineage_clicks):
    ctx = callback_context
    
    # If no button has been clicked yet, show analytics page
    if not ctx.triggered:
        return True, False, False, create_analytics_page()
    
    # Get which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'analytics-tab':
        return True, False, False, create_analytics_page()
    elif button_id == 'quality-tab':
        return False, True, False, create_quality_page()
    elif button_id == 'lineage-tab':
        return False, False, True, create_lineage_page()
    
    # Default to analytics page
    return True, False, False, create_analytics_page()

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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))