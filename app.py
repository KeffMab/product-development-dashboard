import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
import hashlib
import os
import warnings

warnings.filterwarnings('ignore')

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__,
 server=app.server               
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

server = app.server

# Custom CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AI-Solutions Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --main-blue: #1E88E5;
                --light-blue: #64B5F6;
                --dark-blue: #0D47A1;
                --white: #FFFFFF;
                --light-gray: #F5F5F5;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: var(--white);
                color: #333;
                overflow-y: hidden;
                max-height: 100vh;
            }

            .main-header {
                color: var(--dark-blue);
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
                text-align: center;
                padding: 10px;
            }

            .sub-header {
                color: var(--main-blue);
                font-size: 16px;
                font-weight: bold;
                margin-top: 5px;
                margin-bottom: 5px;
            }

            .metric-card {
                background-color: var(--white);
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 10px;
                text-align: center;
                height: 100%;
            }

            .metric-value {
                color: var(--dark-blue);
                font-size: 20px;
                font-weight: bold;
            }

            .metric-label {
                color: gray;
                font-size: 12px;
            }

            .nav-tabs .nav-link {
                color: var(--dark-blue);
                background-color: var(--light-gray);
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
                padding: 5px 10px;
                font-size: 14px;
            }

            .nav-tabs .nav-link.active {
                background-color: var(--main-blue);
                color: white;
            }

            .tab-content {
                padding: 10px;
                border: 1px solid #dee2e6;
                border-top: none;
                border-radius: 0 0 5px 5px;
            }

            .btn-primary {
                background-color: var(--main-blue);
                border-color: var(--main-blue);
                padding: 4px 8px;
                font-size: 12px;
            }

            .btn-primary:hover {
                background-color: var(--dark-blue);
                border-color: var(--dark-blue);
            }

            .password-container {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                background-color: var(--light-gray);
            }

            .chart-container {
                height: 250px;
                margin-bottom: 10px;
            }

            .filter-card {
                background-color: var(--light-gray);
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
            }

            .dashboard-container {
                max-height: calc(100vh - 120px);
                overflow-y: hidden;
            }

            .compact-dropdown .Select-control {
                height: 30px;
                min-height: 30px;
            }

            .compact-dropdown .Select-placeholder,
            .compact-dropdown .Select-value {
                line-height: 30px !important;
                padding-top: 0;
                padding-bottom: 0;
            }

            .compact-dropdown .Select-input {
                height: 30px;
            }

            /* Make all charts more compact */
            .js-plotly-plot {
                margin: 0 !important;
            }

            /* Compact date picker */
            .SingleDatePickerInput {
                height: 30px;
            }

            .DateInput_input {
                height: 30px;
                padding: 0 10px;
                font-size: 12px;
            }

            /* Compact form controls */
            .form-control {
                height: 30px;
                padding: 0 10px;
                font-size: 12px;
            }

            /* Compact row spacing */
            .row {
                margin-bottom: 10px;
            }

            /* Hide scrollbars but allow scrolling if absolutely necessary */
            .scrollable-if-needed {
                overflow-y: auto;
                scrollbar-width: none; /* Firefox */
                -ms-overflow-style: none; /* IE and Edge */
            }

            .scrollable-if-needed::-webkit-scrollbar {
                display: none; /* Chrome, Safari, Opera */
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


# Load data with robust cleaning
def load_data():
    df = pd.read_csv('data.csv')

    # Clean and validate data
    # Handle missing or invalid Order_ID values
    df['Order_ID'] = df['Order_ID'].fillna('unknown_order_id')
    df['Order_ID'] = df['Order_ID'].astype(str)

    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Handle missing dates
    df = df.dropna(subset=['Date'])

    # Create a full datetime column
    df['Time'] = df['Time'].fillna('00:00:00')
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], errors='coerce')

    # Extract month and year for time-based analysis
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Clean numeric columns
    numeric_cols = ['Quantity', 'Unit_Price', 'Total_Amount']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)

    # Calculate additional metrics
    df['Revenue'] = df['Total_Amount']
    # For profit calculation (assuming 30% profit margin)
    df['Profit'] = df['Total_Amount'] * 0.3

    # Clean categorical columns
    categorical_cols = ['Country', 'Customer_Segment', 'Order_Status', 'referral', 'Product_Name']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].astype(str)

    # Ensure Status_Code is numeric
    df['Status_Code'] = pd.to_numeric(df['Status_Code'], errors='coerce')
    df['Status_Code'] = df['Status_Code'].fillna(0).astype(int)

    # Clean User_Agent
    df['User_Agent'] = df['User_Agent'].fillna('Unknown')
    df['User_Agent'] = df['User_Agent'].astype(str)

    return df


# Load data
df = load_data()

# Create Setswana names for salespeople with passwords
salespeople = {
    "Tshepiso": {"password": "pass1", "id": 1},
    "Kgomotso": {"password": "pass2", "id": 2},
    "Lesego": {"password": "pass3", "id": 3},
    "Tumelo": {"password": "pass4", "id": 4},
    "Boitumelo": {"password": "pass5", "id": 5}
}

# Admin credentials
admin_password = "admin123"

# Marketing credentials
marketing_password = "marketing123"


# Assign orders to salespeople based on Order_ID hash
def assign_salespeople(df):
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Use the hash of Order_ID to assign to a salesperson
    for index, row in df_copy.iterrows():
        # Ensure Order_ID is a string
        order_id = str(row['Order_ID'])
        hash_val = int(hashlib.md5(order_id.encode()).hexdigest(), 16)
        salesperson_id = (hash_val % 5) + 1

        # Find the salesperson name with this ID
        salesperson_name = next(name for name, data in salespeople.items() if data["id"] == salesperson_id)
        df_copy.at[index, 'Salesperson'] = salesperson_name

    return df_copy


# Assign salespeople
df = assign_salespeople(df)


# Format currency values
def format_currency(value):
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.1f}K"
    else:
        return f"${value:.2f}"


# Calculate KPIs
def calculate_kpis(filtered_df):
    if filtered_df.empty:
        return {
            'Total Revenue': 0,
            'Total Orders': 0,
            'Average Order Value': 0,
            'Total Profit': 0,
            'Conversion Rate': 0,
            'Sales Growth': 0,
            'Customer Retention': 0,
            'Avg Items Per Order': 0
        }

    total_revenue = filtered_df['Revenue'].sum()
    total_orders = len(filtered_df)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    total_profit = filtered_df['Profit'].sum()
    conversion_rate = len(filtered_df[filtered_df['Status_Code'] == 200]) / len(filtered_df) * 100 if len(
        filtered_df) > 0 else 0

    # Additional KPIs
    customer_retention = len(filtered_df[filtered_df['Customer_Segment'] == 'Returning']) / len(
        filtered_df) * 100 if len(filtered_df) > 0 else 0
    avg_items_per_order = filtered_df['Quantity'].mean()

    # Calculate sales growth (comparing current period to previous period)
    if 'Date' in filtered_df.columns and not filtered_df.empty:
        mid_date = filtered_df['Date'].min() + (filtered_df['Date'].max() - filtered_df['Date'].min()) / 2
        current_period = filtered_df[filtered_df['Date'] >= mid_date]
        previous_period = filtered_df[filtered_df['Date'] < mid_date]

        current_revenue = current_period['Revenue'].sum()
        previous_revenue = previous_period['Revenue'].sum()

        sales_growth = ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
    else:
        sales_growth = 0

    return {
        'Total Revenue': total_revenue,
        'Total Orders': total_orders,
        'Average Order Value': avg_order_value,
        'Total Profit': total_profit,
        'Conversion Rate': conversion_rate,
        'Sales Growth': sales_growth,
        'Customer Retention': customer_retention,
        'Avg Items Per Order': avg_items_per_order
    }


# AI Predictions
def predict_future_sales(filtered_df):
    if filtered_df.empty:
        return np.zeros(30)

    # Group by date and calculate daily sales
    daily_sales = filtered_df.groupby(filtered_df['Date'].dt.date)['Revenue'].sum().reset_index()
    daily_sales['Day'] = range(len(daily_sales))

    if len(daily_sales) < 2:
        return np.zeros(30)

    # Prepare data for prediction
    X = daily_sales[['Day']].values
    y = daily_sales['Revenue'].values

    # Train a simple model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 30 days
    future_days = np.array(range(len(daily_sales), len(daily_sales) + 30)).reshape(-1, 1)
    future_sales = model.predict(future_days)

    # Ensure no negative predictions
    future_sales = np.maximum(future_sales, 0)

    return future_sales


def predict_customer_behavior(filtered_df):
    if filtered_df.empty:
        return pd.DataFrame(columns=['Customer_Segment', 'Revenue', 'Order_Count', 'Avg_Quantity', 'Avg_Order_Value'])

    # Group by customer segment and calculate metrics
    segment_metrics = filtered_df.groupby('Customer_Segment').agg({
        'Revenue': 'sum',
        'Order_ID': 'count',
        'Quantity': 'mean'
    }).reset_index()

    segment_metrics.rename(columns={
        'Order_ID': 'Order_Count',
        'Quantity': 'Avg_Quantity'
    }, inplace=True)

    # Calculate average order value
    segment_metrics['Avg_Order_Value'] = segment_metrics['Revenue'] / segment_metrics['Order_Count']

    return segment_metrics


# Marketing metrics
def calculate_marketing_metrics(filtered_df):
    if filtered_df.empty:
        return {
            'unique_visitors': 0,
            'traffic_sources': pd.DataFrame(columns=['Source', 'Count', 'Percentage']),
            'bounce_rate': 0,
            'conversion_rate': 0,
            'revenue_generated': 0,
            'device_usage': pd.DataFrame(columns=['Device', 'Count', 'Percentage'])
        }

    # 1. Page Traffic (Unique Visitors)
    unique_visitors = filtered_df['IP_Address'].nunique()

    # 2. Traffic Sources
    traffic_sources = filtered_df['referral'].value_counts().reset_index()
    traffic_sources.columns = ['Source', 'Count']
    traffic_sources['Percentage'] = (traffic_sources['Count'] / traffic_sources['Count'].sum() * 100).round(1)

    # 3. Bounce Rate (Single-page visits / Total visits)
    # Group by IP to find single-page visits
    visits_per_ip = filtered_df.groupby('IP_Address').size().reset_index(name='visit_count')
    single_page_visits = len(visits_per_ip[visits_per_ip['visit_count'] == 1])
    bounce_rate = (single_page_visits / len(visits_per_ip) * 100) if len(visits_per_ip) > 0 else 0

    # 4. Conversion Rate (Completed orders / Total visits)
    completed_orders = len(filtered_df[filtered_df['Order_Status'] == 'Delivered'])
    conversion_rate = (completed_orders / unique_visitors * 100) if unique_visitors > 0 else 0

    # 5. Revenue Generated
    revenue_generated = filtered_df['Revenue'].sum()

    # 6. Device Usage
    # Extract device info from User_Agent
    filtered_df['Device'] = filtered_df['User_Agent'].apply(
        lambda x: 'Mobile' if any(
            mobile in x for mobile in ['Mobile', 'Android', 'iPhone', 'iPad', 'iPod']) else 'Desktop'
    )
    device_usage = filtered_df['Device'].value_counts().reset_index()
    device_usage.columns = ['Device', 'Count']
    device_usage['Percentage'] = (device_usage['Count'] / device_usage['Count'].sum() * 100).round(1)

    return {
        'unique_visitors': unique_visitors,
        'traffic_sources': traffic_sources,
        'bounce_rate': bounce_rate,
        'conversion_rate': conversion_rate,
        'revenue_generated': revenue_generated,
        'device_usage': device_usage
    }


# Create compact visualizations
def create_revenue_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()

    # Group by date and calculate daily revenue
    daily_revenue = filtered_df.groupby(filtered_df['Date'].dt.date)['Revenue'].sum().reset_index()

    # Create interactive line chart
    fig = px.line(
        daily_revenue,
        x='Date',
        y='Revenue',
        title='Daily Revenue',
        labels={'Revenue': 'Revenue ($)', 'Date': 'Date'},
        template='plotly_white'
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        xaxis_title_font_size=10,
        yaxis_title_font_size=10,
        legend_font_size=8
    )

    return fig


def create_product_performance_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()

    # Group by product and calculate metrics
    product_performance = filtered_df.groupby('Product_Name').agg({
        'Revenue': 'sum',
        'Order_ID': 'count',
        'Quantity': 'sum'
    }).reset_index()

    product_performance.rename(columns={
        'Order_ID': 'Order_Count',
        'Quantity': 'Units_Sold'
    }, inplace=True)

    # Sort by revenue
    product_performance = product_performance.sort_values('Revenue', ascending=False)

    # Create interactive bar chart
    fig = px.bar(
        product_performance,
        x='Product_Name',
        y='Revenue',
        color='Units_Sold',
        title='Product Performance',
        labels={'Revenue': 'Revenue ($)', 'Product_Name': 'Product', 'Units_Sold': 'Units Sold'},
        template='plotly_white',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        xaxis_title_font_size=10,
        yaxis_title_font_size=10,
        legend_font_size=8
    )

    return fig


def create_geographic_sales_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()

    # Group by country and calculate total revenue
    country_sales = filtered_df.groupby('Country')['Revenue'].sum().reset_index()

    # Sort by revenue
    country_sales = country_sales.sort_values('Revenue', ascending=False)

    # Create interactive choropleth map
    fig = px.choropleth(
        country_sales,
        locations='Country',
        locationmode='country names',
        color='Revenue',
        title='Geographic Sales',
        color_continuous_scale='Blues',
        template='plotly_white'
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )

    return fig


def create_customer_segment_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()

    # Group by customer segment and calculate total revenue
    segment_revenue = filtered_df.groupby('Customer_Segment')['Revenue'].sum().reset_index()

    # Create interactive pie chart
    fig = px.pie(
        segment_revenue,
        values='Revenue',
        names='Customer_Segment',
        title='Revenue by Segment',
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Blues
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        legend_font_size=8
    )

    fig.update_traces(textinfo='percent', textfont_size=8)

    return fig


def create_time_based_activity_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()

    # Extract hour from time
    filtered_df['Hour'] = pd.to_datetime(filtered_df['Time']).dt.hour

    # Group by hour and count activities
    hourly_activity = filtered_df.groupby('Hour').size().reset_index(name='Activity_Count')

    # Create interactive line chart
    fig = px.line(
        hourly_activity,
        x='Hour',
        y='Activity_Count',
        title='Activity by Hour',
        labels={'Activity_Count': 'Activities', 'Hour': 'Hour of Day'},
        template='plotly_white'
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        xaxis_title_font_size=10,
        yaxis_title_font_size=10,
        xaxis=dict(tickmode='linear', tick0=0, dtick=4)
    )

    return fig


def create_sales_prediction_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()

    # Get future sales predictions
    future_sales = predict_future_sales(filtered_df)

    # Create date range for predictions
    last_date = filtered_df['Date'].max() if not filtered_df.empty else datetime.now()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    # Create dataframe for visualization
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Revenue': future_sales
    })

    # Group historical data by date
    historical_sales = filtered_df.groupby(filtered_df['Date'].dt.date)['Revenue'].sum().reset_index()
    historical_sales['Date'] = pd.to_datetime(historical_sales['Date'])

    # Create interactive line chart
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_sales['Date'],
        y=historical_sales['Revenue'],
        mode='lines',
        name='Historical',
        line=dict(color='#1E88E5', width=2)
    ))

    # Add prediction data
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted_Revenue'],
        mode='lines',
        name='Predicted',
        line=dict(color='#0D47A1', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Revenue Forecast',
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        xaxis_title_font_size=10,
        yaxis_title_font_size=10,
        legend_font_size=8,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def create_customer_behavior_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()

    # Get customer behavior predictions
    segment_metrics = predict_customer_behavior(filtered_df)

    # Create interactive bar chart
    fig = px.bar(
        segment_metrics,
        x='Customer_Segment',
        y='Avg_Order_Value',
        color='Revenue',
        title='Customer Analysis',
        labels={
            'Customer_Segment': 'Segment',
            'Avg_Order_Value': 'Avg Order ($)',
            'Revenue': 'Revenue ($)'
        },
        template='plotly_white',
        color_continuous_scale='Blues',
        text='Order_Count'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside', textfont_size=8)

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        xaxis_title_font_size=10,
        yaxis_title_font_size=10,
        legend_font_size=8
    )

    return fig


def create_traffic_sources_chart(traffic_sources):
    if traffic_sources.empty:
        return go.Figure()

    fig = px.pie(
        traffic_sources,
        values='Count',
        names='Source',
        title='Traffic Sources',
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Blues,
        hover_data=['Percentage']
    )

    fig.update_traces(textinfo='percent', textfont_size=8)

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        legend_font_size=8
    )

    return fig


def create_device_usage_chart(device_usage):
    if device_usage.empty:
        return go.Figure()

    fig = px.pie(
        device_usage,
        values='Count',
        names='Device',
        title='Device Usage',
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Blues,
        hover_data=['Percentage']
    )

    fig.update_traces(textinfo='percent', textfont_size=8)

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12,
        legend_font_size=8
    )

    return fig


def create_bounce_rate_gauge(bounce_rate):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bounce_rate,
        title={'text': "Bounce Rate"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0D47A1"},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#0D47A1",
            'steps': [
                {'range': [0, 30], 'color': '#64B5F6'},
                {'range': [30, 70], 'color': '#1E88E5'},
                {'range': [70, 100], 'color': '#0D47A1'}
            ]
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12
    )

    return fig


def create_conversion_rate_gauge(conversion_rate):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conversion_rate,
        title={'text': "Conversion Rate"},
        gauge={
            'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "#0D47A1"},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#0D47A1",
            'steps': [
                {'range': [0, 5], 'color': '#64B5F6'},
                {'range': [5, 10], 'color': '#1E88E5'},
                {'range': [10, 20], 'color': '#0D47A1'}
            ]
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12
    )

    return fig


# Create KPI cards
def create_kpi_card(title, value, prefix="", suffix=""):
    formatted_value = f"{prefix}{value}{suffix}"

    return dbc.Card(
        dbc.CardBody([
            html.Div(formatted_value, className="metric-value"),
            html.Div(title, className="metric-label")
        ]),
        className="metric-card mb-2"
    )


# Create progress bar
def create_progress_bar(value, max_value, title):
    progress_percentage = min(value / max_value * 100, 100)

    return html.Div([
        html.Div(title, className="sub-header"),
        dbc.Progress(value=progress_percentage, color="primary", className="mb-1", style={"height": "10px"}),
        html.Div(f"{format_currency(value)} of {format_currency(max_value)} ({progress_percentage:.1f}%)",
                 style={"fontSize": "10px"})
    ])


# Create gauge chart
def create_profit_gauge(profit, target):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=profit,
        number={'prefix': "$", 'valueformat': ",.1f", 'suffix': "K" if profit < 1e6 else "M"},
        title={'text': "Total Profit"},
        gauge={
            'axis': {'range': [0, target * 0.4], 'tickwidth': 1, 'tickcolor': "#0D47A1"},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#0D47A1",
            'steps': [
                {'range': [0, target * 0.1], 'color': '#64B5F6'},
                {'range': [target * 0.1, target * 0.2], 'color': '#1E88E5'},
                {'range': [target * 0.2, target * 0.4], 'color': '#0D47A1'}
            ]
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12
    )

    return fig


# Create compact date filter
def create_date_filter(id_prefix):
    return dbc.Card(
        dbc.CardBody([
            html.Div("Date Range", className="sub-header"),
            dbc.Row([
                dbc.Col([
                    html.Label("Start", style={"fontSize": "10px"}),
                    dcc.DatePickerSingle(
                        id=f'{id_prefix}-start-date',
                        min_date_allowed=df['Date'].min().date(),
                        max_date_allowed=df['Date'].max().date(),
                        initial_visible_month=df['Date'].min().date(),
                        date=df['Date'].min().date(),
                        className="mb-1",
                        style={"fontSize": "10px"}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("End", style={"fontSize": "10px"}),
                    dcc.DatePickerSingle(
                        id=f'{id_prefix}-end-date',
                        min_date_allowed=df['Date'].min().date(),
                        max_date_allowed=df['Date'].max().date(),
                        initial_visible_month=df['Date'].max().date(),
                        date=df['Date'].max().date(),
                        className="mb-1",
                        style={"fontSize": "10px"}
                    )
                ], width=6)
            ])
        ]),
        className="filter-card"
    )


# Create country filter
def create_country_filter(id_prefix):
    countries = sorted(df['Country'].unique())

    return dbc.Card(
        dbc.CardBody([
            html.Div("Country", className="sub-header"),
            dcc.Dropdown(
                id=f'{id_prefix}-country-filter',
                options=[{'label': country, 'value': country} for country in countries],
                value=countries,
                multi=True,
                className="mb-1 compact-dropdown",
                style={"fontSize": "10px"}
            )
        ]),
        className="filter-card"
    )


# Create product filter
def create_product_filter(id_prefix):
    products = sorted(df['Product_Name'].unique())

    return dbc.Card(
        dbc.CardBody([
            html.Div("Product", className="sub-header"),
            dcc.Dropdown(
                id=f'{id_prefix}-product-filter',
                options=[{'label': product, 'value': product} for product in products],
                value=products,
                multi=True,
                className="mb-1 compact-dropdown",
                style={"fontSize": "10px"}
            )
        ]),
        className="filter-card"
    )


# Create referral filter
def create_referral_filter(id_prefix):
    referrals = sorted(df['referral'].unique())

    return dbc.Card(
        dbc.CardBody([
            html.Div("Referral Source", className="sub-header"),
            dcc.Dropdown(
                id=f'{id_prefix}-referral-filter',
                options=[{'label': referral, 'value': referral} for referral in referrals],
                value=referrals,
                multi=True,
                className="mb-1 compact-dropdown",
                style={"fontSize": "10px"}
            )
        ]),
        className="filter-card"
    )


# Create login form
def create_login_form(id_prefix, title):
    return html.Div([
        html.Div(title, className="sub-header"),
        dbc.Input(
            id=f"{id_prefix}-password",
            type="password",
            placeholder="Enter password",
            className="mb-2",
            style={"fontSize": "12px"}
        ),
        dbc.Button("Login", id=f"{id_prefix}-login-button", color="primary", className="mb-2")
    ], className="password-container")


# Create salesperson buttons
def create_salesperson_buttons():
    buttons = []

    for name in salespeople.keys():
        buttons.append(
            dbc.Button(
                f"{name}",
                id={'type': 'salesperson-button', 'name': name},
                color="primary",
                className="mb-1 me-1",
                style={"fontSize": "12px"}
            )
        )

    return html.Div(buttons)


# App layout
app.layout = html.Div([
    html.Div("AI-Solutions Analytics Dashboard", className="main-header"),

    # Store components for state management
    dcc.Store(id='authentication-store', data={'authenticated': False, 'user_type': None, 'current_user': None}),
    dcc.Store(id='selected-salesperson-store', data=None),

    # Tabs for navigation
    dbc.Tabs([
        # Home tab
        dbc.Tab(label="Home", children=[
            html.Div("Welcome to AI-Solutions Analytics Dashboard", className="sub-header mt-2"),

            html.P("""
            This dashboard provides analytics for AI-Solutions, helping to improve marketing and sales strategies through data-backed insights.
            """, style={"fontSize": "12px"}),

            html.Div("Overview Metrics", className="sub-header mt-2"),

            # KPI cards
            dbc.Row([
                dbc.Col(create_kpi_card("Total Revenue", format_currency(df['Revenue'].sum())), width=4),
                dbc.Col(create_kpi_card("Total Orders", f"{len(df):,}"), width=4),
                dbc.Col(create_kpi_card("Total Profit", format_currency(df['Profit'].sum())), width=4)
            ])
        ], tab_id="home-tab"),

        # Salespeople tab
        dbc.Tab(label="Salespeople", children=[
            html.Div("Salespeople Performance", className="sub-header mt-2"),

            # Salesperson selection
            html.Div(id="salesperson-selection-container", children=[
                dbc.Row([
                    dbc.Col(create_salesperson_buttons(), width=12)
                ])
            ]),

            # Password field (initially hidden)
            html.Div(id="salesperson-password-container"),

            # Salesperson dashboard (initially hidden)
            html.Div(id="salesperson-dashboard-container")
        ], tab_id="salespeople-tab"),

        # Admin tab
        dbc.Tab(label="Admin", children=[
            html.Div("Admin Dashboard", className="sub-header mt-2"),

            # Admin login form
            html.Div(id="admin-login-container", children=[
                create_login_form("admin", "Admin Login")
            ]),

            # Admin dashboard (initially hidden)
            html.Div(id="admin-dashboard-container")
        ], tab_id="admin-tab"),

        # Marketing tab
        dbc.Tab(label="Marketing", children=[
            html.Div("Marketing Dashboard", className="sub-header mt-2"),

            # Marketing login form
            html.Div(id="marketing-login-container", children=[
                create_login_form("marketing", "Marketing Login")
            ]),

            # Marketing dashboard (initially hidden)
            html.Div(id="marketing-dashboard-container")
        ], tab_id="marketing-tab")
    ], id="tabs")
])


# Callback for salesperson button click
@app.callback(
    Output("salesperson-password-container", "children"),
    Output("selected-salesperson-store", "data"),
    Input({"type": "salesperson-button", "name": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def show_salesperson_password(n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    salesperson_name = eval(button_id)["name"]

    password_form = html.Div([
        html.Div(f"Enter password for {salesperson_name}", className="sub-header"),
        dbc.Input(
            id="salesperson-password",
            type="password",
            placeholder="Enter password",
            className="mb-2",
            style={"fontSize": "12px"}
        ),
        dbc.Button("Login", id="salesperson-login-button", color="primary", className="mb-2")
    ], className="password-container")

    return password_form, salesperson_name


# Callback for salesperson login
@app.callback(
    Output("authentication-store", "data", allow_duplicate=True),
    Input("salesperson-login-button", "n_clicks"),
    State("salesperson-password", "value"),
    State("selected-salesperson-store", "data"),
    State("authentication-store", "data"),
    prevent_initial_call=True
)
def login_salesperson(n_clicks, password, selected_salesperson, auth_data):
    if not n_clicks or not password or not selected_salesperson:
        return dash.no_update

    if password == salespeople[selected_salesperson]["password"]:
        auth_data["authenticated"] = True
        auth_data["user_type"] = "salesperson"
        auth_data["current_user"] = selected_salesperson
        return auth_data

    return dash.no_update


# Callback for admin login
@app.callback(
    Output("authentication-store", "data", allow_duplicate=True),
    Input("admin-login-button", "n_clicks"),
    State("admin-password", "value"),
    State("authentication-store", "data"),
    prevent_initial_call=True
)
def login_admin(n_clicks, password, auth_data):
    if not n_clicks or not password:
        return dash.no_update

    if password == admin_password:
        auth_data["authenticated"] = True
        auth_data["user_type"] = "admin"
        auth_data["current_user"] = "admin"
        return auth_data

    return dash.no_update


# Callback for marketing login
@app.callback(
    Output("authentication-store", "data", allow_duplicate=True),
    Input("marketing-login-button", "n_clicks"),
    State("marketing-password", "value"),
    State("authentication-store", "data"),
    prevent_initial_call=True
)
def login_marketing(n_clicks, password, auth_data):
    if not n_clicks or not password:
        return dash.no_update

    if password == marketing_password:
        auth_data["authenticated"] = True
        auth_data["user_type"] = "marketing"
        auth_data["current_user"] = "marketing"
        return auth_data

    return dash.no_update


# Callback to update salesperson dashboard
@app.callback(
    Output("salesperson-dashboard-container", "children"),
    Input("authentication-store", "data"),
    prevent_initial_call=True
)
def update_salesperson_dashboard(auth_data):
    if not auth_data["authenticated"] or auth_data["user_type"] != "salesperson":
        return dash.no_update

    salesperson = auth_data["current_user"]

    # Filter data for this salesperson
    salesperson_df = df[df['Salesperson'] == salesperson]

    # Calculate KPIs for this salesperson
    salesperson_kpis = calculate_kpis(salesperson_df)

    # Create dashboard content
    dashboard = html.Div([
        html.Div(f"{salesperson}'s Performance Dashboard", className="sub-header mt-2"),

        # Filters in a single row
        dbc.Row([
            dbc.Col(create_date_filter("salesperson"), width=4),
            dbc.Col(create_country_filter("salesperson"), width=4),
            dbc.Col(create_product_filter("salesperson"), width=4)
        ], className="mb-2"),

        # KPI cards
        dbc.Row([
            dbc.Col(create_kpi_card("Total Revenue (TR)", format_currency(salesperson_kpis['Total Revenue'])), width=4),
            dbc.Col(create_kpi_card("Total Orders (TO)", f"{salesperson_kpis['Total Orders']:,}"), width=4),
            dbc.Col(create_kpi_card("Average Order Value", format_currency(salesperson_kpis['Average Order Value'])),
                    width=4)
        ], className="mb-2"),

        # Sales Target and Profit in a single row
        dbc.Row([
            dbc.Col([
                create_progress_bar(salesperson_kpis['Total Revenue'], 100000, "Sales Target Progress")
            ], width=6),
            dbc.Col([
                html.Div("Profit Made", className="sub-header"),
                dcc.Graph(
                    id="salesperson-profit-gauge",
                    figure=create_profit_gauge(salesperson_kpis['Total Profit'], 100000),
                    config={'displayModeBar': False}
                )
            ], width=6)
        ], className="mb-2"),

        # Charts in a 2x3 grid (2 rows, 3 columns)
        dbc.Row([
            # First row of charts
            dbc.Col([
                dcc.Graph(
                    id="salesperson-revenue-chart",
                    figure=create_revenue_chart(salesperson_df),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="salesperson-product-chart",
                    figure=create_product_performance_chart(salesperson_df),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="salesperson-geo-chart",
                    figure=create_geographic_sales_chart(salesperson_df),
                    className="chart-container"
                )
            ], width=4)
        ], className="mb-2"),

        # Second row of charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id="salesperson-segment-chart",
                    figure=create_customer_segment_chart(salesperson_df),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="salesperson-time-chart",
                    figure=create_time_based_activity_chart(salesperson_df),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="salesperson-prediction-chart",
                    figure=create_sales_prediction_chart(salesperson_df),
                    className="chart-container"
                )
            ], width=4)
        ])
    ], className="dashboard-container")

    return dashboard


# Callback to update admin dashboard
@app.callback(
    Output("admin-dashboard-container", "children"),
    Input("authentication-store", "data"),
    prevent_initial_call=True
)
def update_admin_dashboard(auth_data):
    if not auth_data["authenticated"] or auth_data["user_type"] != "admin":
        return dash.no_update

    # Calculate KPIs for all data
    overall_kpis = calculate_kpis(df)

    # Create dashboard content
    dashboard = html.Div([
        html.Div("Overall Performance Dashboard", className="sub-header mt-2"),

        # Filters in a single row
        dbc.Row([
            dbc.Col(create_date_filter("admin"), width=4),
            dbc.Col(create_country_filter("admin"), width=4),
            dbc.Col(create_product_filter("admin"), width=4)
        ], className="mb-2"),

        # KPI cards
        dbc.Row([
            dbc.Col(create_kpi_card("Total Revenue", format_currency(overall_kpis['Total Revenue'])), width=4),
            dbc.Col(create_kpi_card("Total Orders", f"{overall_kpis['Total Orders']:,}"), width=4),
            dbc.Col(create_kpi_card("Total Profit", format_currency(overall_kpis['Total Profit'])), width=4)
        ], className="mb-2"),

        # Sales Target and Profit in a single row
        dbc.Row([
            dbc.Col([
                create_progress_bar(overall_kpis['Total Revenue'], 100000, "Sales Target Progress")
            ], width=6),
            dbc.Col([
                html.Div("Profit Made", className="sub-header"),
                dcc.Graph(
                    id="admin-profit-gauge",
                    figure=create_profit_gauge(overall_kpis['Total Profit'], 100000),
                    config={'displayModeBar': False}
                )
            ], width=6)
        ], className="mb-2"),

        # Charts in a 2x3 grid (2 rows, 3 columns)
        dbc.Row([
            # First row of charts
            dbc.Col([
                dcc.Graph(
                    id="admin-team-performance",
                    figure=px.bar(
                        df.groupby('Salesperson').agg({
                            'Revenue': 'sum',
                            'Order_ID': 'count',
                            'Profit': 'sum'
                        }).reset_index().rename(columns={'Order_ID': 'Orders'}),
                        x='Salesperson',
                        y='Revenue',
                        color='Profit',
                        title='Team Performance',
                        labels={
                            'Salesperson': 'Salesperson',
                            'Revenue': 'Revenue ($)',
                            'Profit': 'Profit ($)'
                        },
                        template='plotly_white',
                        color_continuous_scale='Blues',
                        text='Orders'
                    ).update_traces(texttemplate='%{text}', textposition='outside', textfont_size=8).update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#0D47A1', size=10),
                        title_font_size=12
                    ),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="admin-revenue-chart",
                    figure=create_revenue_chart(df),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="admin-product-chart",
                    figure=create_product_performance_chart(df),
                    className="chart-container"
                )
            ], width=4)
        ], className="mb-2"),

        # Second row of charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id="admin-geo-chart",
                    figure=create_geographic_sales_chart(df),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="admin-segment-chart",
                    figure=create_customer_segment_chart(df),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="admin-prediction-chart",
                    figure=create_sales_prediction_chart(df),
                    className="chart-container"
                )
            ], width=4)
        ])
    ], className="dashboard-container")

    return dashboard


# Callback to update marketing dashboard
@app.callback(
    Output("marketing-dashboard-container", "children"),
    Input("authentication-store", "data"),
    prevent_initial_call=True
)
def update_marketing_dashboard(auth_data):
    if not auth_data["authenticated"] or auth_data["user_type"] != "marketing":
        return dash.no_update

    # Calculate marketing metrics
    marketing_metrics = calculate_marketing_metrics(df)

    # Create dashboard content
    dashboard = html.Div([
        html.Div("Marketing Performance Dashboard", className="sub-header mt-2"),

        # Filters in a single row
        dbc.Row([
            dbc.Col(create_date_filter("marketing"), width=4),
            dbc.Col(create_referral_filter("marketing"), width=4),
            dbc.Col(create_country_filter("marketing"), width=4)
        ], className="mb-2"),

        # KPI cards
        dbc.Row([
            dbc.Col(create_kpi_card("Unique Visitors", f"{marketing_metrics['unique_visitors']:,}"), width=4),
            dbc.Col(create_kpi_card("Bounce Rate", f"{marketing_metrics['bounce_rate']:.1f}%", suffix="%"), width=4),
            dbc.Col(create_kpi_card("Conversion Rate", f"{marketing_metrics['conversion_rate']:.1f}%", suffix="%"),
                    width=4)
        ], className="mb-2"),

        # Second row of KPIs
        dbc.Row([
            dbc.Col(create_kpi_card("Revenue Generated", format_currency(marketing_metrics['revenue_generated'])),
                    width=4),
            dbc.Col(
                create_kpi_card(
                    f"Top Traffic Source ({marketing_metrics['traffic_sources'].iloc[0]['Percentage']}%)" if not
                    marketing_metrics['traffic_sources'].empty else "Top Traffic Source",
                    marketing_metrics['traffic_sources'].iloc[0]['Source'] if not marketing_metrics[
                        'traffic_sources'].empty else "N/A"
                ),
                width=4
            ),
            dbc.Col(
                create_kpi_card(
                    f"Top Device ({marketing_metrics['device_usage'].iloc[0]['Percentage']}%)" if not marketing_metrics[
                        'device_usage'].empty else "Top Device",
                    marketing_metrics['device_usage'].iloc[0]['Device'] if not marketing_metrics[
                        'device_usage'].empty else "N/A"
                ),
                width=4
            )
        ], className="mb-2"),

        # Charts in a 2x3 grid (2 rows, 3 columns)
        dbc.Row([
            # First row of charts
            dbc.Col([
                dcc.Graph(
                    id="marketing-traffic-chart",
                    figure=create_traffic_sources_chart(marketing_metrics['traffic_sources']),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="marketing-device-chart",
                    figure=create_device_usage_chart(marketing_metrics['device_usage']),
                    className="chart-container"
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="marketing-referral-revenue-chart",
                    figure=px.bar(
                        df.groupby('referral')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False),
                        x='referral',
                        y='Revenue',
                        title='Revenue by Referral',
                        labels={'referral': 'Referral Source', 'Revenue': 'Revenue ($)'},
                        template='plotly_white',
                        color='Revenue',
                        color_continuous_scale='Blues'
                    ).update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#0D47A1', size=10),
                        title_font_size=12
                    ),
                    className="chart-container"
                )
            ], width=4)
        ], className="mb-2"),

        # Second row of charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id="marketing-bounce-rate-gauge",
                    figure=create_bounce_rate_gauge(marketing_metrics['bounce_rate']),
                    config={'displayModeBar': False}
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="marketing-conversion-rate-gauge",
                    figure=create_conversion_rate_gauge(marketing_metrics['conversion_rate']),
                    config={'displayModeBar': False}
                )
            ], width=4),
            dbc.Col([
                dcc.Graph(
                    id="marketing-revenue-time-chart",
                    figure=px.line(
                        df.groupby(df['Date'].dt.date)['Revenue'].sum().reset_index(),
                        x='Date',
                        y='Revenue',
                        title='Daily Revenue',
                        labels={'Revenue': 'Revenue ($)', 'Date': 'Date'},
                        template='plotly_white'
                    ).update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#0D47A1', size=10),
                        title_font_size=12
                    ),
                    className="chart-container"
                )
            ], width=4)
        ])
    ], className="dashboard-container")

    return dashboard


# Callback for salesperson dashboard filters
@app.callback(
    Output("salesperson-revenue-chart", "figure"),
    Output("salesperson-product-chart", "figure"),
    Output("salesperson-geo-chart", "figure"),
    Output("salesperson-segment-chart", "figure"),
    Output("salesperson-time-chart", "figure"),
    Output("salesperson-prediction-chart", "figure"),
    Output("salesperson-profit-gauge", "figure"),
    [Input("salesperson-start-date", "date"),
     Input("salesperson-end-date", "date"),
     Input("salesperson-country-filter", "value"),
     Input("salesperson-product-filter", "value")],
    State("authentication-store", "data"),
    prevent_initial_call=True
)
def update_salesperson_charts(start_date, end_date, countries, products, auth_data):
    if not auth_data["authenticated"] or auth_data["user_type"] != "salesperson":
        return [dash.no_update] * 7

    salesperson = auth_data["current_user"]

    # Convert string dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data for this salesperson
    filtered_df = df[df['Salesperson'] == salesperson]

    # Apply date filter
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

    # Apply country filter independently
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]

    # Apply product filter independently
    if products:
        filtered_df = filtered_df[filtered_df['Product_Name'].isin(products)]

    # Calculate KPIs for filtered data
    filtered_kpis = calculate_kpis(filtered_df)

    # Create charts
    revenue_chart = create_revenue_chart(filtered_df)
    product_chart = create_product_performance_chart(filtered_df)
    geo_chart = create_geographic_sales_chart(filtered_df)
    segment_chart = create_customer_segment_chart(filtered_df)
    time_chart = create_time_based_activity_chart(filtered_df)
    prediction_chart = create_sales_prediction_chart(filtered_df)
    profit_gauge = create_profit_gauge(filtered_kpis['Total Profit'], 100000)

    return revenue_chart, product_chart, geo_chart, segment_chart, time_chart, prediction_chart, profit_gauge


# Callback for admin dashboard filters
@app.callback(
    Output("admin-revenue-chart", "figure"),
    Output("admin-product-chart", "figure"),
    Output("admin-geo-chart", "figure"),
    Output("admin-segment-chart", "figure"),
    Output("admin-prediction-chart", "figure"),
    Output("admin-profit-gauge", "figure"),
    Output("admin-team-performance", "figure"),
    [Input("admin-start-date", "date"),
     Input("admin-end-date", "date"),
     Input("admin-country-filter", "value"),
     Input("admin-product-filter", "value")],
    prevent_initial_call=True
)
def update_admin_charts(start_date, end_date, countries, products):
    # Convert string dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Start with all data
    filtered_df = df.copy()

    # Apply date filter
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

    # Apply country filter independently
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]

    # Apply product filter independently
    if products:
        filtered_df = filtered_df[filtered_df['Product_Name'].isin(products)]

    # Calculate KPIs for filtered data
    filtered_kpis = calculate_kpis(filtered_df)

    # Create charts
    revenue_chart = create_revenue_chart(filtered_df)
    product_chart = create_product_performance_chart(filtered_df)
    geo_chart = create_geographic_sales_chart(filtered_df)
    segment_chart = create_customer_segment_chart(filtered_df)
    prediction_chart = create_sales_prediction_chart(filtered_df)
    profit_gauge = create_profit_gauge(filtered_kpis['Total Profit'], 100000)

    # Team performance chart
    team_performance = filtered_df.groupby('Salesperson').agg({
        'Revenue': 'sum',
        'Order_ID': 'count',
        'Profit': 'sum'
    }).reset_index()

    team_performance.rename(columns={
        'Order_ID': 'Orders'
    }, inplace=True)

    # Sort by revenue
    team_performance = team_performance.sort_values('Revenue', ascending=False)

    team_chart = px.bar(
        team_performance,
        x='Salesperson',
        y='Revenue',
        color='Profit',
        title='Team Performance',
        labels={
            'Salesperson': 'Salesperson',
            'Revenue': 'Revenue ($)',
            'Profit': 'Profit ($)'
        },
        template='plotly_white',
        color_continuous_scale='Blues',
        text='Orders'
    )

    team_chart.update_traces(texttemplate='%{text}', textposition='outside', textfont_size=8)

    team_chart.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12
    )

    return revenue_chart, product_chart, geo_chart, segment_chart, prediction_chart, profit_gauge, team_chart


# Callback for marketing dashboard filters
@app.callback(
    Output("marketing-traffic-chart", "figure"),
    Output("marketing-device-chart", "figure"),
    Output("marketing-referral-revenue-chart", "figure"),
    Output("marketing-bounce-rate-gauge", "figure"),
    Output("marketing-conversion-rate-gauge", "figure"),
    Output("marketing-revenue-time-chart", "figure"),
    [Input("marketing-start-date", "date"),
     Input("marketing-end-date", "date"),
     Input("marketing-country-filter", "value"),
     Input("marketing-referral-filter", "value")],
    prevent_initial_call=True
)
def update_marketing_charts(start_date, end_date, countries, referrals):
    # Convert string dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Start with all data
    filtered_df = df.copy()

    # Apply date filter
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

    # Apply country filter independently
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]

    # Apply referral filter independently
    if referrals:
        filtered_df = filtered_df[filtered_df['referral'].isin(referrals)]

    # Calculate marketing metrics
    marketing_metrics = calculate_marketing_metrics(filtered_df)

    # Create charts
    traffic_chart = create_traffic_sources_chart(marketing_metrics['traffic_sources'])
    device_chart = create_device_usage_chart(marketing_metrics['device_usage'])

    # Referral revenue chart
    referral_revenue = filtered_df.groupby('referral')['Revenue'].sum().reset_index()
    referral_revenue = referral_revenue.sort_values('Revenue', ascending=False)

    referral_chart = px.bar(
        referral_revenue,
        x='referral',
        y='Revenue',
        title='Revenue by Referral Source',
        labels={'referral': 'Referral Source', 'Revenue': 'Revenue ($)'},
        template='plotly_white',
        color='Revenue',
        color_continuous_scale='Blues'
    )

    referral_chart.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12
    )

    # Gauge charts
    bounce_gauge = create_bounce_rate_gauge(marketing_metrics['bounce_rate'])
    conversion_gauge = create_conversion_rate_gauge(marketing_metrics['conversion_rate'])

    # Revenue over time
    daily_revenue = filtered_df.groupby(filtered_df['Date'].dt.date)['Revenue'].sum().reset_index()

    revenue_time_chart = px.line(
        daily_revenue,
        x='Date',
        y='Revenue',
        title='Daily Revenue from Marketing',
        labels={'Revenue': 'Revenue ($)', 'Date': 'Date'},
        template='plotly_white'
    )

    revenue_time_chart.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#0D47A1', size=10),
        title_font_size=12
    )

    return traffic_chart, device_chart, referral_chart, bounce_gauge, conversion_gauge, revenue_time_chart


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
