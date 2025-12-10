# management_app.py - 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import json
import io

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS FOR BETTER UI ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE INITIALIZATION ==========
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# ========== HELPER FUNCTIONS ==========
def generate_sample_data():
    """Generate comprehensive sample marketing data"""
    np.random.seed(42)
    
    # Generate dates for last 90 days
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    data = {
        'Date': dates,
        'Channel': np.random.choice(['Email', 'Social Media', 'PPC', 'Organic Search', 'Direct', 'Affiliate'], 90),
        'Campaign': np.random.choice(['Winter Sale', 'Spring Launch', 'Summer Promo', 'Back to School', 'Black Friday'], 90),
        'Region': np.random.choice(['North America', 'Europe', 'Asia', 'South America', 'Australia'], 90),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home Goods', 'Books', 'Beauty'], 90),
        'Clicks': np.random.randint(100, 5000, 90),
        'Impressions': np.random.randint(1000, 20000, 90),
        'Conversions': np.random.randint(10, 500, 90),
        'Revenue': np.random.randint(1000, 50000, 90),
        'Cost': np.random.randint(500, 25000, 90),
        'Visitors': np.random.randint(500, 10000, 90),
        'Bounce_Rate': np.random.uniform(20, 70, 90),
        'Session_Duration': np.random.uniform(60, 600, 90)
    }
    
    df = pd.DataFrame(data)
    df['CTR'] = (df['Clicks'] / df['Impressions'] * 100).round(2)
    df['Conversion_Rate'] = (df['Conversions'] / df['Clicks'] * 100).round(2)
    df['ROI'] = ((df['Revenue'] - df['Cost']) / df['Cost'] * 100).round(2)
    df['CPA'] = (df['Cost'] / df['Conversions']).round(2)
    
    return df

def analyze_data(df):
    """Perform comprehensive data analysis"""
    if df is None or df.empty:
        return {}
    
    analysis = {}
    
    # Basic metrics
    analysis['total_revenue'] = df['Revenue'].sum()
    analysis['total_cost'] = df['Cost'].sum()
    analysis['total_profit'] = analysis['total_revenue'] - analysis['total_cost']
    analysis['overall_roi'] = (analysis['total_profit'] / analysis['total_cost'] * 100) if analysis['total_cost'] > 0 else 0
    
    # Channel performance
    channel_stats = df.groupby('Channel').agg({
        'Revenue': 'sum',
        'Cost': 'sum',
        'Conversions': 'sum',
        'Clicks': 'sum'
    }).reset_index()
    
    channel_stats['ROI'] = ((channel_stats['Revenue'] - channel_stats['Cost']) / channel_stats['Cost'] * 100).round(2)
    channel_stats['CPA'] = (channel_stats['Cost'] / channel_stats['Conversions']).round(2)
    channel_stats['Conversion_Rate'] = (channel_stats['Conversions'] / channel_stats['Clicks'] * 100).round(2)
    
    analysis['channel_performance'] = channel_stats
    analysis['top_channel'] = channel_stats.loc[channel_stats['ROI'].idxmax(), 'Channel'] if not channel_stats.empty else 'N/A'
    
    # Time-based trends
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    
    weekly_trend = df.groupby('Week').agg({'Revenue': 'sum', 'Cost': 'sum'}).reset_index()
    analysis['weekly_trend'] = weekly_trend
    
    # Campaign analysis
    campaign_stats = df.groupby('Campaign').agg({
        'Revenue': 'sum',
        'ROI': 'mean'
    }).reset_index()
    analysis['campaign_performance'] = campaign_stats
    
    # Region analysis
    region_stats = df.groupby('Region').agg({
        'Revenue': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    analysis['region_performance'] = region_stats
    
    return analysis

def generate_marketing_insights(analysis):
    """Generate actionable insights from analysis"""
    insights = []
    
    if not analysis:
        return ["No data available for insights"]
    
    # ROI insights
    top_channel = analysis.get('top_channel', 'N/A')
    overall_roi = analysis.get('overall_roi', 0)
    
    if overall_roi > 100:
        insights.append(f"🎯 **Excellent Performance**: Overall ROI is {overall_roi:.1f}% - marketing efforts are highly effective")
    elif overall_roi > 0:
        insights.append(f"📈 **Positive Performance**: Overall ROI is {overall_roi:.1f}% - consider scaling successful channels")
    else:
        insights.append(f"⚠️ **Attention Needed**: Overall ROI is {overall_roi:.1f}% - review and optimize marketing strategy")
    
    if top_channel != 'N/A':
        insights.append(f"🏆 **Top Channel**: {top_channel} delivers the highest ROI - consider allocating more budget")
    
    # Channel-specific insights
    if 'channel_performance' in analysis:
        channel_df = analysis['channel_performance']
        for _, row in channel_df.iterrows():
            if row['ROI'] < 0:
                insights.append(f"🔍 **Optimize {row['Channel']}**: Negative ROI of {row['ROI']:.1f}% - review strategy or reduce spend")
            elif row['CPA'] > 100:
                insights.append(f"💰 **High CPA Alert**: {row['Channel']} has CPA of ${row['CPA']:.2f} - optimize targeting")
    
    # Time-based insights
    if 'weekly_trend' in analysis:
        weekly_df = analysis['weekly_trend']
        if len(weekly_df) >= 2:
            last_week = weekly_df.iloc[-1]['Revenue']
            prev_week = weekly_df.iloc[-2]['Revenue']
            if last_week > prev_week * 1.2:
                insights.append(f"🚀 **Growth Spike**: Revenue increased by {((last_week/prev_week)-1)*100:.1f}% week-over-week")
    
    # Add general best practices
    insights.extend([
        "💡 **Best Practice**: A/B test email subject lines to improve open rates",
        "💡 **Best Practice**: Retarget website visitors within 24 hours for higher conversion",
        "💡 **Best Practice**: Use video content on social media for 3x higher engagement"
    ])
    
    return insights

def create_visualizations(df, analysis):
    """Create interactive visualizations"""
    visualizations = {}
    
    if df is None or df.empty:
        return visualizations
    
    # 1. Revenue vs Cost Over Time
    try:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['Date'], y=df['Revenue'], name='Revenue', 
                                  line=dict(color='green', width=2)))
        fig1.add_trace(go.Scatter(x=df['Date'], y=df['Cost'], name='Cost', 
                                  line=dict(color='red', width=2)))
        fig1.update_layout(title='Revenue vs Marketing Cost Over Time',
                          xaxis_title='Date',
                          yaxis_title='Amount ($)',
                          template='plotly_white')
        visualizations['revenue_cost_trend'] = fig1
    except:
        pass
    
    # 2. Channel Performance (Bar Chart)
    try:
        if 'channel_performance' in analysis:
            channel_df = analysis['channel_performance']
            fig2 = px.bar(channel_df, x='Channel', y='ROI',
                         color='ROI',
                         title='ROI by Marketing Channel',
                         color_continuous_scale='RdYlGn')
            fig2.update_layout(xaxis_title='Channel', yaxis_title='ROI (%)')
            visualizations['channel_roi'] = fig2
    except:
        pass
    
    # 3. Conversion Rate by Campaign
    try:
        campaign_ctr = df.groupby('Campaign').agg({
            'Conversion_Rate': 'mean',
            'Revenue': 'sum'
        }).reset_index()
        fig3 = px.scatter(campaign_ctr, x='Conversion_Rate', y='Revenue',
                         size='Revenue', color='Campaign',
                         title='Campaign Performance: Conversion Rate vs Revenue',
                         hover_name='Campaign')
        fig3.update_layout(xaxis_title='Average Conversion Rate (%)',
                          yaxis_title='Total Revenue ($)')
        visualizations['campaign_performance'] = fig3
    except:
        pass
    
    # 4. Regional Performance Map
    try:
        region_df = analysis.get('region_performance', pd.DataFrame())
        if not region_df.empty:
            fig4 = px.pie(region_df, values='Revenue', names='Region',
                         title='Revenue Distribution by Region',
                         hole=0.4)
            fig4.update_traces(textposition='inside', textinfo='percent+label')
            visualizations['region_distribution'] = fig4
    except:
        pass
    
    # 5. ROI Distribution
    try:
        fig5 = px.histogram(df, x='ROI', nbins=30,
                           title='Distribution of Campaign ROI',
                           labels={'ROI': 'Return on Investment (%)'})
        fig5.update_layout(bargap=0.1)
        visualizations['roi_distribution'] = fig5
    except:
        pass
    
    return visualizations

def handle_chat_query(query, df, analysis):
    """Handle natural language queries"""
    query_lower = query.lower()
    response = ""
    
    if df is None or df.empty:
        return "Please load data first to get insights."
    
    # Revenue queries
    if any(word in query_lower for word in ['revenue', 'sales', 'income']):
        total_rev = df['Revenue'].sum()
        avg_daily = df.groupby('Date')['Revenue'].sum().mean()
        response = f"💰 **Revenue Insights**:\n- Total Revenue: ${total_rev:,.0f}\n- Average Daily Revenue: ${avg_daily:,.0f}"
        if 'channel_performance' in analysis:
            top_channel = analysis['channel_performance'].nlargest(1, 'Revenue')
            if not top_channel.empty:
                response += f"\n- Top Revenue Channel: {top_channel.iloc[0]['Channel']} (${top_channel.iloc[0]['Revenue']:,.0f})"
    
    # ROI queries
    elif any(word in query_lower for word in ['roi', 'return', 'investment']):
        if 'overall_roi' in analysis:
            roi = analysis['overall_roi']
            response = f"📈 **ROI Analysis**:\n- Overall ROI: {roi:.1f}%"
            if roi > 100:
                response += "\n- 🎯 Excellent performance! Marketing is highly effective."
            elif roi > 0:
                response += "\n- 📊 Positive returns. Consider scaling successful campaigns."
            else:
                response += "\n- ⚠️ Needs optimization. Review underperforming channels."
    
    # Channel performance
    elif any(word in query_lower for word in ['channel', 'platform', 'medium']):
        if 'channel_performance' in analysis:
            channels = analysis['channel_performance'].sort_values('ROI', ascending=False)
            response = "📱 **Channel Performance**:\n"
            for _, row in channels.head(3).iterrows():
                response += f"- {row['Channel']}: {row['ROI']:.1f}% ROI, ${row['CPA']:.2f} CPA\n"
    
    # Campaign performance
    elif any(word in query_lower for word in ['campaign', 'promotion', 'offer']):
        if 'campaign_performance' in analysis:
            campaigns = analysis['campaign_performance'].sort_values('Revenue', ascending=False)
            response = "🎯 **Top Campaigns**:\n"
            for _, row in campaigns.head(3).iterrows():
                response += f"- {row['Campaign']}: ${row['Revenue']:,.0f} revenue, {row['ROI']:.1f}% ROI\n"
    
    # General analytics
    elif any(word in query_lower for word in ['analyze', 'insights', 'summary']):
        insights = generate_marketing_insights(analysis)
        response = "🔍 **Key Insights**:\n" + "\n".join([f"- {insight}" for insight in insights[:5]])
    
    # Default response
    else:
        response = f"🤖 I've analyzed your query about '{query}'. Based on your marketing data:\n"
        response += f"- Dataset has {len(df)} records across {df['Channel'].nunique()} channels\n"
        response += f"- Time period: {df['Date'].min().date()} to {df['Date'].max().date()}\n"
        response += "You can ask about:\n• Revenue and ROI\n• Channel performance\n• Campaign analysis\n• Regional insights"
    
    return response

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092658.png", width=80)
    st.title("📊 Dashboard Control")
    
    st.subheader("📁 Data Management")
    
    # Data Source Selection
    data_source = st.radio(
        "Choose Data Source:",
        ["Upload CSV File", "Use Sample Data", "Enter Data Manually"],
        index=1
    )
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your marketing data CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    elif data_source == "Use Sample Data":
        if st.button("Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                df = generate_sample_data()
                st.session_state.current_data = df
                st.session_state.data_loaded = True
                st.success(f"✅ Generated {len(df)} sample records")
    
    else:  # Manual Entry
        st.info("Manual entry coming soon. Use sample data for now.")
    
    st.divider()
    
    st.subheader("⚙️ Analysis Settings")
    
    analysis_type = st.multiselect(
        "Select Analysis Types:",
        ["Performance Metrics", "Channel Analysis", "Trend Analysis", "Regional Analysis", "ROI Optimization"],
        default=["Performance Metrics", "Channel Analysis"]
    )
    
    date_range = st.date_input(
        "Date Range (if applicable):",
        [datetime.now().date() - timedelta(days=30), datetime.now().date()]
    )
    
    st.divider()

if st.button("🔄 Force Show Chat Test"):
    st.session_state.data_loaded = True
    st.session_state.current_data = generate_sample_data()
    st.rerun()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📥 Export", use_container_width=True):
            if st.session_state.data_loaded:
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="marketing_data_export.csv",
                    mime="text/csv"
                )

# ========== MAIN CONTENT AREA ==========
st.markdown('<h1 class="main-header">📈 Marketing Analytics Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Transform your marketing data into actionable insights with AI-powered analysis.")

# Check if data is loaded
if not st.session_state.data_loaded:
    st.info("👈 Please load data from the sidebar to begin analysis.")
    
    # Show sample data preview
    with st.expander("🎯 What you can analyze:", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
            <h4>📱 Channel Performance</h4>
            • ROI by channel<br>
            • Conversion rates<br>
            • Cost per acquisition
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>💰 Revenue Insights</h4>
            • Revenue trends<br>
            • Profit margins<br>
            • ROI optimization
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card">
            <h4>🎯 Campaign Analysis</h4>
            • Campaign performance<br>
            • A/B test results<br>
            • Seasonal trends
            </div>
            """, unsafe_allow_html=True)
    
    st.stop()

# Data is loaded - proceed with analysis
df = st.session_state.current_data

# Perform analysis
with st.spinner("Analyzing data..."):
    analysis = analyze_data(df)
    st.session_state.analysis_results = analysis
    insights = generate_marketing_insights(analysis)
    visualizations = create_visualizations(df, analysis)

# ========== TOP METRICS DASHBOARD ==========
st.markdown('<h2 class="sub-header">📊 Performance Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
    <h3>${analysis.get('total_revenue', 0):,.0f}</h3>
    <p>Total Revenue</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    roi = analysis.get('overall_roi', 0)
    roi_color = "green" if roi > 0 else "red"
    st.markdown(f"""
    <div class="metric-card">
    <h3 style="color: {roi_color}">{roi:.1f}%</h3>
    <p>Overall ROI</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
    <h3>{analysis.get('channel_performance', pd.DataFrame()).shape[0]}</h3>
    <p>Channels Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    profit = analysis.get('total_profit', 0)
    st.markdown(f"""
    <div class="metric-card">
    <h3>${profit:,.0f}</h3>
    <p>Total Profit</p>
    </div>
    """, unsafe_allow_html=True)

# ========== INSIGHTS SECTION ==========
st.markdown('<h2 class="sub-header">💡 AI-Powered Insights</h2>', unsafe_allow_html=True)

for insight in insights[:5]:  # Show top 5 insights
    st.info(insight)

# ========== VISUALIZATIONS ==========
st.markdown('<h2 class="sub-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Channels", "Campaigns", "Regions"])

with tab1:
    if 'revenue_cost_trend' in visualizations:
        st.plotly_chart(visualizations['revenue_cost_trend'], use_container_width=True)
    else:
        st.info("Trend visualization not available")

with tab2:
    if 'channel_roi' in visualizations:
        st.plotly_chart(visualizations['channel_roi'], use_container_width=True)
    else:
        st.info("Channel visualization not available")

with tab3:
    if 'campaign_performance' in visualizations:
        st.plotly_chart(visualizations['campaign_performance'], use_container_width=True)
    else:
        st.info("Campaign visualization not available")

with tab4:
    if 'region_distribution' in visualizations:
        st.plotly_chart(visualizations['region_distribution'], use_container_width=True)
    else:
        st.info("Region visualization not available")

# ========== DATA EXPLORER ==========
st.markdown('<h2 class="sub-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)

with st.expander("View Raw Data", expanded=False):
    st.dataframe(df, use_container_width=True)

with st.expander("Data Statistics", expanded=False):
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

# ========== AI CHAT ASSISTANT ==========
st.markdown('<h2 class="sub-header">🤖 Marketing AI Assistant</h2>', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your marketing data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = handle_chat_query(prompt, df, analysis)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# ========== REPORT GENERATION ==========
st.markdown('<h2 class="sub-header">📋 Report Generation</h2>', unsafe_allow_html=True)

if st.button("Generate Comprehensive Report", use_container_width=True, type="primary"):
    with st.spinner("Generating report..."):
        # Create report content
        report_content = f"""
        # Marketing Analytics Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Executive Summary
        - Total Revenue: ${analysis.get('total_revenue', 0):,.0f}
        - Total Cost: ${analysis.get('total_cost', 0):,.0f}
        - Overall ROI: {analysis.get('overall_roi', 0):.1f}%
        - Total Profit: ${analysis.get('total_profit', 0):,.0f}
        
        ## Channel Performance
        """
        
        if 'channel_performance' in analysis:
            for _, row in analysis['channel_performance'].iterrows():
                report_content += f"\n- **{row['Channel']}**: {row['ROI']:.1f}% ROI, ${row['CPA']:.2f} CPA, {row['Conversion_Rate']:.1f}% conversion rate"
        
        report_content += "\n\n## Key Insights\n"
        for insight in insights[:7]:
            report_content += f"\n- {insight}"
        
        report_content += f"\n\n## Data Summary\n- Records analyzed: {len(df)}"
        report_content += f"\n- Time period: {df['Date'].min().date()} to {df['Date'].max().date()}"
        report_content += f"\n- Channels: {', '.join(df['Channel'].unique().tolist())}"
        
        # Display report
        st.markdown(report_content)
        
        # Download button
        st.download_button(
            label="📥 Download Report as Text",
            data=report_content,
            file_name=f"marketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ========== FOOTER ==========
st.divider()
st.caption("""
Marketing Analytics Dashboard v1.0 • Powered by Streamlit • Data refreshed on demand
""")

# ========== ERROR HANDLING & DEBUG INFO (Hidden by default) ==========
with st.expander("🛠️ Debug Information", expanded=False):
    st.write("**Session State:**", st.session_state.keys())
    st.write("**Data Shape:**", df.shape if df is not None else "No data")
    st.write("**Columns:**", list(df.columns) if df is not None else [])
    st.write("**Analysis Keys:**", list(analysis.keys()) if analysis else [])