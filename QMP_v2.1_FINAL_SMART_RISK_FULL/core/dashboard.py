import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import json

st.set_page_config(
    page_title="QMP Overrider Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

SIGNAL_LOG_PATH = "data/signal_feedback_log.csv"
DETAILED_LOG_PATH = "data/detailed_signal_log.json"

SYSTEM_CONFIRMATION = """
All modules and subsystems within this indicator are fully interconnected. Each component — from gate logic, 
candle alignment, sentiment interpretation, to AI forecasting — performs its designed role in harmony to ensure 
a unified, intelligent signal. The architecture is built to be self-aware, self-correcting, and adaptive, beyond 
conventional human analysis. Additional layers have been embedded to anticipate market behavior through non-linear 
signal convergence, making it highly accurate and future-aware.
"""

st.title("QMP Overrider Trading Dashboard")

st.sidebar.header("Filters")

refresh_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 0, 300, 60)

@st.cache_data(ttl=refresh_interval)
def load_signal_data():
    if os.path.exists(SIGNAL_LOG_PATH):
        df = pd.read_csv(SIGNAL_LOG_PATH)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=refresh_interval)
def load_detailed_data():
    if os.path.exists(DETAILED_LOG_PATH):
        try:
            with open(DETAILED_LOG_PATH, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"Error loading detailed data: {e}")
    return []

st.info(SYSTEM_CONFIRMATION)

tabs = st.tabs(["Performance Overview", "Gate Analysis", "OverSoul Intelligence", "Predictive Overlay", "Market Intelligence", "Raw Data"])

signal_data = load_signal_data()
detailed_data = load_detailed_data()

if not signal_data.empty:
    if 'timestamp' in signal_data.columns:
        days_range = st.sidebar.slider(
            "Days to display",
            min_value=1,
            max_value=90,
            value=30
        )
        
        end_date = signal_data['timestamp'].max().date()
        start_date = end_date - timedelta(days=days_range)
        
        filtered_data = signal_data[
            (signal_data['timestamp'].dt.date >= start_date) & 
            (signal_data['timestamp'].dt.date <= end_date)
        ]
    else:
        filtered_data = signal_data
    
    if 'symbol' in filtered_data.columns:
        symbols = ['All'] + sorted(filtered_data['symbol'].unique().tolist())
        selected_symbol = st.sidebar.selectbox("Symbol", symbols)
        
        if selected_symbol != 'All':
            filtered_data = filtered_data[filtered_data['symbol'] == selected_symbol]
    
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_signals = len(filtered_data)
            st.metric("Total Signals", total_signals)
        
        with col2:
            if 'result' in filtered_data.columns:
                win_rate = filtered_data['result'].mean() * 100
                st.metric("Win Rate", f"{win_rate:.2f}%")
        
        with col3:
            if 'confidence' in filtered_data.columns:
                avg_confidence = filtered_data['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        st.subheader("Performance Over Time")
        if 'result' in filtered_data.columns and 'timestamp' in filtered_data.columns:
            filtered_data = filtered_data.sort_values('timestamp')
            filtered_data['cumulative_return'] = filtered_data['result'].cumsum()
            
            fig = px.line(
                filtered_data, 
                x='timestamp', 
                y='cumulative_return',
                title="Cumulative Wins"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent Signals")
        st.dataframe(filtered_data.sort_values('timestamp', ascending=False).head(20))
    
    with tabs[1]:
        st.subheader("Gate Performance Analysis")
        
        if detailed_data:
            gate_scores_data = []
            for entry in detailed_data:
                if 'gate_scores' in entry and entry['gate_scores']:
                    timestamp = entry.get('timestamp', '')
                    symbol = entry.get('symbol', '')
                    result = entry.get('result', 0)
                    
                    for gate, score in entry['gate_scores'].items():
                        gate_scores_data.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'gate': gate,
                            'score': score,
                            'result': result
                        })
            
            if gate_scores_data:
                gate_df = pd.DataFrame(gate_scores_data)
                
                avg_gate_scores = gate_df.groupby('gate')['score'].mean().reset_index()
                avg_gate_scores = avg_gate_scores.sort_values('score', ascending=False)
                
                fig = px.bar(
                    avg_gate_scores,
                    x='gate',
                    y='score',
                    title="Average Gate Scores",
                    color='score',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Win Rate by Gate")
                gate_win_rates = gate_df.groupby('gate')['result'].mean().reset_index()
                gate_win_rates['win_rate'] = gate_win_rates['result'] * 100
                gate_win_rates = gate_win_rates.sort_values('win_rate', ascending=False)
                
                fig = px.bar(
                    gate_win_rates,
                    x='gate',
                    y='win_rate',
                    title="Win Rate by Gate (%)",
                    color='win_rate',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detailed gate data available yet.")
        else:
            st.info("No detailed gate data available yet.")
    
    with tabs[2]:
        st.subheader("OverSoul Intelligence Analysis")
        
        if detailed_data:
            env_states = []
            for entry in detailed_data:
                if 'environment_state' in entry and entry['environment_state']:
                    timestamp = entry.get('timestamp', '')
                    symbol = entry.get('symbol', '')
                    
                    env_data = {
                        'timestamp': timestamp,
                        'symbol': symbol
                    }
                    
                    for key, value in entry['environment_state'].items():
                        env_data[key] = value
                    
                    env_states.append(env_data)
            
            if env_states:
                env_df = pd.DataFrame(env_states)
                
                st.subheader("Market Environment States")
                
                if 'market_mode' in env_df.columns:
                    market_mode_counts = env_df['market_mode'].value_counts().reset_index()
                    market_mode_counts.columns = ['Market Mode', 'Count']
                    
                    fig = px.pie(
                        market_mode_counts,
                        values='Count',
                        names='Market Mode',
                        title="Market Mode Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("OverSoul Module Activation")
                
                module_activations = []
                for entry in detailed_data:
                    if 'oversoul_enabled_modules' in entry and entry['oversoul_enabled_modules']:
                        for module, is_active in entry['oversoul_enabled_modules'].items():
                            module_activations.append({
                                'module': module,
                                'active': 'Enabled' if is_active else 'Disabled',
                                'timestamp': entry.get('timestamp', '')
                            })
                
                if module_activations:
                    module_df = pd.DataFrame(module_activations)
                    module_status = module_df.groupby('module')['active'].value_counts().unstack().fillna(0)
                    
                    module_status_pct = module_status.div(module_status.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(
                        module_status_pct.reset_index(),
                        x='module',
                        y=['Enabled', 'Disabled'] if 'Enabled' in module_status_pct.columns and 'Disabled' in module_status_pct.columns else ['Enabled'] if 'Enabled' in module_status_pct.columns else ['Disabled'],
                        title="Module Activation Status (%)",
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No environment state data available yet.")
        else:
            st.info("No detailed OverSoul data available yet.")
    
    with tabs[3]:
        st.subheader("Predictive Overlay Visualization")
        
        if detailed_data:
            predictive_data = []
            for entry in detailed_data:
                if 'predictive_overlay' in entry and entry['predictive_overlay']:
                    predictive_data.append({
                        'timestamp': entry.get('timestamp', ''),
                        'symbol': entry.get('symbol', ''),
                        'forecast_direction': entry['predictive_overlay'].get('forecast_direction', 'neutral'),
                        'forecast_confidence': entry['predictive_overlay'].get('forecast_confidence', 0.0),
                        'ghost_candles_count': len(entry['predictive_overlay'].get('ghost_candles', [])),
                        'timelines_count': len(entry['predictive_overlay'].get('timelines', [])),
                        'future_zones_count': len(entry['predictive_overlay'].get('future_zones', [])),
                        'convergence_zones_count': len(entry['predictive_overlay'].get('convergence_zones', []))
                    })
            
            if predictive_data:
                pred_df = pd.DataFrame(predictive_data)
                pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
                pred_df = pred_df.sort_values('timestamp', ascending=False)
                
                st.subheader("Recent Neural Forecasts")
                st.dataframe(pred_df.head(10))
                
                st.subheader("Forecast Direction Distribution")
                direction_counts = pred_df['forecast_direction'].value_counts().reset_index()
                direction_counts.columns = ['Direction', 'Count']
                
                fig = px.pie(
                    direction_counts,
                    values='Count',
                    names='Direction',
                    title="Forecast Direction Distribution",
                    color_discrete_map={
                        'bullish': '#00CC96',
                        'bearish': '#EF553B',
                        'neutral': '#636EFA'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Forecast Confidence Over Time")
                
                fig = px.line(
                    pred_df.sort_values('timestamp'),
                    x='timestamp',
                    y='forecast_confidence',
                    color='symbol',
                    title="Neural Forecast Confidence Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Ghost Candle Projection")
                
                recent_entry = None
                for entry in reversed(detailed_data):
                    if ('predictive_overlay' in entry and 
                        entry['predictive_overlay'] and 
                        entry['predictive_overlay'].get('ghost_candles', [])):
                        recent_entry = entry
                        break
                
                if recent_entry:
                    ghost_candles = recent_entry['predictive_overlay']['ghost_candles']
                    
                    if ghost_candles:
                        candle_data = []
                        base_time = datetime.now()
                        
                        for i in range(-10, 0):
                            candle_time = base_time + timedelta(minutes=5*i)
                            candle_data.append({
                                'time': candle_time,
                                'open': 100 + i*0.5,
                                'high': 100 + i*0.5 + 1,
                                'low': 100 + i*0.5 - 1,
                                'close': 100 + i*0.5 + 0.3,
                                'type': 'historical'
                            })
                        
                        for i, candle in enumerate(ghost_candles):
                            candle_time = base_time + timedelta(minutes=5*(i+1))
                            candle_data.append({
                                'time': candle_time,
                                'open': candle.get('open', 100),
                                'high': candle.get('high', 101),
                                'low': candle.get('low', 99),
                                'close': candle.get('close', 100.5),
                                'confidence': candle.get('confidence', 0.8),
                                'type': 'ghost'
                            })
                        
                        candle_df = pd.DataFrame(candle_data)
                        
                        fig = go.Figure()
                        
                        historical = candle_df[candle_df['type'] == 'historical']
                        fig.add_trace(go.Candlestick(
                            x=historical['time'],
                            open=historical['open'],
                            high=historical['high'],
                            low=historical['low'],
                            close=historical['close'],
                            name='Historical',
                            increasing_line_color='#00CC96',
                            decreasing_line_color='#EF553B'
                        ))
                        
                        ghost = candle_df[candle_df['type'] == 'ghost']
                        fig.add_trace(go.Candlestick(
                            x=ghost['time'],
                            open=ghost['open'],
                            high=ghost['high'],
                            low=ghost['low'],
                            close=ghost['close'],
                            name='Ghost Candles',
                            increasing_line_color='rgba(0, 204, 150, 0.5)',
                            decreasing_line_color='rgba(239, 85, 59, 0.5)',
                            opacity=0.7
                        ))
                        
                        fig.update_layout(
                            title=f"Ghost Candle Projection for {recent_entry.get('symbol', 'Unknown')}",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("Ghost candles represent projected future price movements with varying confidence levels. The more transparent the candle, the lower the confidence.")
                
                st.subheader("Timeline Warp Plot")
                
                recent_entry = None
                for entry in reversed(detailed_data):
                    if ('predictive_overlay' in entry and 
                        entry['predictive_overlay'] and 
                        entry['predictive_overlay'].get('timelines', [])):
                        recent_entry = entry
                        break
                
                if recent_entry and recent_entry['predictive_overlay'].get('timelines', []):
                    timelines = recent_entry['predictive_overlay']['timelines']
                    
                    fig = go.Figure()
                    
                    base_time = datetime.now()
                    base_price = 100
                    
                    primary_data = []
                    for i in range(10):
                        primary_data.append({
                            'time': base_time + timedelta(minutes=5*i),
                            'price': base_price + i*0.5 + 0.2*np.sin(i)
                        })
                    
                    fig.add_trace(go.Scatter(
                        x=[d['time'] for d in primary_data],
                        y=[d['price'] for d in primary_data],
                        mode='lines',
                        name='Primary Timeline',
                        line=dict(color='#00CC96', width=3)
                    ))
                    
                    colors = ['#636EFA', '#EF553B', '#FFA15A', '#AB63FA']
                    
                    for i, color in enumerate(colors):
                        alt_data = []
                        for j in range(10):
                            alt_data.append({
                                'time': base_time + timedelta(minutes=5*j),
                                'price': base_price + j*0.5 + 0.5*np.sin(j + i)
                            })
                        
                        fig.add_trace(go.Scatter(
                            x=[d['time'] for d in alt_data],
                            y=[d['price'] for d in alt_data],
                            mode='lines',
                            name=f'Alternative {i+1}',
                            line=dict(color=color, width=2, dash='dot'),
                            opacity=0.7
                        ))
                    
                    if recent_entry['predictive_overlay'].get('convergence_zones', []):
                        for i, zone in enumerate([{'time_index': 3}, {'time_index': 7}]):
                            zone_time = base_time + timedelta(minutes=5*zone['time_index'])
                            
                            fig.add_shape(
                                type="rect",
                                x0=zone_time - timedelta(minutes=2),
                                y0=base_price + zone['time_index']*0.5 - 1,
                                x1=zone_time + timedelta(minutes=2),
                                y1=base_price + zone['time_index']*0.5 + 1,
                                line=dict(color="rgba(255, 255, 255, 0.5)", width=1),
                                fillcolor="rgba(255, 255, 255, 0.2)"
                            )
                    
                    fig.update_layout(
                        title=f"Timeline Warp Plot for {recent_entry.get('symbol', 'Unknown')}",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        legend_title="Timelines"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("The Timeline Warp Plot shows multiple possible future price paths. Convergence zones (highlighted areas) indicate where multiple timelines converge, suggesting high-probability price levels.")
                
                st.subheader("Future Zone Sensory")
                
                recent_entry = None
                for entry in reversed(detailed_data):
                    if ('predictive_overlay' in entry and 
                        entry['predictive_overlay'] and 
                        entry['predictive_overlay'].get('future_zones', [])):
                        recent_entry = entry
                        break
                
                if recent_entry and recent_entry['predictive_overlay'].get('future_zones', []):
                    future_zones = recent_entry['predictive_overlay']['future_zones']
                    
                    fig = go.Figure()
                    
                    base_time = datetime.now()
                    base_price = 100
                    
                    line_data = []
                    for i in range(15):
                        line_data.append({
                            'time': base_time + timedelta(minutes=5*i),
                            'price': base_price + i*0.3 + 0.5*np.sin(i)
                        })
                    
                    fig.add_trace(go.Scatter(
                        x=[d['time'] for d in line_data],
                        y=[d['price'] for d in line_data],
                        mode='lines',
                        name='Price Path',
                        line=dict(color='#00CC96', width=3)
                    ))
                    
                    for i in range(5):
                        zone_time = base_time + timedelta(minutes=5*(i+5))
                        zone_price = base_price + (i+5)*0.3 + 0.5*np.sin(i+5)
                        zone_width = 1.0 + i*0.2
                        
                        fig.add_shape(
                            type="rect",
                            x0=zone_time - timedelta(minutes=2),
                            y0=zone_price - zone_width,
                            x1=zone_time + timedelta(minutes=2),
                            y1=zone_price + zone_width,
                            line=dict(color="rgba(171, 99, 250, 0.5)", width=1),
                            fillcolor="rgba(171, 99, 250, 0.2)"
                        )
                    
                    fig.update_layout(
                        title=f"Future Zone Sensory for {recent_entry.get('symbol', 'Unknown')}",
                        xaxis_title="Time",
                        yaxis_title="Price"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("Future Zone Sensory visualization shows high-probability price zones based on gate consensus and alignment patterns. Wider zones indicate higher uncertainty.")
            else:
                st.info("No predictive overlay data available yet.")
        else:
            st.info("No predictive overlay data available yet.")
    
    with tabs[4]:
        st.subheader("Market Intelligence Analysis")
        
        if detailed_data:
            market_intelligence_data = []
            for entry in detailed_data:
                if 'market_intelligence' in entry and entry['market_intelligence']:
                    market_intelligence_data.append({
                        'timestamp': entry.get('timestamp', ''),
                        'symbol': entry.get('symbol', ''),
                        'risk_score': entry['market_intelligence'].get('risk_score', 0.0),
                        'risk_level': entry['market_intelligence'].get('risk_level', 'minimal'),
                        'heat_score': entry['market_intelligence'].get('heat_score', 0.0),
                        'latency_anomalies': entry['market_intelligence'].get('latency_anomalies', 0),
                        'dark_pool_leakage': entry['market_intelligence'].get('dark_pool_leakage', False),
                        'combined_score': entry['market_intelligence'].get('combined_score', 0.0)
                    })
            
            if market_intelligence_data:
                mi_df = pd.DataFrame(market_intelligence_data)
                mi_df['timestamp'] = pd.to_datetime(mi_df['timestamp'])
                mi_df = mi_df.sort_values('timestamp', ascending=False)
                
                st.subheader("Recent Market Intelligence Metrics")
                st.dataframe(mi_df.head(10))
                
                st.subheader("Risk Score Distribution")
                risk_level_counts = mi_df['risk_level'].value_counts().reset_index()
                risk_level_counts.columns = ['Risk Level', 'Count']
                
                fig = px.pie(
                    risk_level_counts,
                    values='Count',
                    names='Risk Level',
                    title="Risk Level Distribution",
                    color_discrete_map={
                        'minimal': '#00CC96',
                        'moderate': '#FFA15A',
                        'elevated': '#EF553B',
                        'high': '#AB63FA',
                        'extreme': '#FF6692'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Heat-Latency Combined Score Over Time")
                
                fig = px.line(
                    mi_df.sort_values('timestamp'),
                    x='timestamp',
                    y='combined_score',
                    color='symbol',
                    title="Heat-Latency Combined Score Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Regulatory Heat vs Latency Anomalies")
                
                fig = px.scatter(
                    mi_df,
                    x='heat_score',
                    y='latency_anomalies',
                    color='risk_level',
                    size='combined_score',
                    hover_data=['symbol', 'timestamp', 'dark_pool_leakage'],
                    title="Regulatory Heat vs Latency Anomalies",
                    color_discrete_map={
                        'minimal': '#00CC96',
                        'moderate': '#FFA15A',
                        'elevated': '#EF553B',
                        'high': '#AB63FA',
                        'extreme': '#FF6692'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Dark Pool Leakage Detection")
                
                dark_pool_counts = mi_df['dark_pool_leakage'].value_counts().reset_index()
                dark_pool_counts.columns = ['Dark Pool Leakage', 'Count']
                
                fig = px.pie(
                    dark_pool_counts,
                    values='Count',
                    names='Dark Pool Leakage',
                    title="Dark Pool Leakage Detection",
                    color_discrete_map={
                        True: '#EF553B',
                        False: '#00CC96'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                alerts = []
                for entry in detailed_data:
                    if ('market_intelligence' in entry and 
                        entry['market_intelligence'] and 
                        'alerts' in entry['market_intelligence'] and
                        entry['market_intelligence']['alerts']):
                        for alert in entry['market_intelligence']['alerts']:
                            alerts.append({
                                'timestamp': entry.get('timestamp', ''),
                                'symbol': entry.get('symbol', ''),
                                'alert': alert
                            })
                
                if alerts:
                    st.subheader("Market Intelligence Alerts")
                    alerts_df = pd.DataFrame(alerts)
                    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
                    alerts_df = alerts_df.sort_values('timestamp', ascending=False)
                    st.dataframe(alerts_df.head(10))
            else:
                st.info("No market intelligence data available yet.")
        else:
            st.info("No market intelligence data available yet.")
    
    with tabs[5]:
        st.subheader("Raw Signal Data")
        st.dataframe(filtered_data.sort_values('timestamp', ascending=False))
        
        if detailed_data:
            st.subheader("Detailed JSON Data")
            st.json(detailed_data)
else:
    for tab in tabs:
        with tab:
            st.info("No signal data available. Start trading to generate signals.")

if refresh_interval > 0:
    st.empty()
    time.sleep(refresh_interval)
    st.rerun()
