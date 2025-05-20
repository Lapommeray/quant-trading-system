import streamlit as st
import pandas as pd
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

tabs = st.tabs(["Performance Overview", "Gate Analysis", "OverSoul Intelligence", "Raw Data"])

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
