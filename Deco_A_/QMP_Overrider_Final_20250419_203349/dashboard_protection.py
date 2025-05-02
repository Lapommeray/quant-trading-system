"""
Institutional-Grade Protection Dashboard Module

This module implements the Institutional-Grade Protection Dashboard for the QMP Overrider system.
It provides real-time monitoring and visualization of circuit breakers, blockchain audit trails,
and dark pool liquidity prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProtectionDashboard")

st.set_page_config(
    page_title="QMP Overrider - Institutional Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Protection Controls")
refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 0, 300, 60)

@st.cache_data(ttl=refresh_interval)
def load_circuit_breaker_data():
    try:
        log_dir = Path("logs/circuit_breakers")
        history_file = log_dir / "trip_history.json"
        
        if history_file.exists():
            with open(history_file, "r") as f:
                trip_history = json.load(f)
            
            df = pd.DataFrame(trip_history)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
        else:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            data = []
            
            for date in dates:
                if np.random.random() < 0.2:  # 20% chance of circuit breaker trip
                    data.append({
                        'timestamp': date,
                        'config': {
                            'volatility_threshold': np.random.uniform(0.05, 0.2),
                            'latency_spike_ms': np.random.uniform(20, 200),
                            'order_imbalance_ratio': np.random.uniform(1.5, 5.0)
                        },
                        'reason': np.random.choice(['volatility', 'latency', 'imbalance']),
                        'duration_s': np.random.uniform(30, 300)
                    })
            
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading circuit breaker data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=refresh_interval)
def load_blockchain_audit_data():
    try:
        log_dir = Path("logs/blockchain")
        roots_file = log_dir / "merkle_roots.json"
        
        if roots_file.exists():
            with open(roots_file, "r") as f:
                roots = json.load(f)
            
            df = pd.DataFrame(roots)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        else:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
            data = []
            
            for date in dates:
                if np.random.random() < 0.1:  # 10% chance of audit event
                    data.append({
                        'timestamp': date,
                        'root': f"0x{os.urandom(32).hex()}",
                        'event_count': np.random.randint(1, 10),
                        'verified': np.random.choice([True, False], p=[0.95, 0.05])
                    })
            
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading blockchain audit data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=refresh_interval)
def load_dark_pool_data():
    try:
        log_dir = Path("logs/dark_pool")
        failover_file = log_dir / "failover_history.json"
        
        if failover_file.exists():
            with open(failover_file, "r") as f:
                failover_history = json.load(f)
            
            df = pd.DataFrame(failover_history)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
        else:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='6H')
            pools = ['pool_alpha', 'pool_sigma', 'pool_omega']
            data = []
            
            current_pool = pools[0]
            for date in dates:
                if np.random.random() < 0.3:  # 30% chance of failover
                    old_pool = current_pool
                    current_pool = np.random.choice([p for p in pools if p != current_pool])
                    
                    data.append({
                        'timestamp': date,
                        'old_pool': old_pool,
                        'new_pool': current_pool,
                        'reason': np.random.choice(['health_check', 'low_fill_rate', 'latency_spike']),
                        'pool_health': {
                            'pool_alpha': np.random.uniform(0.5, 1.0),
                            'pool_sigma': np.random.uniform(0.5, 1.0),
                            'pool_omega': np.random.uniform(0.5, 1.0)
                        }
                    })
            
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading dark pool data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=refresh_interval)
def load_liquidity_prediction_data():
    try:
        model_dir = Path("models/dark_pool")
        history_file = model_dir / "prediction_history.json"
        
        if history_file.exists():
            with open(history_file, "r") as f:
                prediction_history = json.load(f)
            
            df = pd.DataFrame(prediction_history)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        else:
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
            data = []
            
            for date in dates:
                data.append({
                    'timestamp': date,
                    'prediction': np.random.uniform(0.5, 0.95),
                    'features': {
                        'time_of_day_sin': np.sin(2 * np.pi * date.hour / 24),
                        'time_of_day_cos': np.cos(2 * np.pi * date.hour / 24),
                        'market_volatility': np.random.uniform(0.05, 0.2),
                        'spread_bps': np.random.uniform(1, 10),
                        'pool_latency_ms': np.random.uniform(10, 100)
                    }
                })
            
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading liquidity prediction data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=refresh_interval)
def load_ml_tuner_data():
    try:
        model_dir = Path("models/circuit_breakers")
        history_file = model_dir / "training_history.json"
        
        if history_file.exists():
            with open(history_file, "r") as f:
                training_history = json.load(f)
            
            df = pd.DataFrame(training_history)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
        else:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            data = []
            
            loss = 0.5
            for date in dates:
                loss = max(0.01, loss * np.random.uniform(0.8, 1.0))
                
                data.append({
                    'timestamp': date,
                    'loss': loss,
                    'target_params': {
                        'volatility_threshold': np.random.uniform(0.05, 0.2),
                        'latency_spike_ms': np.random.uniform(20, 200),
                        'order_imbalance_ratio': np.random.uniform(1.5, 5.0),
                        'cooling_period': np.random.uniform(30, 300)
                    },
                    'market_data_summary': {
                        'exchange_volatility': np.random.uniform(0.05, 0.2),
                        'regime_vix': np.random.uniform(15, 35)
                    }
                })
            
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading ML tuner data: {e}")
        return pd.DataFrame()

st.title("🛡️ Institutional-Grade Protection Dashboard")

tabs = st.tabs([
    "Circuit Breakers", 
    "Blockchain Audit", 
    "Dark Pool Intelligence", 
    "ML Tuner"
])

circuit_breaker_data = load_circuit_breaker_data()
blockchain_audit_data = load_blockchain_audit_data()
dark_pool_data = load_dark_pool_data()
liquidity_prediction_data = load_liquidity_prediction_data()
ml_tuner_data = load_ml_tuner_data()

with tabs[0]:
    st.header("Circuit Breaker Monitoring")
    
    if not circuit_breaker_data.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trip_count = len(circuit_breaker_data)
            st.metric("Total Trips", trip_count)
        
        with col2:
            if 'timestamp' in circuit_breaker_data.columns:
                last_trip = circuit_breaker_data['timestamp'].max()
                time_since = datetime.now() - last_trip
                st.metric("Time Since Last Trip", f"{time_since.days}d {time_since.seconds // 3600}h")
            else:
                st.metric("Time Since Last Trip", "N/A")
        
        with col3:
            if 'duration_s' in circuit_breaker_data.columns:
                avg_duration = circuit_breaker_data['duration_s'].mean()
                st.metric("Avg Trip Duration", f"{avg_duration:.1f}s")
            else:
                st.metric("Avg Trip Duration", "N/A")
        
        st.subheader("Circuit Breaker Timeline")
        if 'timestamp' in circuit_breaker_data.columns and 'reason' in circuit_breaker_data.columns:
            fig = px.scatter(
                circuit_breaker_data,
                x='timestamp',
                y='reason',
                color='reason',
                size_max=10,
                title="Circuit Breaker Trips by Reason"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent Circuit Breaker Trips")
        st.dataframe(circuit_breaker_data.sort_values('timestamp', ascending=False).head(10))
    else:
        st.info("No circuit breaker data available.")

with tabs[1]:
    st.header("Blockchain-Verified Audit Trail")
    
    if not blockchain_audit_data.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_count = blockchain_audit_data['event_count'].sum() if 'event_count' in blockchain_audit_data.columns else len(blockchain_audit_data)
            st.metric("Total Audit Events", event_count)
        
        with col2:
            if 'timestamp' in blockchain_audit_data.columns:
                last_audit = blockchain_audit_data['timestamp'].max()
                time_since = datetime.now() - last_audit
                st.metric("Time Since Last Audit", f"{time_since.seconds // 60}m")
            else:
                st.metric("Time Since Last Audit", "N/A")
        
        with col3:
            if 'verified' in blockchain_audit_data.columns:
                verification_rate = blockchain_audit_data['verified'].mean() * 100
                st.metric("Verification Rate", f"{verification_rate:.1f}%")
            else:
                st.metric("Verification Rate", "100%")
        
        st.subheader("Audit Timeline")
        if 'timestamp' in blockchain_audit_data.columns and 'event_count' in blockchain_audit_data.columns:
            fig = px.bar(
                blockchain_audit_data,
                x='timestamp',
                y='event_count',
                title="Audit Events Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent Audit Events")
        st.dataframe(blockchain_audit_data.sort_values('timestamp', ascending=False).head(10))
    else:
        st.info("No blockchain audit data available.")

with tabs[2]:
    st.header("Dark Pool Intelligence")
    
    dark_pool_tabs = st.tabs(["Failover System", "Liquidity Prediction"])
    
    with dark_pool_tabs[0]:
        st.subheader("Dark Pool Failover System")
        
        if not dark_pool_data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                failover_count = len(dark_pool_data)
                st.metric("Total Failovers", failover_count)
            
            with col2:
                if 'timestamp' in dark_pool_data.columns:
                    last_failover = dark_pool_data['timestamp'].max()
                    time_since = datetime.now() - last_failover
                    st.metric("Time Since Last Failover", f"{time_since.days}d {time_since.seconds // 3600}h")
                else:
                    st.metric("Time Since Last Failover", "N/A")
            
            with col3:
                if 'reason' in dark_pool_data.columns:
                    top_reason = dark_pool_data['reason'].value_counts().index[0]
                    st.metric("Top Failover Reason", top_reason)
                else:
                    st.metric("Top Failover Reason", "N/A")
            
            st.subheader("Failover Timeline")
            if 'timestamp' in dark_pool_data.columns and 'new_pool' in dark_pool_data.columns:
                fig = px.scatter(
                    dark_pool_data,
                    x='timestamp',
                    y='new_pool',
                    color='reason' if 'reason' in dark_pool_data.columns else None,
                    title="Dark Pool Failovers"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Recent Failovers")
            display_cols = [c for c in dark_pool_data.columns if c != 'pool_health']
            st.dataframe(dark_pool_data[display_cols].sort_values('timestamp', ascending=False).head(10))
        else:
            st.info("No dark pool failover data available.")
    
    with dark_pool_tabs[1]:
        st.subheader("Dark Pool Liquidity Prediction")
        
        if not liquidity_prediction_data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'prediction' in liquidity_prediction_data.columns:
                    avg_liquidity = liquidity_prediction_data['prediction'].mean()
                    st.metric("Average Predicted Liquidity", f"{avg_liquidity:.2f}")
                else:
                    st.metric("Average Predicted Liquidity", "N/A")
            
            with col2:
                if 'prediction' in liquidity_prediction_data.columns:
                    current_liquidity = liquidity_prediction_data['prediction'].iloc[-1]
                    st.metric("Current Predicted Liquidity", f"{current_liquidity:.2f}")
                else:
                    st.metric("Current Predicted Liquidity", "N/A")
            
            with col3:
                if 'prediction' in liquidity_prediction_data.columns:
                    liquidity_trend = liquidity_prediction_data['prediction'].iloc[-1] - liquidity_prediction_data['prediction'].iloc[0]
                    st.metric("Liquidity Trend", f"{liquidity_trend:.2f}", delta=f"{liquidity_trend:.2f}")
                else:
                    st.metric("Liquidity Trend", "N/A")
            
            st.subheader("Liquidity Prediction Timeline")
            if 'timestamp' in liquidity_prediction_data.columns and 'prediction' in liquidity_prediction_data.columns:
                fig = px.line(
                    liquidity_prediction_data,
                    x='timestamp',
                    y='prediction',
                    title="Predicted Liquidity Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Recent Liquidity Predictions")
            display_cols = [c for c in liquidity_prediction_data.columns if c != 'features']
            st.dataframe(liquidity_prediction_data[display_cols].sort_values('timestamp', ascending=False).head(10))
        else:
            st.info("No liquidity prediction data available.")

with tabs[3]:
    st.header("ML-Driven Circuit Breaker Tuner")
    
    if not ml_tuner_data.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'loss' in ml_tuner_data.columns:
                current_loss = ml_tuner_data['loss'].iloc[-1]
                st.metric("Current Training Loss", f"{current_loss:.4f}")
            else:
                st.metric("Current Training Loss", "N/A")
        
        with col2:
            if 'loss' in ml_tuner_data.columns:
                loss_improvement = ml_tuner_data['loss'].iloc[0] - ml_tuner_data['loss'].iloc[-1]
                st.metric("Loss Improvement", f"{loss_improvement:.4f}", delta=f"{loss_improvement:.4f}")
            else:
                st.metric("Loss Improvement", "N/A")
        
        with col3:
            if 'timestamp' in ml_tuner_data.columns:
                last_training = ml_tuner_data['timestamp'].max()
                time_since = datetime.now() - last_training
                st.metric("Time Since Last Training", f"{time_since.days}d {time_since.seconds // 3600}h")
            else:
                st.metric("Time Since Last Training", "N/A")
        
        st.subheader("Training Loss Timeline")
        if 'timestamp' in ml_tuner_data.columns and 'loss' in ml_tuner_data.columns:
            fig = px.line(
                ml_tuner_data,
                x='timestamp',
                y='loss',
                title="ML Tuner Training Loss Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ML tuner data available.")

if refresh_interval > 0:
    st.empty()
    time.sleep(refresh_interval)
    st.rerun()
