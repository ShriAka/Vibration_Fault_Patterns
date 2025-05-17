import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.stats import describe

# Fault frequency multipliers
bpfi, bpfo, ftf, bsf = 4.93, 3.07, 0.38, 2.04

def generate_fault_signal(fault_type, rot_freq, fs=10000, duration=1.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)

    base_amp = 1

    if fault_type == "Normal":
        # 1X + minor harmonics + noise
        signal += 1.0 * np.sin(2 * np.pi * rot_freq * t)
        signal += 0.2 * np.sin(2 * np.pi * 2 * rot_freq * t)
        signal += 0.1 * np.sin(2 * np.pi * 3 * rot_freq * t)
        signal += 0.05 * np.random.randn(len(t))
    elif fault_type == "Unbalance":
        signal += 0.8 * base_amp * np.sin(2 * np.pi * rot_freq * t)
    elif fault_type == "Misalignment":
        signal += 0.4 * base_amp * np.sin(2 * np.pi * rot_freq * t)
        signal += 0.6 * base_amp * np.sin(2 * np.pi * 2 * rot_freq * t)
    elif fault_type == "Looseness":
        for i in range(1, 11):
            signal += (0.2 - 0.02 * i) * base_amp * np.sin(2 * np.pi * i * rot_freq * t)
        signal += 0.2 * np.random.randn(len(t))  # raised noise floor
    elif fault_type == "Bearing Fault":
        # Start with a 'normal' baseline signal
        signal += 1.0 * np.sin(2 * np.pi * rot_freq * t)
        signal += 0.2 * np.sin(2 * np.pi * 2 * rot_freq * t)
        signal += 0.1 * np.sin(2 * np.pi * 3 * rot_freq * t)
        signal += 0.05 * np.random.randn(len(t))
    
        # Add bearing fault frequencies (BPFI)
        signal += 0.2 * np.sin(2 * np.pi * bpfi * rot_freq * t)
        signal += 0.18 * np.sin(2 * np.pi * 2 * bpfi * rot_freq * t)
        signal += 0.18 * np.sin(2 * np.pi * 3 * bpfi * rot_freq * t)
    
        # Add regular impacts spaced by 1 / (rot_freq * bpfi)
        impact_interval = 1 / (rot_freq * bpfi)
        impact_times = np.arange(0, duration, impact_interval)
        impact_width = int(0.001 * fs)  # small pulse width (1 ms)
    
        for it in impact_times:
            idx = int(it * fs)
            if 0 <= idx < len(t) - impact_width:
                impact = np.hanning(impact_width) * 0.3  # soft impulse shape
                signal[idx:idx + impact_width] += impact
    return t, signal

def compute_fft(y, fs):
    n = len(y)
    yf = fft(y)
    xf = fftfreq(n, 1/fs)[:n//2]
    amp = 2.0 / n * np.abs(yf[:n//2])
    return xf, amp

def compute_stats(y):
    desc = describe(y)
    rms = np.sqrt(np.mean(y**2))
    crest = np.max(np.abs(y)) / rms if rms else 0
    return {
        "Mean": desc.mean,
        "RMS": rms,
        "Peak": np.max(np.abs(y)),
        "Crest Factor": crest,
        "Variance": desc.variance,
        "Skewness": desc.skewness,
        "Kurtosis": desc.kurtosis
    }
from scipy.signal import hilbert

def calculate_envelope_spectrum(y, fs):
    analytic = hilbert(y - np.mean(y))
    envelope = np.abs(analytic)
    return compute_fft(envelope - np.mean(envelope), fs)
    
def plot_signal(t, y, xf, yf):
    env_xf, env_yf = calculate_envelope_spectrum(y, fs)

    fig_time = go.Figure().add_trace(go.Scatter(x=t, y=y, name='Time')).update_layout(title="Time Domain")
    fig_fft = go.Figure().add_trace(go.Scatter(x=xf, y=yf, name='FFT')).update_layout(title="Frequency Spectrum")
    fig_env = go.Figure().add_trace(go.Scatter(x=env_xf, y=env_yf, name='Envelope')).update_layout(title="Envelope Spectrum")

    return fig_time, fig_fft, fig_env


# --- Streamlit UI ---
st.set_page_config("Vibration Fault Pattern Simulator", layout="wide")
st.title("Vibration Fault Pattern Simulator")

col1, col2 = st.columns(2)
with col1:
    fault_type = st.selectbox("Select Fault Condition", ["Normal", "Unbalance", "Misalignment", "Looseness", "Bearing Fault"])
with col2:
    rpm = st.number_input("Enter Machine Speed (RPM)", min_value=60, max_value=60000, value=1800)

fs = 10000
duration = 1.0
rot_freq = rpm / 60.0

if st.button("Generate & Analyze"):
    t, signal = generate_fault_signal(fault_type, rot_freq, fs, duration)
    xf, yf = compute_fft(signal, fs)

    fig_time, fig_fft, fig_env = plot_signal(t, signal, xf, yf)
    st.plotly_chart(fig_time)
    st.plotly_chart(fig_fft)
    st.plotly_chart(fig_env)

    
    st.subheader("Signal Statistics")
    stats = compute_stats(signal)
    cols = st.columns(4)
    for i, (k, v) in enumerate(stats.items()):
        cols[i % 4].metric(k, f"{v:.4g}")
