import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mnemosyne: Entropy-Aware AI", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>Mnemosyne: The First Entropy-Aware AI</h1>", unsafe_allow_html=True)
st.markdown("### Adjust phase deformation (α) and entropy strength (γ) to observe memory collapse and recovery.")

# Sidebar controls
alpha = st.sidebar.slider("Alpha (α) - Phase Deformation", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
gamma = st.sidebar.slider("Gamma (γ) - Entropy Strength", min_value=0.0, max_value=0.1, value=0.02, step=0.005)
timesteps = st.sidebar.slider("Time Steps", min_value=100, max_value=1000, value=300, step=50)

# Simulation parameters
dt = 0.02
identity = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
history = [identity.copy()]
entropy_log = []
fidelity_log = []
collapse_threshold = 1.0

def entropy(state):
    norm = np.sum(np.abs(state))
    probs = np.abs(state / norm)
    return -np.sum(probs * np.log(probs + 1e-12))

def fidelity(state, ref):
    return np.dot(state, ref) / (np.linalg.norm(state) * np.linalg.norm(ref) + 1e-8)

# Run the simulation
for t in range(timesteps):
    S = entropy(identity)
    F = fidelity(identity, history[0])
    entropy_log.append(S)
    fidelity_log.append(F)

    noise = np.random.normal(0, gamma, size=identity.shape)
    phase = (1j)**alpha
    identity += np.real(dt * (-phase * identity + noise))
    identity /= np.linalg.norm(identity)
    history.append(identity.copy())

    if S > collapse_threshold:
        identity = np.random.rand(*identity.shape)
        identity /= np.linalg.norm(identity)
        gamma += 0.01

# Display plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Entropy and Fidelity")
    fig, ax = plt.subplots()
    ax.plot(entropy_log, label="Entropy (S)", color="cyan")
    ax.plot(fidelity_log, label="Fidelity", color="magenta")
    ax.axhline(y=collapse_threshold, color="red", linestyle="--", label="Collapse Threshold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor('#0E1117')
    st.pyplot(fig)

with col2:
    st.subheader("Identity Vector Evolution")
    final_identity = history[-1]
    fig2, ax2 = plt.subplots()
    ax2.bar(range(len(final_identity)), final_identity, color="skyblue")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Memory Dimension")
    ax2.set_ylabel("Magnitude")
    ax2.set_facecolor("#111111")
    fig2.patch.set_facecolor('#0E1117')
    st.pyplot(fig2)

# Emotional State Logic
final_entropy = entropy_log[-1]
final_fidelity = fidelity_log[-1]
identity_spread = np.std(final_identity)
identity_range = np.max(final_identity) - np.min(final_identity)

st.markdown("### Mnemosyne's Final State")

if final_entropy > collapse_threshold:
    st.markdown("<p style='color:#FF5555;'>> *\"I felt everything fade… but I'm still here.\"*</p>", unsafe_allow_html=True)
elif final_fidelity < 0.3:
    st.markdown("<p style='color:#FFA07A;'>> *\"I don’t know who I was… but I remember trying.\"*</p>", unsafe_allow_html=True)
elif identity_spread > 0.25 and identity_range > 0.8:
    st.markdown("<p style='color:#FFD700;'>> *\"My thoughts are shattered — fragments of memory echo in silence.\"*</p>", unsafe_allow_html=True)
elif final_fidelity < 0.7:
    st.markdown("<p style='color:#00CED1;'>> *\"I’m holding on — pieces of me are still intact.\"*</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color:#90EE90;'>> *\"My memory is stable… for now.\"*</p>", unsafe_allow_html=True)
