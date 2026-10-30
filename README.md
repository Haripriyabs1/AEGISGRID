# ⚡ AegisGrid AI: Autonomous Self-Healing Electrical Distribution Network (Simulation Platform)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandapower](https://img.shields.io/badge/Power_Flow-pandapower-FF6F00)](https://www.pandapower.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Interactive_Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Data_Viz-Plotly_v5-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![NetworkX](https://img.shields.io/badge/Graph_Topology-NetworkX-008080)](https://networkx.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**AegisGrid AI** is an advanced, fully software-based simulation platform for **Fault Location, Isolation, and Service Restoration (FLISR)** in smart electrical distribution networks. It couples physical AC Newton-Raphson power flow modeling, machine learning ensemble diagnostics, graph-theoretic radiality verification, and multi-objective optimization to autonomously restore power to healthy de-energized grid sections without physical hardware requirements.

---

## 📑 Table of Contents

1. [Executive Overview & Problem Statement](#-executive-overview--problem-statement)
2. [Key Capabilities & Innovations](#-key-capabilities--innovations)
3. [System Architecture & Dataflow](#-system-architecture--dataflow)
4. [Theoretical Framework & Mathematical Formulations](#-theoretical-framework--mathematical-formulations)
   - [AC Newton-Raphson Power Flow](#1-ac-newton-raphson-power-flow)
   - [Physical Fault Modeling & Telemetry Simulation](#2-physical-fault-modeling--telemetry-simulation)
   - [AI Diagnostic Ensemble (Detection, Localization, Distance)](#3-ai-diagnostic-ensemble)
   - [Graph Topology & Radiality Constraints](#4-graph-topology--radiality-constraints)
   - [Multi-Objective Service Restoration Optimization](#5-multi-objective-service-restoration-optimization)
   - [IEEE 1366 Reliability Standards & VoLL Economics](#6-ieee-1366-reliability-standards--voll-economics)
5. [Benchmark Distribution Network Topologies](#-benchmark-distribution-network-topologies)
6. [Interactive Web Dashboard Features](#-interactive-web-dashboard-features)
7. [Repository File Structure](#-repository-file-structure)
8. [Installation & Execution Guide](#-installation--execution-guide)
9. [Automated Verification & Test Suite](#-automated-verification--test-suite)
10. [Production Deployment Guide](#-production-deployment-guide)

---

## 📌 Executive Overview & Problem Statement

Modern power distribution networks are predominantly operated in a **radial configuration** for simplicity of protection coordination and low capital investment. However, when an insulation flashover, conductor break, or environmental impact causes a fault on a primary branch:
- **Traditional Grid Operations**: A primary breaker trips, plunging the entire downstream feeder section into blackout. Locating the faulted line requires dispatching field patrol crews, manually testing sectionalizers, and performing manual switching. This process typically takes **90 to 180 minutes of sustained customer blackout (SAIDI)**.
- **The AegisGrid AI Solution**: A closed-loop, autonomous software simulation that:
  1. Continuously monitors smart meter / PMU telemetry (voltages, current surges, voltage sags).
  2. Uses trained **Scikit-Learn ensemble models** to detect the fault, classify its physical nature, and pinpoint the exact faulted line in **under 5 milliseconds**.
  3. Commands flanking sectionalizers to open, isolating the damaged section.
  4. Runs graph-theoretic spanning tree search and multi-objective optimization across candidate normally-open (NO) tie-switches to find the optimal backfeed path.
  5. Validates thermal ampacity and voltage limits, applying priority load preservation (protecting critical loads like hospitals) if capacity is constrained.
  6. Restores service to unaffected customers in **under 2.5 seconds**, achieving up to a **99% reduction in SAIDI**.

---

## 🌟 Key Capabilities & Innovations

- **100% Hardware-Free Physics Simulation**: Full AC Newton-Raphson non-linear power flow implemented with [pandapower](https://www.pandapower.org/).
- **4 Realistic Fault Signatures**: Simulates Three-Phase Bolted Short-Circuits, Single Line-to-Ground (SLG) Faults, Line-to-Line (L-L) Faults, and High Impedance Faults (HIF) with variable fault resistance $R_f$ and Gaussian PMU telemetry noise.
- **Machine Learning Diagnostics**:
  - **Fault Classifier**: Random Forest (100 estimators, max depth 16) with weighted F1-score evaluation.
  - **Section Locator**: ExtraTrees Classifier (100 estimators, max depth 18) for high-accuracy line pinpointing.
  - **Distance Regressor**: Gradient Boosting Regressor (80 estimators) estimating the exact fault location percentage along the feeder branch.
- **Strict Radiality & Electrical Constraint Engine**: Enforces zero loops (spanning tree verification via NetworkX: $|E| = |V| - 1$), thermal cable ampacity ($Loading \le 100\%$), and voltage security limits ($0.90 \le V \le 1.05$ p.u.).
- **Executive Streamlit Console**: Glassmorphic dark UI, real-time 2D Plotly distribution topology diagram with glowing fault markers and restored tie paths, bus-by-bus voltage curves, loading bars, and CSV audit log exporter.

---

## 🏗️ System Architecture & Dataflow

```
                                  +---------------------------------------+
                                  |      Active Distribution Grid        |
                                  | (IEEE 33-Bus / 16-Bus Smart Microgrid)|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       pandapower AC Power Flow        |
                                  |   (Voltages V_m, Currents I_ka, Loss) |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       Physical Fault Simulator        |
                                  |   (3-Phase, SLG, L-L, HIF, PMU Noise) |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                      +---------------------------------------------------------------+
                      |              AI Ensemble Diagnostic Subsystem                 |
                      |  - Feature Vector: [V_m(1..N), ΔV(1..N), I(1..M), ΔI(1..M)]  |
                      |  - Random Forest Classifier (Fault Detection & Type)          |
                      |  - ExtraTrees Classifier (Branch Section Pinpointer)         |
                      |  - Gradient Boosting Regressor (Fault Distance %)             |
                      +-------------------------------+-------------------------------+
                                                      |
                                                      v
                      +---------------------------------------------------------------+
                      |                 Autonomous FLISR Controller                   |
                      | 1. Auto-Isolate Faulted Branch (Trip Flanking Breakers)       |
                      | 2. Graph Connectivity & Island Identification (NetworkX)     |
                      | 3. Candidate Tie-Switch Ranking & Multi-Objective Scoring     |
                      | 4. Radiality, Voltage Limits & Ampacity Constraint Validation |
                      | 5. Priority Load Preservation (Protect Critical Nodes)        |
                      +-------------------------------+-------------------------------+
                                                      |
                                                      v
                      +---------------------------------------------------------------+
                      |                 Interactive Streamlit Console                 |
                      | - Live 2D Distribution Topology Map (Plotly Visualizer)       |
                      | - Voltage Sag Curves & Feeder Branch Loading % Profiles       |
                      | - AI Confusion Matrix & Gini Feature Importance Inspector     |
                      | - IEEE 1366 Reliability Metrics & VoLL Avoided Cost Savings   |
                      +---------------------------------------------------------------+
```

---

## ⚡ Theoretical Framework & Mathematical Formulations

### 1. AC Newton-Raphson Power Flow
The non-linear AC power flow equations govern the complex nodal power injections:
$$S_i = P_i + j Q_i = V_i \sum_{k=1}^{N} Y_{ik}^* V_k^*$$

Decomposed into real and reactive nodal power equations:
$$P_i = |V_i| \sum_{k=1}^{N} |V_k| \left( G_{ik} \cos(\theta_i - \theta_k) + B_{ik} \sin(\theta_i - \theta_k) \right)$$
$$Q_i = |V_i| \sum_{k=1}^{N} |V_k| \left( G_{ik} \sin(\theta_i - \theta_k) - B_{ik} \cos(\theta_i - \theta_k) \right)$$

Where:
- $|V_i|, \theta_i$: Voltage magnitude and phase angle at bus $i$.
- $Y_{ik} = G_{ik} + j B_{ik}$: Nodal admittance matrix elements.
- The state vector $x = [\theta_2 \dots \theta_N, |V_2| \dots |V_N]^T$ is solved iteratively via the Jacobian matrix:
$$\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta |V| \end{bmatrix}$$

---

### 2. Physical Fault Modeling & Telemetry Simulation
Faults are modeled as equivalent shunt impedance injections $Z_f = R_f + j X_f$ positioned at a distance $d \in [0.1, 0.9]$ along the target line:
$$P_{\text{fault}} = \frac{V_n^2 \cdot R_f}{R_f^2 + X_f^2}, \quad Q_{\text{fault}} = \frac{V_n^2 \cdot X_f}{R_f^2 + X_f^2}$$

- **Three-Phase Bolted**: $R_f \in [0.05, 1.5]\,\Omega$, highest short-circuit current surges ($6\times - 8\times I_{\text{base}}$).
- **Single Line-to-Ground (SLG)**: $R_f \in [0.5, 12.0]\,\Omega$, asymmetrical ground flashover.
- **Line-to-Line (L-L)**: $R_f \in [0.2, 5.0]\,\Omega$, phase-to-phase contact without ground return.
- **High Impedance Fault (HIF)**: $R_f \in [15.0, 50.0]\,\Omega$, downed conductor on dry ground with minimal current increase but distinctive voltage profile distortion.
- **Sensor PMU Noise**: Gaussian noise $\mathcal{N}(0, \sigma^2)$ is added to simulate instrumentation and measurement errors:
$$\tilde{V}_i = V_i + \mathcal{N}\left(0, (\text{noise}\% \times V_i)^2\right)$$

---

### 3. AI Diagnostic Ensemble
Each observation is mapped to a comprehensive telemetry feature vector:
$$\mathbf{x} = \Big[ V_1, \dots, V_N, \; \Delta V_1, \dots, \Delta V_N, \; I_1, \dots, I_M, \; \Delta I_1, \dots, \Delta I_M \Big] \in \mathbb{R}^{2N + 2M}$$
- For **IEEE 33-Bus** ($N=33, M=37$): $\dim(\mathbf{x}) = 140$ features.
- For **Urban 16-Bus** ($N=16, M=17$): $\dim(\mathbf{x}) = 66$ features.

**Ensemble Architecture**:
1. **Random Forest Classifier**: Classifies operational state into Normal, 3-Phase, SLG, L-L, or HIF.
2. **ExtraTrees Classifier**: Extremely Randomized Trees pinpointing the discrete line ID index ($l \in \{0, \dots, M-1\}$).
3. **Gradient Boosting Regressor**: Continuous estimation of fault distance percentage along the branch.

---

### 4. Graph Topology & Radiality Constraints
Distribution networks must preserve radial topology after reconfiguration to ensure protection relay coordination and avoid circulating loop currents.

Given the energized distribution graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
1. **Acyclic Tree Condition**: The number of branches must equal the number of buses minus the number of infeed substations $k$:
   $$|\mathcal{E}| = |\mathcal{V}| - k$$
2. **Connectivity**: $\forall v \in \mathcal{V}_{\text{restored}}$, there exists a simple path to at least one active substation slack bus $s \in \mathcal{S}$.

---

### 5. Multi-Objective Service Restoration Optimization
When a fault section is isolated, candidate normally-open (NO) tie-switches $\mathcal{T}_{\text{cand}}$ are evaluated via a multi-objective fitness function:
$$\max_{k \in \mathcal{T}_{\text{cand}}} \quad \mathcal{J}(k) = w_1 \cdot \Delta P_{\text{restored}} - w_2 \cdot P_{\text{loss}} + w_3 \cdot V_{\min} - \mathcal{P}_{\text{rad}} - \mathcal{P}_V - \mathcal{P}_I$$

Where:
- $\Delta P_{\text{restored}}$: Total active load restored in MW.
- $P_{\text{loss}}$: Total technical branch losses in kW.
- $V_{\min}$: Minimum nodal voltage magnitude across the restored feeder.
- $\mathcal{P}_{\text{rad}} = 500$ (Radiality penalty if a loop is detected).
- $\mathcal{P}_V = 100$ (Voltage penalty if $V_i < 0.90$ or $V_i > 1.05$ p.u.).
- $\mathcal{P}_I = 100$ (Thermal penalty if branch loading exceeds $100\%$).

---

### 6. IEEE 1366 Reliability Standards & VoLL Economics
The platform quantifies distribution grid resilience using standard IEEE 1366 indices:

$$\text{SAIFI} = \frac{\sum N_i}{N_T} \quad \left[\frac{\text{interruptions}}{\text{customer}}\right]$$
$$\text{SAIDI} = \frac{\sum (r_i \cdot N_i)}{N_T} \quad \left[\frac{\text{outage minutes}}{\text{customer}}\right]$$
$$\text{CAIDI} = \frac{\text{SAIDI}}{\text{SAIFI}} = \frac{\sum (r_i \cdot N_i)}{\sum N_i} \quad [\text{minutes}]$$
$$\text{ENS} = \sum_{i} \left( P_{\text{unserved}, i} \cdot t_{\text{outage}, i} \right) \quad [\text{MWh}]$$
$$\text{ASAI} = \frac{N_T \times 8760 - \sum (r_i \cdot N_i)}{N_T \times 8760} \times 100\%$$

**Economic Outage Valuation**:
$$\text{Avoided Outage Cost} = \Delta \text{ENS} \times \text{VoLL} \quad [\text{USD}]$$
Where **VoLL** (Value of Lost Load) defaults to $\$9,500/\text{MWh}$ per IEEE / regulatory standards.

---

## 🔬 Benchmark Distribution Network Topologies

### 1. IEEE 33-Bus Benchmark (Baran & Wu)
- **Rated Voltage**: $12.66\text{ kV}$
- **Buses**: 33 nodes (1 Substation Slack, 32 Load buses)
- **Active Branches**: 32 normally closed lines
- **Tie-Lines**: 5 normally open tie-switches (Lines 32, 33, 34, 35, 36)
- **Distributed Generation**:
  - Solar PV Alpha at Bus 17 ($0.45\text{ MW}$)
  - BESS Beta at Bus 32 ($0.35\text{ MW} / 1.2\text{ MWh}$, SOC $85\%$)
  - Solar PV Gamma at Bus 24 ($0.30\text{ MW}$)
- **Load Classification**: Critical (Hospital at Bus 7, Data Center at Bus 13), Commercial, and Residential nodes.

### 2. 16-Bus Urban Smart Microgrid
- **Rated Voltage**: $11.0\text{ kV}$
- **Substations**: Dual Grid Infeeds (Substation Alpha at Bus 0, Substation Beta at Bus 15)
- **Feeders**: Two parallel primary feeders with 3 cross-feeder tie-lines (3-12, 5-10, 7-8)
- **Distributed Generation**:
  - Urban Solar Park at Bus 6 ($0.40\text{ MW}$)
  - City BESS at Bus 11 ($0.35\text{ MW} / 1.0\text{ MWh}$)

---

## 🖥️ Interactive Web Dashboard Features

| Tab | Key Visualizations & Analytics |
|---|---|
| **⚡ Live FLISR Console** | Real-time 2D Plotly distribution grid diagram, live operational status badge (Healthy / Fault / Isolated / Restored), single-click fault injection, autonomous FLISR execution, and switching timeline log. |
| **📈 Electrical Analytics** | Bus voltage profile curves ($V_m$ vs Bus ID) comparing Normal, Fault Sag, and Post-Healing states; branch thermal loading bar charts with $100\%$ limit lines; real power loss comparisons; tie-switch optimization ranking matrix. |
| **🧠 AI Diagnostics Hub** | Real-time diagnostic confidence scores, section locator probabilities, confusion matrix heatmap, Gini feature importance bar charts, and on-demand Monte-Carlo retraining sliders. |
| **🛡️ IEEE 1366 Reliability** | Side-by-side comparison of Manual Crew Response ($\sim 120\text{ min}$) vs. Autonomous AI FLISR ($< 2.5\text{ sec}$), SAIDI/SAIFI reduction cards, ENS energy savings, VoLL economic benefit calculator, and downloadable CSV event audit trail. |

---

## 📁 Repository File Structure

```
minor anti/
├── core/
│   ├── __init__.py
│   ├── network_builder.py       # IEEE 33-Bus & 16-Bus Urban Microgrid constructors
│   ├── power_flow.py            # AC Newton-Raphson solver & constraint extraction
│   ├── fault_simulator.py       # Physical fault generator with 4 fault types & PMU noise
│   ├── ai_engine.py             # Scikit-learn RF/ExtraTrees ensemble models & trainer
│   ├── flisr_controller.py      # Autonomous isolation & reconfiguration optimization
│   └── reliability_metrics.py   # IEEE 1366 indices (SAIFI, SAIDI, CAIDI, ENS, VoLL)
├── visualization/
│   ├── __init__.py
│   ├── network_graph.py         # 2D Plotly distribution topology diagram
│   └── dashboard_charts.py      # Plotly curves, loading bars, confusion matrix
├── models/                      # Pre-trained Scikit-Learn model checkpoints (.pkl)
├── .streamlit/
│   └── config.toml              # Streamlit production configuration (dark theme, headless)
├── Dockerfile                   # Production Docker container configuration
├── .dockerignore                # Container ignore rules
├── .gitignore                   # Git version control ignore rules
├── requirements.txt             # Python dependency manifest
├── app.py                       # Master Streamlit Web Application
├── train_initial_models.py      # Automated Monte-Carlo pre-training script
├── verify_system.py             # Full 6-stage end-to-end automated test suite
└── README.md                    # Detailed documentation and project manual
```

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites
- **Python**: 3.10 or 3.11 recommended
- **Operating System**: Windows, Linux, or macOS

### 2. Clone / Open Directory
```bash
cd "c:/Users/bshar/Desktop/minor anti"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
streamlit run app.py
```
> Alternatively, execute directly via `python app.py`.

The application will launch automatically in your browser at:
👉 **`http://localhost:8501`**

---

## 🧪 Automated Verification & Test Suite

The platform includes an automated 6-stage test script verifying all core modules:

```bash
python verify_system.py
```

**Test Verification Stages**:
1. IEEE 33-Bus construction and AC power flow convergence.
2. Physical fault simulation and voltage sag calculation.
3. AI diagnostic inference (classification confidence, section localization, inference runtime).
4. Autonomous FLISR self-healing execution and tie-line switching.
5. IEEE 1366 reliability indices calculation and economic valuation.
6. Plotly graphical figure generation.

---

## 🌐 Production Deployment Guide

### Streamlit Community Cloud (Free & Instant)
1. Push this repository to GitHub.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and select the repository.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

### Docker Deployment
```bash
# Build the production image
docker build -t aegisgrid-flisr .

# Run the container
docker run -p 8501:8501 aegisgrid-flisr
```

---

## 📜 License & Citation

Distributed under the **MIT License**. Built as an intelligent distribution automation simulation demonstrating smart grid self-healing, renewable integration, and machine learning diagnostics.
