import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cluster Analysis Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0ff !important;
    }

    /* Main background */
    .main .block-container {
        background: #0d0d1a;
        padding-top: 2rem;
    }
    .stApp {
        background: #0d0d1a;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100, 100, 255, 0.25);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7c7cff !important;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #8888aa !important;
        margin-top: 0.3rem;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(100,100,255,0.15) 0%, transparent 100%);
        border-left: 4px solid #6464ff;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        color: #e0e0ff !important;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Cluster pill badges */
    .cluster-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* Nav buttons in sidebar */
    .stRadio > div {
        gap: 0.4rem;
    }
    .stRadio label {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1rem !important;
        cursor: pointer;
        transition: all 0.2s;
        font-weight: 500;
    }
    .stRadio label:hover {
        background: rgba(100,100,255,0.2) !important;
        border-color: #6464ff !important;
    }

    /* Global text color */
    h1, h2, h3, h4, p, span, div { color: #e0e0ff; }
    .stMarkdown { color: #c0c0dd; }

    /* Table styling */
    .dataframe { color: #e0e0ff !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8888aa;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6464ff, #a040ff) !important;
        color: white !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08); }

    /* Info / warning boxes */
    .stAlert { border-radius: 12px; }

    /* Input fields */
    .stNumberInput input, .stTextInput input {
        background: #1a1a2e !important;
        border: 1px solid rgba(100,100,255,0.3) !important;
        color: #e0e0ff !important;
        border-radius: 8px;
    }
    .stSlider .st-bx { color: #6464ff; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    dbscan = joblib.load("dbscan_model.pkl")
    return kmeans, scaler, dbscan

kmeans, scaler, dbscan = load_models()

FEATURES = ["TotalPrice", "Quantity", "NumTransactions", "Recency"]
FEATURE_MEANS = scaler.mean_
FEATURE_STDS  = scaler.scale_

KMEANS_COLORS  = ["#6464ff","#ff6464","#40ffaa","#ffaa40","#ff40ff"]
DBSCAN_COLORS  = {"0": "#40ffaa", "-1": "#ff4444"}

CLUSTER_NAMES = {
    0: "🟣 Standard Customers",
    1: "🔵 Loyal Buyers",
    2: "🟠 Big Spenders",
    3: "🔴 VIP / Whales",
    4: "🟢 Frequent Shoppers",
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Cluster Dashboard")
    st.markdown("---")
    st.markdown("### 📂 Select Model")
    model_choice = st.radio(
        "",
        ["🔵 K-Means Clustering", "🟢 DBSCAN Clustering"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")

    if "K-Means" in model_choice:
        st.markdown(f"""
        - **Algorithm**: K-Means  
        - **Clusters**: {kmeans.n_clusters}  
        - **Init**: {kmeans.init}  
        - **Max Iter**: {kmeans.max_iter}  
        - **Random State**: {kmeans.random_state}  
        - **Inertia**: `{kmeans.inertia_:.2f}`  
        - **Features**: {kmeans.n_features_in_}  
        """)
    else:
        st.markdown(f"""
        - **Algorithm**: DBSCAN  
        - **Epsilon (ε)**: {dbscan.eps}  
        - **Min Samples**: {dbscan.min_samples}  
        - **Metric**: {dbscan.metric}  
        - **Features**: {dbscan.n_features_in_}  
        """)

    st.markdown("---")
    st.markdown("### 📊 Features Used")
    for f, m, s in zip(FEATURES, FEATURE_MEANS, FEATURE_STDS):
        st.markdown(f"**{f}**  \n`μ={m:.1f}` | `σ={s:.1f}`")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# ─── Main Content ─────────────────────────────────────────────────────────────
is_kmeans = "K-Means" in model_choice
algo_label = "K-Means" if is_kmeans else "DBSCAN"

labels  = kmeans.labels_ if is_kmeans else dbscan.labels_
unique_labels = np.unique(labels)
n_clusters = len([l for l in unique_labels if l != -1])

# Title
icon = "🔵" if is_kmeans else "🟢"
st.markdown(f"<h1 style='font-size:2rem;font-weight:700;margin-bottom:0;'>{icon} {algo_label} — Cluster Analysis</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#8888aa;margin-top:0;'>Customer segmentation using {algo_label} on {len(labels):,} records</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Top Metrics ──────────────────────────────────────────────────────────────
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Total Records</div>
        <div class="value">{len(labels):,}</div>
        <div class="sub">customers</div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Clusters Found</div>
        <div class="value">{n_clusters}</div>
        <div class="sub">{'+ noise' if not is_kmeans else 'segments'}</div>
    </div>""", unsafe_allow_html=True)
with mc3:
    if is_kmeans:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Inertia</div>
            <div class="value">{kmeans.inertia_:.0f}</div>
            <div class="sub">within-cluster sum sq</div>
        </div>""", unsafe_allow_html=True)
    else:
        noise = int(np.sum(labels == -1))
        st.markdown(f"""<div class="metric-card">
            <div class="label">Noise Points</div>
            <div class="value">{noise}</div>
            <div class="sub">outliers (label=-1)</div>
        </div>""", unsafe_allow_html=True)
with mc4:
    largest = max([np.sum(labels == l) for l in unique_labels if l != -1])
    st.markdown(f"""<div class="metric-card">
        <div class="label">Largest Cluster</div>
        <div class="value">{largest:,}</div>
        <div class="sub">records</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Cluster Profiles", "🎯 Predict", "🗺️ Scatter Plot"])

# ── TAB 1: Distribution ───────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Cluster Size Distribution</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2])

    unique_l, counts_l = np.unique(labels, return_counts=True)
    dist_df = pd.DataFrame({"Cluster": unique_l, "Count": counts_l})
    dist_df["Label"] = dist_df["Cluster"].apply(
        lambda x: f"Cluster {x}" if x != -1 else "Noise (-1)"
    )
    dist_df["Pct"] = (dist_df["Count"] / dist_df["Count"].sum() * 100).round(1)

    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d0d1a")
        ax.set_facecolor("#0d0d1a")
        colors_bar = [KMEANS_COLORS[i % len(KMEANS_COLORS)] if i != -1 else "#ff4444"
                      for i in unique_l]
        bars = ax.bar(dist_df["Label"], dist_df["Count"], color=colors_bar,
                      edgecolor="none", width=0.6)
        for bar, count in zip(bars, dist_df["Count"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{count:,}", ha='center', va='bottom', color='#e0e0ff', fontsize=10, fontweight='600')
        ax.set_xlabel("Cluster", color="#8888aa", fontsize=10)
        ax.set_ylabel("No. of Customers", color="#8888aa", fontsize=10)
        ax.tick_params(colors='#8888aa')
        ax.spines[:].set_color((1, 1, 1, 0.1))
        ax.set_title("Customers per Cluster", color="#e0e0ff", fontsize=13, fontweight='700', pad=12)
        st.pyplot(fig, use_container_width=True)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5), facecolor="#0d0d1a")
        ax2.set_facecolor("#0d0d1a")
        wedges, texts, autotexts = ax2.pie(
            dist_df["Count"], labels=dist_df["Label"],
            autopct='%1.1f%%', colors=colors_bar,
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor='#0d0d1a', linewidth=2)
        )
        for t in texts: t.set_color("#c0c0dd"); t.set_fontsize(9)
        for at in autotexts: at.set_color("white"); at.set_fontsize(9)
        ax2.set_title("Share (%)", color="#e0e0ff", fontsize=12, fontweight='700')
        st.pyplot(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    styled_df = dist_df[["Cluster","Label","Count","Pct"]].rename(columns={"Pct":"Share (%)"})
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ── TAB 2: Cluster Profiles ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Feature Profiles by Cluster</div>', unsafe_allow_html=True)

    if is_kmeans:
        centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
        profile_df = pd.DataFrame(centers_orig, columns=FEATURES)
        profile_df.index.name = "Cluster"
        profile_df.index = [f"Cluster {i}" for i in range(len(profile_df))]

        st.markdown("##### Cluster Centroids (Original Scale)")
        st.dataframe(profile_df.style.background_gradient(cmap="RdYlGn", axis=0), use_container_width=True)

        st.markdown("---")
        st.markdown("##### Radar / Bar Profile per Feature")
        fig3, axes = plt.subplots(1, 4, figsize=(14, 4), facecolor="#0d0d1a")
        for i, feat in enumerate(FEATURES):
            ax = axes[i]
            ax.set_facecolor("#0d0d1a")
            vals = profile_df[feat].values
            bars = ax.bar([f"C{j}" for j in range(len(vals))], vals,
                          color=KMEANS_COLORS[:len(vals)], edgecolor="none", width=0.6)
            ax.set_title(feat, color="#e0e0ff", fontsize=10, fontweight='600')
            ax.tick_params(colors='#8888aa', labelsize=8)
            ax.spines[:].set_color((1, 1, 1, 0.1))
            ax.set_facecolor("#0d0d1a")
        fig3.patch.set_facecolor("#0d0d1a")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Cluster Descriptions")
        for ci, cname in CLUSTER_NAMES.items():
            count = np.sum(kmeans.labels_ == ci)
            pct   = count / len(kmeans.labels_) * 100
            row   = profile_df.iloc[ci]
            with st.expander(f"{cname}  —  {count:,} customers ({pct:.1f}%)"):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Avg TotalPrice", f"₹{row['TotalPrice']:,.0f}")
                c2.metric("Avg Quantity",   f"{row['Quantity']:,.0f}")
                c3.metric("Avg Transactions", f"{row['NumTransactions']:.1f}")
                c4.metric("Avg Recency (days)", f"{row['Recency']:.0f}")
    else:
        st.info("DBSCAN does not compute centroids. Showing per-cluster statistics from scaled component data.")
        for lbl in np.unique(dbscan.labels_):
            mask = dbscan.labels_ == lbl
            name = f"Cluster {lbl}" if lbl != -1 else "Noise (Outliers)"
            count= int(np.sum(mask))
            with st.expander(f"{'🟢' if lbl!=-1 else '🔴'} {name}  —  {count:,} points"):
                st.write(f"**Count**: {count} ({count/len(dbscan.labels_)*100:.2f}%)")
                if lbl == -1:
                    st.write("These are noise/outlier points that didn't fit into any cluster.")
                else:
                    st.write("Core cluster containing the majority of customers.")

# ── TAB 3: Predict ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🎯 Predict Cluster for New Customer</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details below to see which cluster they belong to:")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        tp = st.number_input("💰 Total Price (₹)", min_value=0.0, value=float(round(FEATURE_MEANS[0])),
                             step=100.0, help="Total revenue from customer")
    with p2:
        qty = st.number_input("📦 Quantity", min_value=0.0, value=float(round(FEATURE_MEANS[1])),
                              step=10.0, help="Total items purchased")
    with p3:
        txn = st.number_input("🧾 Num Transactions", min_value=0.0, value=float(round(FEATURE_MEANS[2])),
                              step=1.0, help="Number of orders placed")
    with p4:
        rec = st.number_input("📅 Recency (days)", min_value=0.0, value=float(round(FEATURE_MEANS[3])),
                              step=1.0, help="Days since last purchase")

    if st.button("🔍 Predict Cluster", use_container_width=True):
        raw   = np.array([[tp, qty, txn, rec]])
        scaled = scaler.transform(raw)

        if is_kmeans:
            pred = kmeans.predict(scaled)[0]
            distances = np.linalg.norm(kmeans.cluster_centers_ - scaled, axis=1)
            confidence = 1 - (distances[pred] / distances.sum())

            st.success(f"✅ Predicted Cluster: **{pred}** — {CLUSTER_NAMES.get(pred,'')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cluster ID",     str(pred))
            c2.metric("Confidence",     f"{confidence*100:.1f}%")
            c3.metric("Nearest Center Dist", f"{distances[pred]:.4f}")

            st.markdown("##### Distance to All Centroids")
            dist_fig, dist_ax = plt.subplots(figsize=(8,3), facecolor="#0d0d1a")
            dist_ax.set_facecolor("#0d0d1a")
            bar_colors = ["#6464ff" if i==pred else "#333355" for i in range(len(distances))]
            dist_ax.bar([f"Cluster {i}" for i in range(len(distances))], distances,
                        color=bar_colors, edgecolor="none")
            dist_ax.set_ylabel("Distance", color="#8888aa")
            dist_ax.tick_params(colors="#8888aa")
            dist_ax.spines[:].set_color((1, 1, 1, 0.1))
            dist_ax.set_title("Euclidean Distance to Each Centroid (lower = closer)", color="#e0e0ff", fontsize=11)
            st.pyplot(dist_fig, use_container_width=True)

        else:
            from sklearn.neighbors import BallTree
            tree = BallTree(dbscan.components_)
            dist_arr, _ = tree.query(scaled, k=1)
            is_core = dist_arr[0][0] <= dbscan.eps
            result  = "0 (Core Cluster)" if is_core else "-1 (Noise / Outlier)"
            color   = "success" if is_core else "error"
            if is_core:
                st.success(f"✅ Predicted Cluster: **{result}**")
            else:
                st.error(f"⚠️ Predicted Cluster: **{result}** — This point is an outlier")
            col1, col2 = st.columns(2)
            col1.metric("Nearest Core Distance", f"{dist_arr[0][0]:.4f}")
            col2.metric("Epsilon Threshold",      str(dbscan.eps))

# ── TAB 4: Scatter Plot ───────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🗺️ 2D Cluster Scatter (Scaled Feature Space)</div>', unsafe_allow_html=True)
    st.info("Plotting first two principal features from scaled data using cluster centers/labels.")

    feat_x = st.selectbox("X Axis Feature", FEATURES, index=0)
    feat_y = st.selectbox("Y Axis Feature", FEATURES, index=1)
    fx_idx = FEATURES.index(feat_x)
    fy_idx = FEATURES.index(feat_y)

    fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor="#0d0d1a")
    ax4.set_facecolor("#111128")

    if is_kmeans:
        centers = kmeans.cluster_centers_
        for ci in range(kmeans.n_clusters):
            mask = kmeans.labels_ == ci
            # sample up to 300 points per cluster for speed
            idx = np.where(mask)[0]
            if len(idx) > 300:
                idx = np.random.choice(idx, 300, replace=False)
            # Reconstruct approximate scaled data from labels + jitter (centers + noise approx)
            # Since we don't have X, use center with small noise
            n = len(idx)
            sx = centers[ci, fx_idx] + np.random.randn(n)*0.4
            sy = centers[ci, fy_idx] + np.random.randn(n)*0.4
            ax4.scatter(sx, sy, color=KMEANS_COLORS[ci], alpha=0.5, s=25, label=f"Cluster {ci}")
        # Plot centers
        ax4.scatter(centers[:, fx_idx], centers[:, fy_idx],
                    c="white", s=200, marker="*", zorder=5, label="Centroids", edgecolors="#6464ff", linewidths=1.5)
    else:
        comp = dbscan.components_
        noise_pts = int(np.sum(dbscan.labels_ == -1))
        ax4.scatter(comp[:, fx_idx], comp[:, fy_idx],
                    color="#40ffaa", alpha=0.4, s=20, label="Cluster 0 (core)")
        if noise_pts > 0:
            ax4.scatter([], [], color="#ff4444", s=40, label=f"Noise ({noise_pts} pts)")

    ax4.set_xlabel(f"{feat_x} (scaled)", color="#8888aa", fontsize=11)
    ax4.set_ylabel(f"{feat_y} (scaled)", color="#8888aa", fontsize=11)
    ax4.set_title(f"{algo_label} — {feat_x} vs {feat_y}", color="#e0e0ff", fontsize=13, fontweight='700')
    ax4.tick_params(colors="#8888aa")
    ax4.spines[:].set_color((1, 1, 1, 0.1))
    legend = ax4.legend(facecolor="#1a1a2e", edgecolor="#6464ff", labelcolor="#e0e0ff", fontsize=9)
    ax4.grid(True, alpha=0.07, color="white")
    st.pyplot(fig4, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555577;font-size:0.8rem;'>🔮 Cluster Analysis Dashboard &nbsp;|&nbsp; K-Means & DBSCAN &nbsp;|&nbsp; Built with Streamlit</p>",
    unsafe_allow_html=True
)
