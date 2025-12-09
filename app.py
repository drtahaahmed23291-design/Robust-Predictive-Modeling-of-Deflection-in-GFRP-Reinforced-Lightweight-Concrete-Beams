import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# ============================
# Scientific UI - Styling
# ============================
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #f4f5f7 !important;
    color: #000 !important;
    font-family: "Segoe UI", system-ui, sans-serif;
}

/* Titles */
h1, h2, h3, h4 {
    color: #8B0000 !important;
    font-weight: 700 !important;
}
.stMarkdown, .stHeader, .stSubheader {
    color: #222 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 3px solid #8B0000;
    box-shadow: 2px 0 8px rgba(0,0,0,0.06);
}

/* Labels */
.stNumberInput > label,
.stTextInput > label,
.stSelectbox > label {
    color: #8B0000 !important;
    font-weight: 600 !important;
}

/* Inputs */
.stNumberInput input,
.stTextInput input,
.stSelectbox div[role="button"],
[data-baseweb="select"] div,
[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 6px !important;
    border: 1px solid #999 !important;
}

/* +/- buttons in number input */
[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] svg {
    filter: invert(0) !important;
}

/* Selectbox arrow */
.stSelectbox svg {
    fill: #000 !important;
}

/* Strong highlight box (for Ie and Δmax, and uncertainty summary) */
.special-bold-box {
    background-color: #ffecec !important;
    color: #8B0000 !important;
    border-left: 5px solid #8B0000 !important;
    border-radius: 7px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    font-weight: bold !important;
    padding: 0.8em 0.7em !important;
    margin-bottom: 8px !important;
}

/* Normal result box (for the other results) */
.result-box {
    background-color: #FFEDE5 !important;
    color: #8B0000 !important;
    border-left: 4px solid #FF7043 !important;
    border-radius: 7px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    padding: 0.6em 0.7em !important;
    margin-bottom: 6px !important;
    font-weight: 500 !important;
}

/* DataFrame container */
div[data-testid="stDataFrameContainer"] {
    background-color: #ffffff !important;
    color: #000 !important;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

/* Buttons (Run / Download / Log) */
div.stButton > button,
[data-testid="stDownloadButton"] > button {
    background-color: #8B0000 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.5em 1.4em !important;
    font-size: 1rem !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.18);
    transition: 0.2s ease-in-out;
}
div.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    background-color: #A40000 !important;
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 5px 16px rgba(0,0,0,0.22);
}

/* Checkboxes */
div[data-testid="stCheckbox"] p, 
div[data-testid="stCheckbox"] label, 
[data-baseweb="checkbox"] label, 
[data-baseweb="checkbox"] span {
    color: #000 !important;
    opacity: 1 !important;
}
div[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: #8B0000 !important;
    background-color: #fff !important;
    border: 1px solid #333 !important;
}
[data-baseweb="checkbox"] > label > div:first-child {
    background-color: #fff !important;
    border: 1px solid #333 !important;
    box-shadow: none !important;
}
[data-baseweb="checkbox"] input:checked + div {
    background-color: #8B0000 !important;
    border-color: #8B0000 !important;
}
[data-baseweb="checkbox"] svg {
    fill: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Title
# ===============================
st.title("Robust Predictive Modeling of Deflection in GFRP-Reinforced Lightweight Concrete Beams")
st.markdown(
    "by Nady M. Abdel-Fattah, Taha A. Fesekh, Ehab M. Lotfy, Erfan Abdel-Latif, "
    "and Abdel-Rahman M. Naguib"
)

# =========================
# Helpers
# =========================
def compute_section_and_deflection(params):
    """
    Deterministic calculations and section properties.
    Returns a dict with key results.
    """
    (section_type, Pa, X, L, fc, Ef, Af, rho_f, rho_fb, d, d_prime, b,
     y1, B, y2, t, Es) = params

    # Concrete modulus
    Ec = 4700 * (fc ** 0.5)
    nf = Ef / Ec if Ec != 0 else 0

    # beta_d = 0.05 (rho_f / rho_fb) (Ef/Es + 1), capped at 0.5
    if rho_fb != 0 and Es != 0:
        beta_raw = 0.05 * (rho_f / rho_fb) * (Ef / Es + 1)
    else:
        beta_raw = 0.50
    beta_d = min(beta_raw, 0.50)

    # alpha_c (replacing lambda_e)
    if d != 0 and d_prime != 0:
        Xd_ratio = X / d
        alpha_c = (
            -3.52
            + 0.0793 * Xd_ratio
            + 0.0893 * d_prime
            + 62.5 / d_prime
            + 0.029 * (Xd_ratio ** 2)
            - 0.0118 * Xd_ratio * d_prime
        )
    else:
        Xd_ratio = 0
        alpha_c = 0.55

    alpha_c = min(max(alpha_c, 0.55), 0.95)

    # Section properties
    if section_type == "T-section":
        A1 = b * y1
        A2 = B * y2
        At = A1 + A2 + (nf - 1) * Af
        yt = (
            A1 * (y1 / 2)
            + A2 * (y1 + y2 / 2)
            + (nf - 1) * Af * d_prime
        ) / At if At != 0 else 0
        Ig = (
            (b * y1 ** 3) / 12
            + A1 * (y1 / 2 - yt) ** 2
            + (B * y2 ** 3) / 12
            + A2 * ((y1 + y2 / 2) - yt) ** 2
            + ((nf - 1) * Af) * (yt - d_prime) ** 2
        )
    else:
        A1 = b * t
        A2 = 0
        At = b * t + (nf - 1) * Af
        yt = (
            (b * t ** 2) / 2
            + (nf - 1) * Af * d_prime
        ) / At if At != 0 else 0
        Ig = (
            (b * t ** 3) / 12
            + b * t * (t / 2 - yt) ** 2
            + ((nf - 1) * Af) * (yt - d_prime) ** 2
        )

    # Applied max moment
    Ma = (Pa * 1000) / 2 * X  # N.mm

    # Modulus of rupture and cracking moment
    fr = 0.62 * math.sqrt(fc)
    Mcr = fr * Ig / yt if yt != 0 else 0

    # Neutral axis for cracked section
    if section_type == "T-section":
        a = B / 2
    else:
        a = b / 2
    b_quad = nf * Af
    c = -(nf * Af) * d
    delta = b_quad ** 2 - 4 * a * c
    if delta < 0 or a == 0:
        Z = 0
    else:
        Z = (-b_quad + math.sqrt(delta)) / (2 * a)

    # Cracked moment of inertia
    if section_type == "T-section":
        Icr = B * Z ** 3 / 3 + (nf * Af) * (d - Z) ** 2
    else:
        Icr = b * Z ** 3 / 3 + (nf * Af) * (d - Z) ** 2

    # Effective I
    if Ma < Mcr:
        Ie = Ig
    else:
        ratio = (Mcr / Ma) ** 3 if Ma > 0 else 0
        Ie = beta_d * ratio * Ig + alpha_c * (1 - ratio) * Icr
        Ie = min(Ie, Ig)

    # Deflection
    if Ec != 0 and Ie != 0:
        delta_max = (Pa * 1000 * X) / (48 * Ec * Ie) * (3 * L ** 2 - 4 * X ** 2)
    else:
        delta_max = 0.0

    return {
        "A1": A1, "A2": A2, "Ec": Ec, "nf": nf, "beta_d": beta_d, "alpha_c": alpha_c,
        "yt": yt, "At": At, "Ig": Ig, "Ma": Ma, "fr": fr, "Mcr": Mcr,
        "Z": Z, "Icr": Icr, "Ie": Ie, "delta_max": delta_max
    }


def ci_from_samples(samples, conf=0.95):
    """Return mean, lower, upper for a two-sided percentile CI."""
    if len(samples) == 0:
        return 0.0, 0.0, 0.0
    s = np.asarray(samples)
    mean = float(np.mean(s))
    alpha = 1 - conf
    low = float(np.percentile(s, 100 * (alpha / 2)))
    high = float(np.percentile(s, 100 * (1 - alpha / 2)))
    return mean, low, high


def draw_positive_normal(mu, rel_sigma, size=1, min_val=1e-9):
    """Sample normal with relative std; truncate to positive."""
    sigma = abs(mu) * rel_sigma
    samp = np.random.normal(mu, sigma, size=size)
    return np.clip(samp, min_val, None)


# =========================
# Inputs
# =========================
section_type = st.selectbox("Section Type", ["T-section", "R-section"])

# Default values
if section_type == "T-section":
    default_Pa, default_X, default_L = 186.6, 750.0, 1900.0
    default_fc, default_Ef, default_Af = 28.00, 50000.0, 314.0
    default_rho_f, default_rho_fb = 0.57, 0.24
    default_d, default_d_prime = 275.0, 25.0
    default_b, default_y1, default_B, default_y2 = 200.0, 225.0, 600.0, 75.0
    default_t = None
else:
    default_Pa, default_X, default_L = 166.94, 750.0, 1900.0
    default_fc, default_Ef, default_Af = 28.00, 50000.0, 314.0
    default_rho_f, default_rho_fb = 0.57, 0.24
    default_d, default_d_prime = 275.0, 25.0
    default_b, default_t = 200.0, 300.0
    default_y1 = default_B = default_y2 = None

st.subheader("Input Data")
col1, col2, col3, col4 = st.columns(4)
with col1:
    Pa = st.number_input("Applied Load (Pa, kN)", value=default_Pa)
with col2:
    X = st.number_input("Flexural-shear span X (mm)", value=default_X)
with col3:
    L = st.number_input("Beam Length L (mm)", value=default_L)
with col4:
    fc = st.number_input("Concrete compressive strength f'c (MPa)", value=default_fc)

col5, col6, col7, col8 = st.columns(4)
with col5:
    Ef = st.number_input("Modulus of elasticity of FRP Ef (MPa)", value=default_Ef)
with col6:
    Af = st.number_input("Area of FRP reinforcement Af (mm2)", value=default_Af)
with col7:
    rho_f = st.number_input("Reinforcement ratio rho_f", value=default_rho_f)
with col8:
    rho_fb = st.number_input("Balanced reinforcement ratio rho_fb", value=default_rho_fb)

colEs1, colEs2 = st.columns(2)
with colEs1:
    Es = st.number_input("Modulus of elasticity of steel Es (MPa)", value=200000.0)
with colEs2:
    pass

col9, col10 = st.columns(2)
with col9:
    d = st.number_input("Effective depth d (mm)", value=default_d)
with col10:
    d_prime = st.number_input("Concrete Cover d' (mm)", value=default_d_prime)

if section_type == "T-section":
    colT1, colT2, colT3, colT4 = st.columns(4)
    with colT1:
        b = st.number_input("Width of flange b (mm)", value=200.0)
    with colT2:
        y1 = st.number_input("Height of flange y1 (mm)", value=225.0)
    with colT3:
        B = st.number_input("Width of web B (mm)", value=600.0)
    with colT4:
        y2 = st.number_input("Height of web y2 (mm)", value=75.0)
    t = None
else:
    colR1, colR2 = st.columns(2)
    with colR1:
        b = st.number_input("Width b (mm)", value=200.0)
    with colR2:
        t = st.number_input("Height t (mm)", value=300.0)
    y1 = B = y2 = None

# =========================
# Uncertainty controls
# =========================
st.subheader("Uncertainty (Optional)")
enable_unc = st.checkbox("Enable uncertainty analysis (Monte Carlo)", value=False)
if enable_unc:
    colu1, colu2, colu3, colu4 = st.columns(4)
    with colu1:
        u_Pa = st.number_input("Pa uncertainty ±% (rel. std)", value=5.0, min_value=0.0, step=0.5)
    with colu2:
        u_fc = st.number_input("f'c uncertainty ±% (rel. std)", value=7.5, min_value=0.0, step=0.5)
    with colu3:
        u_Ef = st.number_input("Ef uncertainty ±% (rel. std)", value=5.0, min_value=0.0, step=0.5)
    with colu4:
        u_Af = st.number_input("Af uncertainty ±% (rel. std)", value=2.0, min_value=0.0, step=0.5)

    colu5, colu6 = st.columns(2)
    with colu5:
        Nsim = st.number_input("Number of simulations", value=2000, min_value=100, step=100)
    with colu6:
        conf = st.selectbox("Confidence level", [0.80, 0.90, 0.95, 0.99], index=2)
else:
    u_Pa = u_fc = u_Ef = u_Af = 0.0
    Nsim = 0
    conf = 0.95

# =========================
# Deterministic computation
# =========================
params = (
    section_type, Pa, X, L, fc, Ef, Af, rho_f, rho_fb,
    d, d_prime, b,
    y1 if y1 is not None else 0.0,
    B if B is not None else 0.0,
    y2 if y2 is not None else 0.0,
    t if t is not None else 0.0,
    Es
)

det = compute_section_and_deflection(params)

# =========================
# UI - Run
# =========================
if st.button("Run"):
    # Deterministic results (cards)
    results = []
    if section_type == "T-section":
        A1 = b * y1
        A2 = B * y2
        results.append(f"Area of web A1 = {A1:.2f} mm²")
        results.append(f"Area of flange A2 = {A2:.2f} mm²")
    else:
        A1 = b * t
        results.append(f"Area = {A1:.2f} mm²")

    results.extend([
        f"Elastic modulus of concrete Ec = {det['Ec']:.2f} MPa",
        f"Modular ratio nf (Ef/Ec) = {det['nf']:.3f} (calculated)",
        f"Reduction coefficient βd = {det['beta_d']:.3f}",
        f"Reduction factor α_c = {det['alpha_c']:.3f} (0.55 ≤ α_c ≤ 0.95)",
        f"Neutral axis depth y_t = {det['yt']:.3f} mm",
        f"Equivalent area A_t = {det['At']:.3f} mm²",
        f"Ig (Gross moment of inertia) = {det['Ig']:.3f} mm⁴",
        f"Maximum moment Ma = {det['Ma']:.2f} N·mm",
        f"Modulus of rupture fr = {det['fr']:.3f} MPa",
        f"Cracking moment Mcr = {det['Mcr']:.2f} N·mm",
        f"Compression zone depth Z = {det['Z']:.3f} mm",
        f"Icr (cracked moment of inertia) = {det['Icr']:.3f} mm⁴",
        f"Effective moment of inertia Ie = {det['Ie']:.3f} mm⁴",
        f"Maximum Deflection ∆max = {det['delta_max']:.5f} mm"
    ])

    cols_per_row = 4
    idx_Ie = len(results) - 2
    idx_defl = len(results) - 1

    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            res_idx = i + idx
            if res_idx < len(results):
                text = results[res_idx]
                if res_idx in (idx_Ie, idx_defl):
                    # Strong red box for Ie and Δmax
                    col.markdown(
                        f'<div class="special-bold-box">{text}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Softer peach box for other results
                    col.markdown(
                        f'<div class="result-box">{text}</div>',
                        unsafe_allow_html=True
                    )

    # Instead of st.success
    st.markdown(
        '<div class="result-box">Run completed successfully.</div>',
        unsafe_allow_html=True
    )

    # =========================
    # Uncertainty analysis (Monte Carlo)
    # =========================
    if enable_unc:
        Pa_s = draw_positive_normal(Pa, u_Pa / 100.0, size=Nsim)
        fc_s = draw_positive_normal(fc, u_fc / 100.0, size=Nsim)
        Ef_s = draw_positive_normal(Ef, u_Ef / 100.0, size=Nsim)
        Af_s = draw_positive_normal(Af, u_Af / 100.0, size=Nsim)

        delta_samples = []
        Ie_samples = []

        for i_sim in range(Nsim):
            p = (
                section_type, float(Pa_s[i_sim]), X, L,
                float(fc_s[i_sim]), float(Ef_s[i_sim]), float(Af_s[i_sim]),
                rho_f, rho_fb, d, d_prime, b,
                y1 if y1 is not None else 0.0,
                B if B is not None else 0.0,
                y2 if y2 is not None else 0.0,
                t if t is not None else 0.0,
                Es
            )
            out = compute_section_and_deflection(p)
            delta_samples.append(out["delta_max"])
            Ie_samples.append(out["Ie"])

        d_mean, d_low, d_high = ci_from_samples(delta_samples, conf=float(conf))
        i_mean, i_low, i_high = ci_from_samples(Ie_samples, conf=float(conf))

        st.subheader("Uncertainty Summary")
        coluA, coluB = st.columns(2)
        with coluA:
            st.markdown(
                f'<div class="special-bold-box">∆max = {d_mean:.5f} mm '
                f'(CI {int(conf*100)}%: {d_low:.5f} – {d_high:.5f})</div>',
                unsafe_allow_html=True
            )
        with coluB:
            st.markdown(
                f'<div class="special-bold-box">Ie = {i_mean:.3f} mm⁴ '
                f'(CI {int(conf*100)}%: {i_low:.3f} – {i_high:.3f})</div>',
                unsafe_allow_html=True
            )

        # Histogram of delta_max samples (dark red)
        fig_hist, axh = plt.subplots(figsize=(7, 4))
        axh.hist(delta_samples, bins=40, edgecolor='black', color="darkred")
        axh.set_xlabel("∆max (mm)")
        axh.set_ylabel("Frequency")
        axh.set_title(
            f"Monte Carlo Distribution of ∆max ({Nsim} sims, CI {int(conf*100)}%)"
        )
        st.pyplot(fig_hist)

        # Download samples as CSV
        df_sims = pd.DataFrame({
            "delta_max_mm": delta_samples,
            "Ie_mm4": Ie_samples,
            "Pa_kN": Pa_s,
            "fc_MPa": fc_s,
            "Ef_MPa": Ef_s,
            "Af_mm2": Af_s
        })
        csv_bytes = df_sims.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Monte Carlo samples (CSV)",
            data=csv_bytes,
            file_name="uncertainty_samples.csv",
            mime="text/csv"
        )

    # =========================
    # Load–Deflection curve (deterministic)
    # =========================
    smooth_percentages = np.linspace(0, 100, 51)
    smooth_loads = Pa * (smooth_percentages / 100)
    smooth_deflections = []
    Mcr = det["Mcr"]
    Ig = det["Ig"]
    beta_d = det["beta_d"]
    alpha_c = det["alpha_c"]
    Icr = det["Icr"]
    Ec_val = det["Ec"]

    for load in smooth_loads:
        Ma_i = (load * 1000) / 2 * X
        smoothing_range = 0.1 * Mcr if Mcr != 0 else 1
        x_smooth = (Ma_i - Mcr) / smoothing_range if smoothing_range != 0 else 0
        w_smooth = 0.5 * (1 + np.tanh(x_smooth))
        if Ma_i < 0.01:
            Ie_i = Ig
        else:
            ratio_i = (Mcr / Ma_i) ** 3 if Ma_i > 0 else 1
            Ie_uncracked = Ig
            Ie_cracked = beta_d * ratio_i * Ig + alpha_c * (1 - ratio_i) * Icr
            Ie_i = (1 - w_smooth) * Ie_uncracked + w_smooth * Ie_cracked
            Ie_i = min(Ie_i, Ig)
        if Ec_val != 0 and Ie_i != 0:
            delta_i = (load * 1000 * X) / (48 * Ec_val * Ie_i) * (3 * L ** 2 - 4 * X ** 2)
            if (delta_i < 0) or math.isnan(delta_i) or math.isinf(delta_i):
                delta_i = 0
        else:
            delta_i = 0
        smooth_deflections.append(delta_i)
    smooth_deflections[0] = 0

    main_percentages = np.arange(0, 110, 10)
    main_loads = Pa * (main_percentages / 100)
    main_deflections = []
    for load in main_loads:
        Ma_i = (load * 1000) / 2 * X
        smoothing_range = 0.1 * Mcr if Mcr != 0 else 1
        x_smooth = (Ma_i - Mcr) / smoothing_range if smoothing_range != 0 else 0
        w_smooth = 0.5 * (1 + np.tanh(x_smooth))
        if Ma_i < 0.01:
            Ie_i = Ig
        else:
            ratio_i = (Mcr / Ma_i) ** 3 if Ma_i > 0 else 1
            Ie_uncracked = Ig
            Ie_cracked = beta_d * ratio_i * Ig + alpha_c * (1 - ratio_i) * Icr
            Ie_i = (1 - w_smooth) * Ie_uncracked + w_smooth * Ie_cracked
            Ie_i = min(Ie_i, Ig)
        if Ec_val != 0 and Ie_i != 0:
            delta_i = (load * 1000 * X) / (48 * Ec_val * Ie_i) * (3 * L ** 2 - 4 * X ** 2)
            if (delta_i < 0) or math.isnan(delta_i) or math.isinf(delta_i):
                delta_i = 0
        else:
            delta_i = 0
        main_deflections.append(delta_i)
    main_deflections[0] = 0

    # Table for load–deflection
    df = pd.DataFrame({
        "Load (kN)": [round(l, 2) for l in Pa * (np.arange(0, 101, 5) / 100)],
        "Deflection (mm)": [
            round(dv, 5)
            for dv in np.interp(
                Pa * (np.arange(0, 101, 5) / 100),
                smooth_loads,
                smooth_deflections
            )
        ]
    })

    st.markdown("#### Load-Deflection Curve")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Load-Deflection Data')
    output.seek(0)
    st.download_button(
        label="Download Table as Excel",
        data=output,
        file_name='load_deflection_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Plot load–deflection (blue)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.plot(
        smooth_deflections,
        smooth_loads,
        linestyle="-",
        marker=".",
        markersize=10,
        linewidth=2.5,
        color="blue"
    )
    ax.plot(
        main_deflections,
        main_loads,
        marker="p",
        markersize=10,
        linestyle="None",
        color="blue"
    )
    for defl, load, pct in zip(main_deflections, main_loads, main_percentages):
        if pct >= 20:
            ax.hlines(load, 0, defl, linestyles='dotted', linewidth=1)
            ax.vlines(defl, 0, load, linestyles='dotted', linewidth=1)
            ax.text(defl + 0.7, load, f"{int(pct)}%", va='center', fontsize=11, color="blue")
    ax.set_xlabel("Deflection (mm)")
    ax.set_ylabel("Load (kN)")
    ax.set_title("Load-Deflection Curve")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend().remove()
    st.pyplot(fig)

# ===== Step-by-step Log =====
if 'show_log' not in st.session_state:
    st.session_state.show_log = False

col_log1, col_log2 = st.columns([1, 1])
with col_log1:
    if st.button("Show step-by-step calculations (Log)"):
        st.session_state.show_log = True
with col_log2:
    if st.button("Clear Log"):
        st.session_state.show_log = False

if st.session_state.show_log:
    st.subheader("Step-by-step Calculation Log")
    logs = []
    fc_val = float(fc)
    Ef_val = float(Ef)
    Es_val = float(Es)
    Ec_val = 4700 * (fc_val ** 0.5)
    nf_val = Ef_val / Ec_val if Ec_val != 0 else 0

    if section_type == "T-section":
        A1 = b * y1
        A2 = B * y2
        At = A1 + A2 + (nf_val - 1) * Af
        yt_val = (
            A1 * (y1 / 2)
            + A2 * (y1 + y2 / 2)
            + (nf_val - 1) * Af * d_prime
        ) / At if At != 0 else 0
        Ig_val = (
            (b * y1 ** 3) / 12
            + A1 * (y1 / 2 - yt_val) ** 2
            + (B * y2 ** 3) / 12
            + A2 * ((y1 + y2 / 2) - yt_val) ** 2
            + ((nf_val - 1) * Af) * (yt_val - d_prime) ** 2
        )
    else:
        A1 = b * t
        A2 = 0
        At = b * t + (nf_val - 1) * Af
        yt_val = (
            (b * t ** 2) / 2
            + (nf_val - 1) * Af * d_prime
        ) / At if At != 0 else 0
        Ig_val = (
            (b * t ** 3) / 12
            + b * t * (t / 2 - yt_val) ** 2
            + ((nf_val - 1) * Af) * (yt_val - d_prime) ** 2
        )

    if rho_fb != 0 and Es_val != 0:
        beta_raw_val = 0.05 * (rho_f / rho_fb) * (Ef_val / Es_val + 1)
    else:
        beta_raw_val = 0.50
    beta_d_val = min(beta_raw_val, 0.50)

    if d != 0 and d_prime != 0:
        Xd_ratio = X / d
        alpha_c_val = (
            -3.52
            + 0.0793 * Xd_ratio
            + 0.0893 * d_prime
            + 62.5 / d_prime
            + 0.029 * (Xd_ratio ** 2)
            - 0.0118 * Xd_ratio * d_prime
        )
    else:
        Xd_ratio = 0
        alpha_c_val = 0.55
    alpha_c_val = min(max(alpha_c_val, 0.55), 0.95)

    Ma_val = (Pa * 1000) / 2 * X
    fr_val = 0.62 * math.sqrt(fc_val)
    Mcr_val = fr_val * Ig_val / yt_val if yt_val != 0 else 0

    if section_type == "T-section":
        a_val = B / 2
    else:
        a_val = b / 2
    b_quad_val = nf_val * Af
    c_val = - (nf_val * Af) * d
    disc = b_quad_val ** 2 - 4 * a_val * c_val
    Z_val = 0 if (disc < 0 or a_val == 0) else (-b_quad_val + math.sqrt(disc)) / (2 * a_val)
    Icr_val = (
        B * Z_val ** 3 / 3 + (nf_val * Af) * (d - Z_val) ** 2
    ) if section_type == "T-section" else (
        b * Z_val ** 3 / 3 + (nf_val * Af) * (d - Z_val) ** 2
    )

    if Ma_val < Mcr_val:
        Ie_val = Ig_val
    else:
        ratio_val = (Mcr_val / Ma_val) ** 3 if Ma_val > 0 else 0
        Ie_val = beta_d_val * ratio_val * Ig_val + alpha_c_val * (1 - ratio_val) * Icr_val
        Ie_val = min(Ie_val, Ig_val)
    delta_max_val = (
        (Pa * 1000 * X) / (48 * Ec_val * Ie_val) * (3 * L ** 2 - 4 * X ** 2)
        if (Ec_val != 0 and Ie_val != 0) else 0
    )

    logs.append(f"1) Ec = 4700 * sqrt({fc_val}) = {Ec_val:.2f} MPa")
    logs.append(f"2) nf = Ef / Ec = {Ef_val} / {Ec_val:.2f} = {nf_val:.3f}")
    if section_type == "T-section":
        logs.append(f"3) Web area A1 = {b} * {y1} = {A1:.2f} mm²")
        logs.append(f"4) Flange area A2 = {B} * {y2} = {A2:.2f} mm²")
        logs.append(f"5) At = A1 + A2 + (nf - 1) * Af = {At:.2f} mm²")
        logs.append(f"6) yt = {yt_val:.2f} mm")
        logs.append(f"7) Ig = {Ig_val:.2f} mm⁴")
    else:
        logs.append(f"3) Area = {b} * {t} = {A1:.2f} mm²")
        logs.append(f"4) At = {At:.2f} mm²")
        logs.append(f"5) yt = {yt_val:.2f} mm")
        logs.append(f"6) Ig = {Ig_val:.2f} mm⁴")

    logs.append(
        f"8) βd (raw) = 0.05 * ({rho_f}/{rho_fb}) * (Ef/Es + 1) = {beta_raw_val:.3f} "
        f"→ capped = {beta_d_val:.3f}"
    )
    logs.append(f"9) α_c (capped 0.55–0.95) = {alpha_c_val:.3f}")
    logs.append(f"10) Ma = (({Pa} × 1000) / 2) × {X} = {Ma_val:.2f} N·mm")
    logs.append(f"11) fr = 0.62 * sqrt({fc_val}) = {fr_val:.3f} MPa")
    logs.append(f"12) Mcr = fr * Ig / yt = {Mcr_val:.2f} N·mm")
    if Ma_val < Mcr_val:
        logs.append(f"Ma < Mcr ⇒ Uncracked ⇒ Ie = Ig = {Ig_val:.2f} mm⁴")
    else:
        logs.append(f"13) (Mcr/Ma)^3 = ({Mcr_val:.2f}/{Ma_val:.2f})^3 = {(Mcr_val/Ma_val)**3:.3f}")
        logs.append(f"14) Z = {Z_val:.3f} mm")
        logs.append(f"15) Icr = {Icr_val:.2f} mm⁴")
        logs.append(f"16) Ie = {Ie_val:.2f} mm⁴")
        logs.append(f"17) ∆max = {delta_max_val:.5f} mm")

    for line in logs:
        st.write(line)

    step_text = "\n".join(logs)
    st.download_button(
        label="Download Step-by-step Log (TXT)",
        data=step_text,
        file_name="step_by_step_log.txt",
        mime="text/plain"
    )
