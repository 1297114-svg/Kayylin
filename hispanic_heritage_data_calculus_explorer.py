import datetime
import io
import math
import re
import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Optional ReportLab import with fallback flag
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Streamlit Page Config
st.set_page_config(
    page_title="Hispanic Heritage Data Explorer & Math Lab 2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# CONSTANTS & METADATA
# ------------------------------------------------------------------------------
CURRENT_YEAR = datetime.datetime.now().year
START_YEAR = CURRENT_YEAR - 70
END_YEAR = CURRENT_YEAR - 1

COUNTRIES = {
    "Mexico": "MX",
    "Brazil": "BR",
    "Argentina": "AR",
    "Colombia": "CO",
    "Chile": "CL",
    "Peru": "PE",
    "Venezuela": "VE",
    "Ecuador": "EC",
    "Uruguay": "UY"
}

INDICATORS = {
    "Life Expectancy": {
        "code": "SP.DYN.LE00.IN",
        "units": "years",
        "name": "Life Expectancy at Birth",
        "desc": "Life expectancy at birth indicates the number of years a newborn infant would live if prevailing patterns of mortality at the time of its birth were to stay the same throughout its life."
    },
    "Unemployment Rate": {
        "code": "SL.UEM.TOTL.NE.ZS",
        "units": "% of total labor force",
        "name": "Unemployment, Total (National Estimate)",
        "desc": "Unemployment refers to the share of the labor force that is without work but available for and seeking employment (national estimate)."
    },
    "GDP per Capita": {
        "code": "NY.GDP.PCAP.CD",
        "units": "current US$",
        "name": "GDP per Capita",
        "desc": "Gross domestic product per capita is gross domestic product divided by midyear population. (Note: Not average wealth or household income)."
    },
    "Crude Birth Rate": {
        "code": "SP.DYN.CBRT.IN",
        "units": "per 1,000 people",
        "name": "Birth Rate, Crude",
        "desc": "Crude birth rate indicates the number of live births occurring during the year, per 1,000 population at midyear. (Note: Rate, not total birth count)."
    },
    "Net Migration": {
        "code": "SM.POP.NETM",
        "units": "total net people",
        "name": "Net Migration",
        "desc": "Net migration is the net total of migrants during the period (immigrants minus emigrants). (Note: Not an emigration count)."
    },
    "Intentional Homicide Rate": {
        "code": "VC.IHR.PSRC.P5",
        "units": "per 100,000 people",
        "name": "Intentional Homicides",
        "desc": "Intentional homicides are estimates of unlawful deaths purposefully inflicted on a person by another person per 100,000 people."
    },
    "Consumer Price Inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "units": "annual %",
        "name": "Inflation, Consumer Prices",
        "desc": "Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services."
    },
    "Central Government Debt": {
        "code": "GC.DOD.TOTL.GD.ZS",
        "units": "% of GDP",
        "name": "Central Government Debt, Total",
        "desc": "Debt is the entire stock of direct government fixed term contractual obligations to others outstanding on a particular date."
    },
    "Total Reserves (incl. Gold)": {
        "code": "FI.RES.TOTL.CD",
        "units": "current US$",
        "name": "Total Reserves (includes gold)",
        "desc": "Total reserves comprise holdings of monetary gold, special drawing rights, reserves of IMF members, and holdings of foreign exchange."
    },
    "External Balance on Goods/Services": {
        "code": "NE.RSB.GNFS.ZS",
        "units": "% of GDP",
        "name": "External Balance on Goods and Services",
        "desc": "External balance on goods and services (formerly resource balance) equals exports of goods and services minus imports of goods and services."
    }
}

CONTEXT_PRESETS = {
    "Historical Data Fit": {
        "x_var": "t (Years from Base)",
        "y_var": "Indicator Value",
        "units": "Units",
        "desc": "Polynomial curve of best fit fitted to World Bank historical country indicators.",
        "visual": "business"
    },
    "Business": {
        "x_var": "Time",
        "y_var": "Revenue",
        "units": "Thousand USD",
        "desc": "Corporate quarterly revenue analysis showing growth momentum and turning points.",
        "visual": "business"
    },
    "Engineering": {
        "x_var": "Time",
        "y_var": "Stress Load",
        "units": "MPa",
        "desc": "Structural material stress deflection under dynamic operational load.",
        "visual": "projectile"
    },
    "Medical": {
        "x_var": "Time",
        "y_var": "Drug Concentration",
        "units": "mg/L",
        "desc": "Blood plasma pharmaceutical concentration post-administration over time.",
        "visual": "drug"
    },
    "Sports Science": {
        "x_var": "Time",
        "y_var": "Joint Angle / Load",
        "units": "Degrees / N",
        "desc": "Biomechanics joint flexion and movement dynamics during gait cycle.",
        "visual": "prosthetic"
    },
    "Biology": {
        "x_var": "Time",
        "y_var": "Population",
        "units": "Cells/mL",
        "desc": "Bacterial colony growth curve in a controlled nutrient agar environment.",
        "visual": "population"
    },
    "Bioengineering": {
        "x_var": "Time",
        "y_var": "Bioreactor Yield",
        "units": "g/L",
        "desc": "Enzymatic conversion efficiency rate in continuous bio-reaction.",
        "visual": "population"
    },
    "Economic Forecasting": {
        "x_var": "Time",
        "y_var": "Economic Index",
        "units": "Points",
        "desc": "Macroeconomic leading performance index forecasting expansion and contraction.",
        "visual": "business"
    },
    "Weather": {
        "x_var": "Time",
        "y_var": "Temperature",
        "units": "°C",
        "desc": "Diurnal local surface air temperature oscillation model.",
        "visual": "weather"
    },
    "Custom": {
        "x_var": "x",
        "y_var": "y",
        "units": "units",
        "desc": "User defined mathematical model and variable context.",
        "visual": "business"
    }
}

# ------------------------------------------------------------------------------
# DATA FETCHING & CACHING
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank_data(country_code, indicator_code, start_year, end_year):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {
        "date": f"{start_year}:{end_year}",
        "format": "json",
        "per_page": 1000
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        if not isinstance(data, list) or len(data) < 2 or data[1] is None:
            return pd.DataFrame(columns=["year", "value"]), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "No data returned from API."
        
        records = []
        for item in data[1]:
            val = item.get("value")
            yr = item.get("date")
            if val is not None and yr is not None:
                try:
                    records.append({"year": int(yr), "value": float(val)})
                except ValueError:
                    continue
                    
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("year").reset_index(drop=True)
        retrieval_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return df, retrieval_date, None
    except Exception as e:
        return pd.DataFrame(columns=["year", "value"]), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(e)

# ------------------------------------------------------------------------------
# MATH & POLYNOMIAL FITTING FUNCTIONS
# ------------------------------------------------------------------------------
def fit_polynomial(df, degree):
    if df is None or len(df) < degree + 1:
        return None
    
    years = df['year'].values
    vals = df['value'].values
    base_year = int(years.min())
    t = years - base_year
    
    # Fit polynomial on centered time t
    coeffs = np.polyfit(t, vals, degree) # highest power first: a_d * t^d + ... + a_0
    poly_func = np.poly1d(coeffs)
    
    # Fitted values and residuals
    y_pred = poly_func(t)
    residuals = vals - y_pred
    mae = np.mean(np.abs(residuals))
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((vals - np.mean(vals))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Construct Equation String
    terms = []
    deg = degree
    for c in coeffs:
        if abs(c) > 1e-12:
            sign = "+" if c >= 0 and len(terms) > 0 else ("-" if c < 0 and len(terms) > 0 else "")
            val_str = f"{abs(c):.4e}" if abs(c) < 1e-3 or abs(c) > 1e4 else f"{abs(c):.4f}"
            if deg == 0:
                terms.append(f"{sign} {val_str}".strip())
            elif deg == 1:
                terms.append(f"{sign} {val_str}*t".strip())
            else:
                terms.append(f"{sign} {val_str}*t^{deg}".strip())
        deg -= 1
    
    eq_str = "f(t) = " + " ".join(terms) if terms else "f(t) = 0"
    
    return {
        "coeffs": coeffs,
        "poly_func": poly_func,
        "base_year": base_year,
        "degree": degree,
        "r2": r2,
        "mae": mae,
        "residuals": residuals,
        "eq_str": eq_str,
        "obs_count": len(df),
        "hist_min": years.min(),
        "hist_max": years.max(),
        "mean_val": np.mean(vals),
        "t": t,
        "years": years,
        "vals": vals
    }

def fit_holdout_model(df, degree, holdout_count=5):
    if len(df) <= degree + holdout_count:
        return None
    train_df = df.iloc[:-holdout_count]
    test_df = df.iloc[-holdout_count:]
    
    fit_res = fit_polynomial(train_df, degree)
    if fit_res is None:
        return None
    
    test_t = test_df['year'].values - fit_res['base_year']
    test_preds = fit_res['poly_func'](test_t)
    test_mae = np.mean(np.abs(test_df['value'].values - test_preds))
    
    ss_res = np.sum((test_df['value'].values - test_preds)**2)
    ss_tot = np.sum((test_df['value'].values - np.mean(test_df['value'].values))**2)
    test_r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        "train_count": len(train_df),
        "test_count": len(test_df),
        "test_mae": test_mae,
        "test_r2": test_r2
    }

def evaluate_custom_function(func_str, x):
    """Safely evaluate arbitrary single-variable mathematical expression."""
    safe_dict = {
        "x": x,
        "t": x,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "e": np.e,
        "abs": np.abs,
        "power": np.power
    }
    # Pre-process math notation e.g., ^ -> **
    cleaned = func_str.replace("^", "**")
    # Insert implicit multiplications e.g., 2x -> 2*x, 3sin -> 3*sin
    cleaned = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', cleaned)
    try:
        val = eval(cleaned, {"__builtins__": None}, safe_dict)
        return float(val) if np.isscalar(val) else val
    except Exception:
        return np.nan

# ------------------------------------------------------------------------------
# CALCULUS NUMERICAL ANALYSIS ENGINE
# ------------------------------------------------------------------------------
def numerical_derivatives(func, x, h=1e-5):
    """Compute 1st and 2nd derivatives using central difference formula."""
    try:
        f_x = func(x)
        f_plus = func(x + h)
        f_minus = func(x - h)
        
        if np.isnan(f_x) or np.isnan(f_plus) or np.isnan(f_minus):
            return np.nan, np.nan
        
        d1 = (f_plus - f_minus) / (2 * h)
        d2 = (f_plus - 2 * f_x + f_minus) / (h ** 2)
        return d1, d2
    except Exception:
        return np.nan, np.nan

def analyze_calculus_features(func, domain_min, domain_max, num_points=500):
    xs = np.linspace(domain_min, domain_max, num_points)
    ys = []
    d1s = []
    d2s = []
    
    for x in xs:
        y = func(x)
        d1, d2 = numerical_derivatives(func, x)
        ys.append(y)
        d1s.append(d1)
        d2s.append(d2)
        
    ys = np.array(ys)
    d1s = np.array(d1s)
    d2s = np.array(d2s)
    
    # Critical Points: d1 crosses 0 or d1 is NaN
    critical_points = []
    for i in range(1, len(xs) - 1):
        if not np.isnan(d1s[i-1]) and not np.isnan(d1s[i+1]):
            if d1s[i-1] * d1s[i+1] <= 0: # sign change
                x_crit = xs[i]
                d1_val, d2_val = numerical_derivatives(func, x_crit)
                y_val = func(x_crit)
                pt_type = "Local Minimum" if d2_val > 0 else ("Local Maximum" if d2_val < 0 else "Inconclusive (d2=0)")
                critical_points.append({"x": x_crit, "y": y_val, "d1": d1_val, "d2": d2_val, "type": pt_type})
                
    # Inflection Points: d2 crosses 0 with sign change
    inflection_points = []
    for i in range(1, len(xs) - 1):
        if not np.isnan(d2s[i-1]) and not np.isnan(d2s[i+1]):
            if d2s[i-1] * d2s[i+1] < 0:
                x_inf = xs[i]
                inflection_points.append({"x": x_inf, "y": func(x_inf), "d2": d2s[i]})
                
    # Absolute Extrema on Closed Interval
    valid_mask = ~np.isnan(ys) & ~np.isinf(ys)
    if np.any(valid_mask):
        abs_min_idx = np.argmin(ys[valid_mask])
        abs_max_idx = np.argmax(ys[valid_mask])
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask]
        abs_extrema = {
            "min": {"x": valid_xs[abs_min_idx], "y": valid_ys[abs_min_idx]},
            "max": {"x": valid_xs[abs_max_idx], "y": valid_ys[abs_max_idx]}
        }
    else:
        abs_extrema = {"min": None, "max": None}
        
    return {
        "xs": xs, "ys": ys, "d1s": d1s, "d2s": d2s,
        "critical_points": critical_points,
        "inflection_points": inflection_points,
        "abs_extrema": abs_extrema
    }

# ------------------------------------------------------------------------------
# EXPORT GENERATION UTILITIES
# ------------------------------------------------------------------------------
def generate_excel_bytes(summary_df, working_df, calculated_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Summary & Metadata', index=False)
        working_df.to_excel(writer, sheet_name='Working Data', index=False)
        calculated_df.to_excel(writer, sheet_name='Calculated Results', index=False)
    return output.getvalue()

def generate_pdf_bytes(metadata, fit_res, working_df, calculated_df, fig_buf):
    if not HAS_REPORTLAB:
        return b"ReportLab is not installed."
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph(f"<b>Hispanic Heritage Math Explorer Report</b>", styles['Title']))
    story.append(Paragraph(f"<b>Country:</b> {metadata['country']} | <b>Indicator:</b> {metadata['indicator']}", styles['Heading2']))
    story.append(Paragraph(f"<b>Date Generated:</b> {metadata['date']} | <b>Data Status:</b> {metadata['status']}", styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Model Fit Section
    if fit_res:
        story.append(Paragraph("<b>Model Fit & Statistics</b>", styles['Heading3']))
        fit_info = [
            [Paragraph(f"<b>Equation:</b> {fit_res['eq_str']}", styles['Normal'])],
            [Paragraph(f"<b>Degree:</b> {fit_res['degree']} | <b>Base Year:</b> {fit_res['base_year']} | <b>Observations:</b> {fit_res['obs_count']}", styles['Normal'])],
            [Paragraph(f"<b>R² (In-Sample):</b> {fit_res['r2']:.4f} | <b>MAE (In-Sample):</b> {fit_res['mae']:.4f}", styles['Normal'])]
        ]
        t_fit = Table(fit_info, colWidths=[500])
        t_fit.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke), ('BOX', (0,0), (-1,-1), 1, colors.grey)]))
        story.append(t_fit)
        story.append(Spacer(1, 10))
        
    # Chart Image
    if fig_buf:
        story.append(RLImage(fig_buf, width=500, height=250))
        story.append(Spacer(1, 10))
        
    # Data Preview Table
    story.append(Paragraph("<b>Working Data Sample (First 5 rows)</b>", styles['Heading3']))
    preview_data = [working_df.columns.tolist()] + working_df.head(5).values.tolist()
    t_preview = Table(preview_data, colWidths=[100, 150])
    t_preview.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(t_preview)
    story.append(Spacer(1, 10))
    
    # Calculated Predictions Table
    if not calculated_df.empty:
        story.append(Paragraph("<b>Calculated Predictions Sample</b>", styles['Heading3']))
        calc_data = [calculated_df.columns.tolist()] + calculated_df.head(10).values.tolist()
        t_calc = Table(calc_data, colWidths=[80, 100, 120, 150])
        t_calc.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightblue), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
        story.append(t_calc)
        
    doc.build(story)
    return buffer.getvalue()

# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if 'data_modified' not in st.session_state:
    st.session_state.data_modified = False
if 'user_df' not in st.session_state:
    st.session_state.user_df = None
if 'active_country' not in st.session_state:
    st.session_state.active_country = "Mexico"
if 'active_indicator' not in st.session_state:
    st.session_state.active_indicator = "Life Expectancy"
if 'anim_frame' not in st.session_state:
    st.session_state.anim_frame = 0

# ------------------------------------------------------------------------------
# UI SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.title("🛠️ Navigation & Controls")

course_mode = st.sidebar.selectbox("🎯 Course / Target Level", ["Honors Algebra 2", "Precalculus", "Calculus Mode"])
analysis_tab_sel = st.sidebar.radio("📌 Mode", ["Data Explorer & Curve Fitting", "Interactive Function & Calculus Lab"])

st.sidebar.markdown("---")
st.sidebar.subheader("1. Data Retrieval Controls")
selected_country = st.sidebar.selectbox("Primary Country", list(COUNTRIES.keys()), index=0)
selected_indicator = st.sidebar.selectbox("Indicator", list(INDICATORS.keys()), index=0)
comp_country = st.sidebar.selectbox("Comparison Country (Optional)", ["None"] + list(COUNTRIES.keys()), index=1)

# Reset Session Data on selection change
if selected_country != st.session_state.active_country or selected_indicator != st.session_state.active_indicator:
    st.session_state.active_country = selected_country
    st.session_state.active_indicator = selected_indicator
    st.session_state.data_modified = False
    st.session_state.user_df = None

st.sidebar.markdown("---")
st.sidebar.subheader("2. Polynomial Fit Settings")
poly_degree = st.sidebar.slider("Polynomial Degree (d)", min_value=3, max_value=8, value=3)
sample_step = st.sidebar.slider("Graph Sample Increment (Years)", min_value=1, max_value=10, value=1)
forward_horizon = st.sidebar.slider("Forward Horizon (Years)", min_value=0, max_value=50, value=10)

st.sidebar.markdown("---")
printer_mode = st.sidebar.checkbox("📄 Printer-Friendly Mode", value=False)

# ------------------------------------------------------------------------------
# MAIN CONTENT AREA
# ------------------------------------------------------------------------------
st.title("🏛️ Hispanic Heritage Data Explorer & Math Lab 2.0")
st.markdown("*Authentic World Bank Historical Data Analysis, Polynomial Curve Fitting, and Calculus Function Lab*")

if printer_mode:
    st.info("📄 **Printer-Friendly Mode Active**: Simplified high-contrast layout optimized for document printing and export.")

# Fetch Data
raw_df, ret_date, err_msg = fetch_world_bank_data(
    COUNTRIES[selected_country],
    INDICATORS[selected_indicator]["code"],
    START_YEAR,
    END_YEAR
)

if err_msg:
    st.error(f"⚠️ API Error / Warning: {err_msg}")

# Initialize working data
if st.session_state.user_df is None or not st.session_state.data_modified:
    st.session_state.user_df = raw_df.copy()

working_df = st.session_state.user_df

# TAB 1: DATA EXPLORER & CURVE FITTING
if analysis_tab_sel == "Data Explorer & Curve Fitting":
    st.header(f"📊 {selected_country} - {INDICATORS[selected_indicator]['name']}")
    
    # Metadata callout box
    with st.expander("ℹ️ Indicator Metadata, Units & API Details", expanded=False):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write(f"**Indicator Code:** `{INDICATORS[selected_indicator]['code']}`")
            st.write(f"**Units:** {INDICATORS[selected_indicator]['units']}")
            st.write(f"**Retrieval Date:** {ret_date}")
        with col_m2:
            st.write(f"**Requested Range:** {START_YEAR} to {END_YEAR} (70 years)")
            st.write(f"**Actual Observations Returned:** {len(raw_df)}")
            st.write(f"**Source:** [World Bank Open Data API](https://api.worldbank.org/v2/country/{COUNTRIES[selected_country]}/indicator/{INDICATORS[selected_indicator]['code']}?date={START_YEAR}:{END_YEAR})")
        st.caption(f"**Definition:** {INDICATORS[selected_indicator]['desc']}")

    if len(working_df) < poly_degree + 1:
        st.warning(f"Insufficient data points ({len(working_df)}) to fit a degree-{poly_degree} polynomial. Minimum required is {poly_degree + 1}.")
    else:
        # Fit model
        fit_res = fit_polynomial(working_df, poly_degree)
        holdout_res = fit_holdout_model(working_df, poly_degree)

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Observed Points", f"{fit_res['obs_count']}")
        m_col2.metric("In-Sample R²", f"{fit_res['r2']:.4f}")
        m_col3.metric("In-Sample MAE", f"{fit_res['mae']:.4f} {INDICATORS[selected_indicator]['units']}")
        if holdout_res:
            m_col4.metric("5-Yr Holdout MAE", f"{holdout_res['test_mae']:.4f}")
        else:
            m_col4.metric("5-Yr Holdout MAE", "N/A")

        # Visual Plotting
        st.subheader("📈 Historical Data & Polynomial Extrapolation Fit")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        if printer_mode:
            plt.style.use('default')
        
        # Subsampled markers for sample increment display
        subsampled_df = working_df.iloc[::sample_step]
        ax.scatter(subsampled_df['year'], subsampled_df['value'], color='blue', label=f"Observed ({sample_step}-yr sample step)", zorder=3)
        
        # Plot full smooth fitted curve
        min_yr = working_df['year'].min()
        max_yr = working_df['year'].max()
        extrap_max = max_yr + forward_horizon
        
        t_hist = np.linspace(min_yr, max_yr, 200) - fit_res['base_year']
        ax.plot(t_hist + fit_res['base_year'], fit_res['poly_func'](t_hist), 'g-', label=f"Degree {poly_degree} Fit Curve", linewidth=2)
        
        if forward_horizon > 0:
            t_extrap = np.linspace(max_yr, extrap_max, 100) - fit_res['base_year']
            ax.plot(t_extrap + fit_res['base_year'], fit_res['poly_func'](t_extrap), 'r--', label="Extrapolated Horizon", linewidth=2)
            ax.axvline(max_yr, color='grey', linestyle=':', label='Historical Boundary')

        # Optional Comparison Country Plot
        if comp_country != "None" and comp_country != selected_country:
            comp_df, _, _ = fetch_world_bank_data(COUNTRIES[comp_country], INDICATORS[selected_indicator]["code"], START_YEAR, END_YEAR)
            if not comp_df.empty:
                ax.plot(comp_df['year'], comp_df['value'], color='orange', linestyle='-.', label=f"Comparison: {comp_country}")

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{selected_indicator} ({INDICATORS[selected_indicator]['units']})")
        ax.set_title(f"{selected_country}: {selected_indicator} ({min_yr}-{max_yr})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

        # Buffer image for PDF export
        fig_buf = io.BytesIO()
        fig.savefig(fig_buf, format='png', bbox_inches='tight')
        fig_buf.seek(0)

        # Equation & Pedagogy Card
        st.subheader("🧮 Fitted Mathematical Equation & Pedagogy")
        st.code(f"{fit_res['eq_str']}\nwhere t = year - {fit_res['base_year']}", language="text")

        # Course Level Specific Analysis
        if course_mode == "Honors Algebra 2":
            st.markdown("### 📘 Honors Algebra 2 Analysis")
            st.write(f"- **Domain (Historical):** [{fit_res['hist_min']}, {fit_res['hist_max']}]")
            st.write(f"- **End Behavior:** As $t \\to \\infty$, $f(t) \\to {'+\\infty' if fit_res['coeffs'][0] > 0 else '-\\infty'}$.")
            arc = (fit_res['poly_func'](fit_res['hist_max'] - fit_res['base_year']) - fit_res['poly_func'](0)) / (fit_res['hist_max'] - fit_res['hist_min'])
            st.write(f"- **Overall Average Rate of Change:** {arc:.4f} {INDICATORS[selected_indicator]['units']} per year.")

        elif course_mode == "Precalculus":
            st.markdown("### 📙 Precalculus Analysis")
            st.write(f"- **Degree & Coefficients:** Degree {poly_degree} polynomial with leading coefficient $a_{poly_degree} = {fit_res['coeffs'][0]:.4e}$.")
            d1_poly = np.polyder(fit_res['poly_func'])
            roots = np.roots(d1_poly)
            real_roots = roots[np.isreal(roots)].real
            turning_pts = [r + fit_res['base_year'] for r in real_roots if fit_res['hist_min'] <= (r + fit_res['base_year']) <= fit_res['hist_max']]
            st.write(f"- **Turning Points in Historical Domain:** {len(turning_pts)} identified at years {np.round(turning_pts, 1)}.")

        else: # Calculus
            st.markdown("### 📕 Calculus Function Analysis")
            d1_poly = np.polyder(fit_res['poly_func'])
            d2_poly = np.polyder(d1_poly)
            st.write(f"- **First Derivative $f'(t)$:** `{d1_poly}`")
            st.write(f"- **Second Derivative $f''(t)$:** `{d2_poly}`")

        # Interactive Calculation Panel
        st.markdown("---")
        st.subheader("🔢 Prediction & Batch Calculation Engine")
        
        calc_tab1, calc_tab2 = st.tabs(["Single Year Calculation", "Batch Range Calculation"])
        
        with calc_tab1:
            col_c1, col_c2 = st.columns([1, 2])
            with col_c1:
                calc_year = st.number_input("Enter Target Year (1950-2100)", min_value=1950, max_value=2100, value=CURRENT_YEAR)
                calc_btn = st.button("Calculate Value")
            with col_c2:
                if calc_btn or 'last_calc' in st.session_state:
                    st.session_state.last_calc = calc_year
                    t_val = calc_year - fit_res['base_year']
                    est_val = fit_res['poly_func'](t_val)
                    is_in_range = fit_res['hist_min'] <= calc_year <= fit_res['hist_max']
                    
                    obs_match = working_df[working_df['year'] == calc_year]
                    
                    st.write(f"**Target Year:** {calc_year}")
                    st.write(f"**Estimated Value:** `{est_val:.4f} {INDICATORS[selected_indicator]['units']}`")
                    st.write(f"**Classification:** {'Within Observed Historical Range' if is_in_range else '⚠️ Extrapolated Outside Historical Range'}")
                    if not obs_match.empty:
                        st.info(f"Actual Historical Observation available for {calc_year}: **{obs_match['value'].values[0]:.4f}**")
                        
        with calc_tab2:
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                b_start, b_end = st.slider("Batch Year Range", 1950, 2100, (2020, 2030))
                b_step = st.slider("Range Step", 1, 10, 1)
            with b_step_col if (b_step_col := b_col2) else b_col2:
                custom_yrs_str = st.text_input("Or Custom Comma-Separated Years (e.g. 2025, 2030, 2040)")
                
            calc_batch_df = pd.DataFrame()
            if custom_yrs_str.strip():
                try:
                    parsed_yrs = [int(y.strip()) for y in custom_yrs_str.split(",") if y.strip().isdigit()]
                except Exception:
                    parsed_yrs = []
            else:
                parsed_yrs = list(range(b_start, b_end + 1, b_step))
                
            if parsed_yrs:
                batch_records = []
                for y in parsed_yrs:
                    t_v = y - fit_res['base_year']
                    v_est = fit_res['poly_func'](t_v)
                    in_r = fit_res['hist_min'] <= y <= fit_res['hist_max']
                    batch_records.append({"Year": y, f"Estimated {selected_indicator}": round(v_est, 4), "Units": INDICATORS[selected_indicator]['units'], "In Historical Range": in_r})
                calc_batch_df = pd.DataFrame(batch_records)
                st.dataframe(calc_batch_df, use_container_width=True)

        # Editable Data Scenario Panel
        st.markdown("---")
        st.subheader("✏️ Validate & Edit Historical Data (Scenario Analysis)")
        st.caption("Edit values below to simulate custom scenarios. Edits immediately update polynomial fits.")
        
        col_ed1, col_ed2 = st.columns([3, 1])
        with col_ed1:
            edited_df = st.data_editor(st.session_state.user_df, num_rows="dynamic", key="data_editor_key")
            if not edited_df.equals(st.session_state.user_df):
                st.session_state.user_df = edited_df
                st.session_state.data_modified = True
                st.rerun()
        with col_ed2:
            if st.session_state.data_modified:
                st.warning("⚠️ User-Modified Scenario Active")
            if st.button("🔄 Reset to Source API Data"):
                st.session_state.user_df = raw_df.copy()
                st.session_state.data_modified = False
                st.rerun()

        # Exports Section
        st.markdown("---")
        st.subheader("💾 Export Model Results & Reports")
        
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        # CSV Export
        csv_data = working_df.to_csv(index=False).encode('utf-8')
        exp_col1.download_button("📥 Download CSV Data", data=csv_data, file_name=f"{selected_country}_{selected_indicator}.csv", mime="text/csv")
        
        # Excel Export
        summary_meta_df = pd.DataFrame([
            {"Metric": "Country", "Value": selected_country},
            {"Metric": "Indicator", "Value": selected_indicator},
            {"Metric": "Degree", "Value": poly_degree},
            {"Metric": "R2", "Value": fit_res['r2']},
            {"Metric": "MAE", "Value": fit_res['mae']},
            {"Metric": "Data Modified Status", "Value": "User Modified" if st.session_state.data_modified else "Original Source"}
        ])
        excel_bytes = generate_excel_bytes(summary_meta_df, working_df, calc_batch_df)
        exp_col2.download_button("📥 Download Excel (.xlsx)", data=excel_bytes, file_name=f"{selected_country}_{selected_indicator}_Analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # PDF Export
        if HAS_REPORTLAB:
            pdf_metadata = {
                "country": selected_country,
                "indicator": selected_indicator,
                "date": ret_date,
                "status": "User Modified" if st.session_state.data_modified else "Original Source"
            }
            pdf_bytes = generate_pdf_bytes(pdf_metadata, fit_res, working_df, calc_batch_df, fig_buf)
            exp_col3.download_button("📥 Download PDF Report", data=pdf_bytes, file_name=f"{selected_country}_Report.pdf", mime="application/pdf")
        else:
            exp_col3.info("Install ReportLab for PDF export.")

# TAB 2: INTERACTIVE FUNCTION & CALCULUS LAB
else:
    st.header("🔬 Interactive Calculus & Multi-Context Simulation Lab")
    
    ctx_col1, ctx_col2 = st.columns(2)
    with ctx_col1:
        selected_context = st.selectbox("Select Application Context", list(CONTEXT_PRESETS.keys()))
        ctx_info = CONTEXT_PRESETS[selected_context]
    with ctx_col2:
        func_source = st.radio("Function Definition Mode", ["Use Curve Fit from Data", "Choose Preset Function", "Custom Expression Input"])
        
    # Variables & Custom Context Config
    with st.expander("⚙️ Customize Variable Names, Units & Context Parameters", expanded=False):
        c_col1, c_col2, c_col3 = st.columns(3)
        var_x = c_col1.text_input("Independent Variable (x)", value=ctx_info['x_var'])
        var_y = c_col2.text_input("Dependent Variable (y)", value=ctx_info['y_var'])
        units_str = c_col3.text_input("Units", value=ctx_info['units'])

    # Determine Active Mathematical Function
    active_func = None
    func_display_str = ""
    
    if func_source == "Use Curve Fit from Data":
        if len(working_df) >= poly_degree + 1:
            fit_res = fit_polynomial(working_df, poly_degree)
            base_yr = fit_res['base_year']
            active_func = lambda x: fit_res['poly_func'](x - base_yr)
            func_display_str = f"Fit Model: {fit_res['eq_str']} (t = year - {base_yr})"
            d_min, d_max = float(working_df['year'].min()), float(working_df['year'].max())
        else:
            st.error("Insufficient historical data to fit polynomial.")
            d_min, d_max = 0.0, 10.0
            active_func = lambda x: x**2
            func_display_str = "f(x) = x^2"
            
    elif func_source == "Choose Preset Function":
        preset_choice = st.selectbox("Preset Models", [
            "Polynomial: f(x) = -0.1x^3 + 1.5x^2 - 4x + 10",
            "Trigonometric Oscillation: f(x) = 5*sin(0.8*x) + 12",
            "Rational Response: f(x) = (10*x) / (x^2 + 4)",
            "Exponential Concentration: f(x) = 25*x*exp(-0.5*x)"
        ])
        if "Polynomial" in preset_choice:
            active_func = lambda x: -0.1*x**3 + 1.5*x**2 - 4*x + 10
            func_display_str = "f(x) = -0.1x^3 + 1.5x^2 - 4x + 10"
        elif "Trigonometric" in preset_choice:
            active_func = lambda x: 5*np.sin(0.8*x) + 12
            func_display_str = "f(x) = 5\\sin(0.8x) + 12"
        elif "Rational" in preset_choice:
            active_func = lambda x: (10*x) / (x**2 + 4)
            func_display_str = "f(x) = \\frac{10x}{x^2 + 4}"
        else:
            active_func = lambda x: 25*x*np.exp(-0.5*x)
            func_display_str = "f(x) = 25x e^{-0.5x}"
        d_min, d_max = 0.0, 10.0

    else: # Custom Expression Input
        custom_input = st.text_input("Enter Function f(x)", value="x^3 - 6*x^2 + 9*x + 2")
        d_min = st.number_input("Domain Minimum", value=0.0)
        d_max = st.number_input("Domain Maximum", value=6.0)
        active_func = lambda x: evaluate_custom_function(custom_input, x)
        func_display_str = f"f(x) = {custom_input}"

    # Perform Calculus Feature Analysis
    calc_res = analyze_calculus_features(active_func, d_min, d_max)

    # Display Functions in LaTeX
    st.markdown("### 📐 Mathematical Formulation")
    st.latex(func_display_str)

    # Interactive Inspection Controls
    st.markdown("---")
    st.subheader("🎛️ Interactive Secant & Tangent Analysis Controls")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        x_pt = st.slider("Tangency Point (x)", float(d_min), float(d_max), float((d_min + d_max)/2), step=0.1)
    with ctrl_col2:
        sec_a = st.slider("Secant Interval Start (a)", float(d_min), float(d_max), float(d_min), step=0.1)
    with ctrl_col3:
        sec_b = st.slider("Secant Interval End (b)", float(d_min), float(d_max), float(d_max), step=0.1)

    # Dynamic Rates of Change Calculations
    y_pt = active_func(x_pt)
    d1_pt, d2_pt = numerical_derivatives(active_func, x_pt)
    
    y_a = active_func(sec_a)
    y_b = active_func(sec_b)
    aroc = (y_b - y_a) / (sec_b - sec_a) if sec_b != sec_a else np.nan

    # Dynamic Contextual Interpretation Cards
    st.markdown("---")
    st.subheader("💡 Pedagogical Interpretation & Contextual Meaning")
    
    interp_col1, interp_col2, interp_col3 = st.columns(3)
    with interp_col1:
        st.metric(f"Function Value f({x_pt:.1f})", f"{y_pt:.3f} {units_str}")
        st.caption(f"**Meaning:** Current state of {var_y} at {var_x} = {x_pt:.1f}.")
        
    with interp_col2:
        st.metric(f"Instantaneous Rate f'({x_pt:.1f})", f"{d1_pt:.3f} {units_str}/{var_x}")
        direction = "increasing" if d1_pt > 0 else ("decreasing" if d1_pt < 0 else "stationary")
        st.caption(f"**Meaning:** Rate of change of {var_y}. The system is **{direction}**.")
        
    with interp_col3:
        st.metric(f"Concavity f''({x_pt:.1f})", f"{d2_pt:.3f}")
        conc = "concave upwards (accelerating)" if d2_pt > 0 else ("concave downwards (decelerating)" if d2_pt < 0 else "linear")
        st.caption(f"**Meaning:** Curvature. The growth rate is **{conc}**.")

    # Main Interactive Visual Graph
    st.subheader("📈 Multi-Curve Interactive Plot")
    
    show_f = st.checkbox("Show f(x)", value=True)
    show_d1 = st.checkbox("Show First Derivative f'(x)", value=True)
    show_d2 = st.checkbox("Show Second Derivative f''(x)", value=False)
    show_tangent = st.checkbox("Show Tangent Line at x", value=True)
    show_secant = st.checkbox("Show Secant Line [a, b]", value=True)

    fig_lab, ax_lab = plt.subplots(figsize=(10, 5))
    
    if show_f:
        ax_lab.plot(calc_res['xs'], calc_res['ys'], 'b-', label='f(x)', linewidth=2)
    if show_d1:
        ax_lab.plot(calc_res['xs'], calc_res['d1s'], 'g--', label="f'(x)", linewidth=1.5)
    if show_d2:
        ax_lab.plot(calc_res['xs'], calc_res['d2s'], 'r:', label="f''(x)", linewidth=1.5)
        
    # Tangent line plot
    if show_tangent and not np.isnan(d1_pt):
        t_line_x = np.array([x_pt - 1.5, x_pt + 1.5])
        t_line_y = y_pt + d1_pt * (t_line_x - x_pt)
        ax_lab.plot(t_line_x, t_line_y, 'm-', label=f"Tangent at x={x_pt:.1f}", linewidth=2)
        ax_lab.plot(x_pt, y_pt, 'mo', markersize=8)

    # Secant line plot
    if show_secant and not np.isnan(aroc):
        ax_lab.plot([sec_a, sec_b], [y_a, y_b], 'c--o', label=f"Secant [a={sec_a:.1f}, b={sec_b:.1f}]", linewidth=2)

    # Highlight Critical & Inflection Points
    for cp in calc_res['critical_points']:
        ax_lab.plot(cp['x'], cp['y'], 'ro', markersize=6)
        ax_lab.annotate(f"Crit ({cp['type']})", (cp['x'], cp['y']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    ax_lab.set_xlabel(var_x)
    ax_lab.set_ylabel(var_y)
    ax_lab.set_title(f"Calculus Analysis: {selected_context}")
    ax_lab.legend()
    ax_lab.grid(True, alpha=0.3)
    
    st.pyplot(fig_lab)

    # Graphical Visual Simulation Box
    st.markdown("---")
    st.subheader("🤖 Visual Simulation Model")
    vis_type = ctx_info['visual']
    
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.write(f"**Simulation Mode:** {selected_context}")
        st.write(f"**Current State Value:** `{y_pt:.2f} {units_str}`")
    with sim_col2:
        # Render dynamic visual representation
        fig_sim, ax_sim = plt.subplots(figsize=(6, 2))
        ax_sim.set_xlim(0, 10)
        ax_sim.set_ylim(0, 5)
        ax_sim.axis('off')
        
        # Scale response to graphic box
        norm_val = np.clip((y_pt - np.nanmin(calc_res['ys'])) / (np.nanmax(calc_res['ys']) - np.nanmin(calc_res['ys']) + 1e-6), 0.1, 0.9) if len(calc_res['ys']) > 0 else 0.5

        if vis_type == "prosthetic":
            # Walking leg rotation angle
            angle = norm_val * np.pi / 2
            leg_x = [5, 5 + 3 * np.sin(angle)]
            leg_y = [4, 4 - 3 * np.cos(angle)]
            ax_sim.plot(leg_x, leg_y, 'ro-', linewidth=4, label="Prosthetic Joint Flexion")
            ax_sim.text(5, 4.5, f"Joint Flexion: {y_pt:.1f}°", ha='center')
        elif vis_type == "drug" or vis_type == "population":
            # Tank fluid level / population density
            ax_sim.bar([5], [norm_val * 4], width=2, color='teal', alpha=0.7)
            ax_sim.text(5, 4.5, f"Concentration / Density: {y_pt:.2f}", ha='center')
        elif vis_type == "projectile":
            # Moving height ball
            ax_sim.plot([5], [norm_val * 4], 'go', markersize=16)
            ax_sim.text(5, 4.5, f"Height / Load: {y_pt:.2f}", ha='center')
        else: # Business / Weather
            # Gauge level
            ax_sim.plot([1, 9], [2, 2], 'k-', linewidth=3)
            ax_sim.plot([1 + norm_val * 8], [2], 'bs', markersize=14)
            ax_sim.text(5, 3.5, f"Indicator Level: {y_pt:.2f}", ha='center')
            
        st.pyplot(fig_sim)

# ------------------------------------------------------------------------------
# FOOTER & DEFENSE DISCLOSURES
# ------------------------------------------------------------------------------
st.markdown("---")
with st.expander("🎓 Defensive Disclosures, Assumptions & System Architecture"):
    st.markdown("""
    ### System Architecture & Mathematical Assumptions
    1. **Centered Time Base ($t = \\text{year} - \\text{base\_year}$):**
       Prevents numerical instability and ill-conditioned Vandermonde matrices when fitting higher-degree ($d=3\\dots8$) polynomials to calendar years.
    2. **In-Sample Metrics vs. Holdout Validation:**
       $R^2$ and MAE computed over the entire dataset reflect in-sample fitting accuracy. The 5-year chronological holdout model explicitly tests out-of-sample generalization.
    3. **Numerical Derivative Accuracy:**
       Derivatives are calculated using central finite differences ($h = 10^{-5}$), balancing numerical precision against floating-point roundoff error.
    4. **World Bank API Integrity:**
       No arbitrary data fabrication occurs. If indicator data contains gaps or API failure occurs, warnings are raised without substituting fake values.
    """)