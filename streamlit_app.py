# app.py (v2.1 – side-view fixed by clamping min_side to height for compute only)
"""
Sensor coverage: cone ∩ floor (top/side views), with non-tilted vs tilted sensors.
Enhancement of v1: sidebar controls, URL state, extra exports, and UX polish.
(Patched: avoid st.rerun in callbacks, no state-after-widget warnings,
and side-view stays correct after changes.)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------- Geometry -------------
def ellipse_from_cone(h, fov_deg, tilt_deg, yaw_deg=0.0):
    a = np.array([0.0, 0.0, h])  # apex
    alpha = np.deg2rad(fov_deg / 2.0)

    theta = np.deg2rad(tilt_deg)
    psi   = np.deg2rad(yaw_deg)
    d = np.array([np.sin(theta)*np.cos(psi),
                  np.sin(theta)*np.sin(psi),
                 -np.cos(theta)], dtype=float)
    d /= np.linalg.norm(d)

    M3 = np.outer(d, d) - (np.cos(alpha)**2) * np.eye(3)
    M  = M3[:2, :2]
    q  = -2.0 * (a @ M3)[:2]
    k  =  (a @ M3 @ a)

    xc = -0.5 * np.linalg.solve(M, q)
    kprime = k + q @ xc + xc @ M @ xc
    evals, evecs = np.linalg.eigh(M)
    axes = np.sqrt(-kprime / evals)
    order = np.argsort(axes)[::-1]
    axes  = axes[order]
    evecs = evecs[:, order]
    angle_deg = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return xc, (axes[0], axes[1]), angle_deg


def compute_sensor(sensorX, sensorY, height, min_side, fov, tilt, yaw, name):
    # --- robust min_side (avoid 0 or >= height) ---
    print(f"height {float(height)}, min_side{float(min_side)}")
    ms = float(min_side)
    eps = 1e-6
    ms = max(eps, min(ms, float(height) - eps))  # clamp into (0, height)
    print(f"height {float(height)}, min_side{float(min_side)}")
    alpha = fov / 2.0
    h = float(height) - ms
    if (alpha + tilt) >= 89.9:
        raise ValueError("Cone nearly parallel to floor: reduce tilt or opening angle.")

    # Side-view distances along tilt axis
    leftFloor  = h * np.tan(np.deg2rad(alpha - tilt))
    rightFloor = h * np.tan(np.deg2rad(alpha + tilt))

    # Exact ellipse via quadric ∩ plane
    (cx, cy), (a, b), angle_deg = ellipse_from_cone(h=h, fov_deg=fov, tilt_deg=tilt, yaw_deg=yaw)
    cx_w = sensorX + cx
    cy_w = sensorY + cy

    r0 = np.hypot(sensorX, sensorY)
    ellipse_start = r0 - leftFloor
    ellipse_end   = ellipse_start + (leftFloor + rightFloor)
    center_view   = height * np.tan(np.deg2rad(tilt))

    return dict(
        name=name,
        sensorX=sensorX, sensorY=sensorY,
        height=height, min_side=ms,                 # <= clamped value used for plotting
        fov=fov, tilt=tilt, yaw=yaw,
        alpha=alpha, h=h,
        cx=cx_w, cy=cy_w,
        a=a, b=b, angle_deg=angle_deg,
        ellipse_width=2*a, ellipse_height=2*b,
        area=np.pi * a * b,
        leftFloor=leftFloor, rightFloor=rightFloor,
        ellipse_start=ellipse_start, ellipse_end=ellipse_end,
        center_view=center_view,
        ellipse_origindistance=np.hypot(cx, cy),
    )



def ellipse_poly(cx, cy, a, b, angle_deg, n=200):
    t = np.linspace(0, 2*np.pi, n)
    ca, sa = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    R = np.array([[ca, -sa], [sa, ca]])
    pts = R @ np.vstack([a*np.cos(t), b*np.sin(t)])
    return cx + pts[0], cy + pts[1]


# ------------- App config -------------
st.set_page_config(page_title="QUMEA Sensor Coverage — Floor Coverage", layout="wide")
st.title("QUMEA Sensor Coverage — Floor Coverage")

DEFAULTS = dict(
    height=2.7,              # sensorHeight
    fov=116.0,               # sensorOpeningAngle
    tilt=10.0,               # sensorTiltAngle
    yaw=0.0,                 # sensorRoomRotation
    sensorX=0.0,
    sensorY=0.0,
    min_side=1.3,            # minimumSideHeight
    npts=200,
    autoscale=True,
)

# ---------- URL <-> state helpers ----------
def _qp_get(name: str, fallback):
    val = st.query_params.get(name, None)
    if isinstance(val, list):
        val = val[0] if val else None
    if val is None:
        return fallback
    if name == "autoscale":
        return str(val).lower() == "true"
    if name == "npts":
        try: return int(val)
        except Exception: return fallback
    try: return float(val)
    except Exception: return fallback

def _qp_set_from_state():
    st.query_params.clear()
    st.query_params.update({
        "height":   f"{st.session_state['height']}",
        "fov":      f"{st.session_state['fov']}",
        "tilt":     f"{st.session_state['tilt']}",
        "yaw":      f"{st.session_state['yaw']}",
        "sensorX":  f"{st.session_state['sensorX']}",
        "sensorY":  f"{st.session_state['sensorY']}",
        "min_side": f"{st.session_state['min_side']}",
        "npts":     f"{int(st.session_state['npts'])}",
        "autoscale": "true" if st.session_state["autoscale"] else "false",
    })

def _reset_state_and_url():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.query_params.clear()
    st.query_params.update({
        "height":   f"{DEFAULTS['height']}",
        "fov":      f"{DEFAULTS['fov']}",
        "tilt":     f"{DEFAULTS['tilt']}",
        "yaw":      f"{DEFAULTS['yaw']}",
        "sensorX":  f"{DEFAULTS['sensorX']}",
        "sensorY":  f"{DEFAULTS['sensorY']}",
        "min_side": f"{DEFAULTS['min_side']}",
        "npts":     f"{int(DEFAULTS['npts'])}",
        "autoscale": "true" if DEFAULTS["autoscale"] else "false",
    })

# Initialize session_state from URL (only once)
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.height    = _qp_get("height",   DEFAULTS["height"])
    st.session_state.fov       = _qp_get("fov",      DEFAULTS["fov"])
    st.session_state.tilt      = _qp_get("tilt",     DEFAULTS["tilt"])
    st.session_state.yaw       = _qp_get("yaw",      DEFAULTS["yaw"])
    st.session_state.sensorX   = _qp_get("sensorX",  DEFAULTS["sensorX"])
    st.session_state.sensorY   = _qp_get("sensorY",  DEFAULTS["sensorY"])
    st.session_state.min_side  = _qp_get("min_side", DEFAULTS["min_side"])
    st.session_state.npts      = _qp_get("npts",     DEFAULTS["npts"])
    st.session_state.autoscale = _qp_get("autoscale", DEFAULTS["autoscale"])

# ------------- Sidebar -------------
with st.sidebar:
    st.header("Controls")

    # Actions BEFORE sliders
    a1, a2 = st.columns(2)
    a1.button("Reset to defaults", on_click=_reset_state_and_url)
    a2.button("Copy URL with current state", on_click=_qp_set_from_state)

    height = st.slider("Height [m]", 1.3, 4.3, step=0.01, key="height")
    fov    = st.slider("Opening Angle [°]", 10.0, 130.0, step=0.5, key="fov")
    tilt   = st.slider("Tilt [°]", 0.1, 30.0, step=0.1, key="tilt")
    yaw    = st.slider("Rotation (Yaw) [°]", 0.0, 90.0, step=0.1, key="yaw")

    with st.expander("Advanced", expanded=False):
        sensorX  = st.number_input("Sensor X [m]", step=0.1, format="%.3f", key="sensorX")
        sensorY  = st.number_input("Sensor Y [m]", step=0.1, format="%.3f", key="sensorY")
        min_side = st.number_input(
            "Minimum side height [m]",
            min_value=0.1,
            max_value=10.0, 
            step=0.05,
            format="%.2f",
            key="min_side",
        )
        npts = st.slider("Ellipse smoothness (points)", 64, 720, step=32, key="npts")

    st.subheader("View")
    autoscale = st.checkbox("Autoscale plots", key="autoscale")
    rescale_click = st.button("Rescale now")

# ------------- Read state (post-widgets) -------------
height    = float(st.session_state["height"])
fov       = float(st.session_state["fov"])
tilt      = float(st.session_state["tilt"])
yaw       = float(st.session_state["yaw"])
sensorX   = float(st.session_state["sensorX"])
sensorY   = float(st.session_state["sensorY"])
min_side  = float(st.session_state["min_side"])
npts      = int(st.session_state["npts"])
autoscale = bool(st.session_state["autoscale"])

# ------------- Compute sensors (with safe min_side) -------------
try:
    base = compute_sensor(sensorX, sensorY, height, min_side, fov, tilt=0.0, yaw=yaw, name="non-tilted")
    til  = compute_sensor(sensorX, sensorY, height, min_side, fov, tilt=tilt,  yaw=yaw, name="tilted")
except ValueError as e:
    st.error(str(e))
    st.stop()

# ------------- KPIs -------------
right_gain = til["ellipse_end"]   - base["ellipse_end"]
left_delta = til["ellipse_start"] - base["ellipse_start"]
st.markdown(
    f"**Tilt:** {tilt:.1f}°  |  **Right coverage**: +{right_gain:.2f} m  |  "
    f"**Left coverage**: {left_delta:.2f} m"
)

# ------------- Tables -------------
def table_rows(sensor):
    return {
        "Name": sensor["name"],
        "Center X [m]": f"{sensor['cx']:.3f}",
        "Center Y [m]": f"{sensor['cy']:.3f}",
        "Center distance [m]": f"{sensor['ellipse_origindistance']:.3f}",
        "Major diameter (2a) [m]": f"{2*sensor['a']:.3f}",
        "Minor diameter (2b) [m]": f"{2*sensor['b']:.3f}",
        "Area (πab) [m²]": f"{sensor['area']:.3f}",
        "Major-axis angle [°]": f"{sensor['angle_deg']:.2f}",
    }

def table_rows_side(sensor):
    return {
        "Name": sensor["name"],
        "Left floor [m]": f"{sensor['leftFloor']:.3f}",
        "Right floor [m]": f"{sensor['rightFloor']:.3f}",
        "Ellipse start [m]": f"{sensor['ellipse_start']:.3f}",
        "Ellipse end [m]": f"{sensor['ellipse_end']:.3f}",
        "Axis hit (center_view) [m]": f"{sensor['center_view']:.3f}",
        "Effective min side [m]": f"{sensor['min_side']:.3f}",
    }

st.subheader("Ellipse dimensions")
df_dim = pd.DataFrame([table_rows(base), table_rows(til)]).set_index("Name")
st.dataframe(df_dim, use_container_width=True)

with st.expander("Side-view metrics", expanded=False):
    df_side = pd.DataFrame([table_rows_side(base), table_rows_side(til)]).set_index("Name")
    st.dataframe(df_side, use_container_width=True)

# ------------- Downloads -------------
col_dl1, col_dl2, _ = st.columns([1, 1, 3])
col_dl1.download_button(
    "Download dimensions (CSV)",
    df_dim.to_csv(index=True).encode(),
    file_name="ellipse_dimensions.csv",
    mime="text/csv",
)

with st.expander("Download footprint polygons (CSV)", expanded=False):
    x0, y0 = ellipse_poly(base["cx"], base["cy"], base["a"], base["b"], base["angle_deg"], n=npts)
    x1, y1 = ellipse_poly(til["cx"],  til["cy"],  til["a"],  til["b"],  til["angle_deg"],  n=npts)
    df_poly = pd.DataFrame({
        "name": ["non-tilted"]*len(x0) + ["tilted"]*len(x1),
        "x":    np.concatenate([x0, x1]),
        "y":    np.concatenate([y0, y1]),
        "seq":  np.concatenate([np.arange(len(x0)), np.arange(len(x1))]),
    })
    st.download_button(
        "Download polygons CSV",
        df_poly.to_csv(index=False).encode(),
        file_name="footprints.csv",
        mime="text/csv",
    )

export = {
    "inputs": dict(height=height, fov=fov, tilt=tilt, yaw=yaw,
                   sensorX=sensorX, sensorY=sensorY, min_side=min_side, npts=npts),
    "non_tilted": base,
    "tilted": til,
}
col_dl2.download_button(
    "Download JSON (inputs + outputs)",
    data=json.dumps(export, indent=2).encode(),
    file_name="sensor_coverage.json",
    mime="application/json",
)

# ------------- Plots -------------
def side_view_figure():
    fig = go.Figure()
    colors = {"non-tilted": "#3b82f6", "tilted": "#22c55e"}

    for s in (base, til):
        col = colors[s["name"]]
        # slanted edges
        fig.add_trace(go.Scatter(
            x=[0, s["ellipse_start"]], y=[s["height"], s["min_side"]],
            mode="lines", line=dict(dash="dash", width=2, color=col),
            name=f"{s['name']} left edge"
        ))
        fig.add_trace(go.Scatter(
            x=[0, s["ellipse_end"]], y=[s["height"], s["min_side"]],
            mode="lines", line=dict(dash="dash", width=2, color=col),
            name=f"{s['name']} right edge"
        ))
        # vertical drops
        fig.add_trace(go.Scatter(
            x=[s["ellipse_start"], s["ellipse_start"]], y=[s["min_side"], 0],
            mode="lines", line=dict(dash="dot", width=1.5, color=col),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[s["ellipse_end"], s["ellipse_end"]], y=[s["min_side"], 0],
            mode="lines", line=dict(dash="dot", width=1.5, color=col),
            showlegend=False
        ))
        # axis line
        fig.add_trace(go.Scatter(
            x=[0, s["center_view"]], y=[s["height"], 0],
            mode="lines", line=dict(width=2, color=col),
            name=f"{s['name']} axis"
        ))

    fig.update_xaxes(title_text="Sensor distance [m]")
    fig.update_yaxes(title_text="Z [m]")
    fig.update_layout(
        title="Side view",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # ✅ New inclusive autoscale: keep both dotted lines in view
    if autoscale or rescale_click:
        min_x = min(base["ellipse_start"], til["ellipse_start"])
        max_x = max(base["ellipse_end"],   til["ellipse_end"])
        span  = max(1e-6, max_x - min_x)
        pad_x = max(0.1, 0.05 * span)
        fig.update_xaxes(range=[min_x - pad_x, max_x + pad_x])

        top_y = max(base["height"], til["height"]) * 1.1
        fig.update_yaxes(range=[-0.2, top_y])

    return fig


def top_view_figure():
    fig = go.Figure()
    colors = {"non-tilted": "#3b82f6", "tilted": "#22c55e"}

    x0, y0 = ellipse_poly(base["cx"], base["cy"], base["a"], base["b"], base["angle_deg"], n=npts)
    fig.add_trace(go.Scatter(
        x=x0, y=y0, mode="lines", fill="toself",
        name="non-tilted footprint",
        line=dict(width=2, color=colors["non-tilted"]),
        fillcolor="rgba(59,130,246,0.15)"
    ))

    x1, y1 = ellipse_poly(til["cx"], til["cy"], til["a"], til["b"], til["angle_deg"], n=npts)
    fig.add_trace(go.Scatter(
        x=x1, y=y1, mode="lines", fill="toself",
        name="tilted footprint",
        line=dict(width=2, color=colors["tilted"]),
        fillcolor="rgba(34,197,94,0.15)"
    ))

    t_rot = np.deg2rad(til["yaw"])
    xs = sensorX + til["ellipse_start"] * np.cos(t_rot)
    ys = sensorY + til["ellipse_start"] * np.sin(t_rot)
    xe = sensorX + til["ellipse_end"]   * np.cos(t_rot)
    ye = sensorY + til["ellipse_end"]   * np.sin(t_rot)
    fig.add_trace(go.Scatter(
        x=[xs, xe], y=[ys, ye], mode="lines",
        line=dict(width=2, color="#10b981"),
        name="tilt axis span"
    ))

    fig.update_xaxes(title_text="X [m]")
    fig.update_yaxes(title_text="Y [m]", scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Top view (ellipse footprints)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )

    if autoscale or rescale_click:
        max_r = max(base["ellipse_width"], til["ellipse_width"]) * 0.55
        cx = (base["cx"] + til["cx"]) / 2.0
        cy = (base["cy"] + til["cy"]) / 2.0
        pad = 0.15 * max_r
        fig.update_xaxes(range=[cx - max_r - pad, cx + max_r + pad])
        fig.update_yaxes(range=[cy - max_r - pad, cy + max_r + pad])
    return fig

# --- Layout ---
cL, cR = st.columns(2)
cL.plotly_chart(side_view_figure(), use_container_width=True, theme=None)
cR.plotly_chart(top_view_figure(),  use_container_width=True, theme=None)
