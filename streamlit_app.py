# app.py
"""
Sensor coverage: cone ∩ floor (top/side views), with non-tilted vs tilted sensors.
Port of the matplotlib "reworked final" to Streamlit + Plotly.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ========= Geometry (same math) =========
def ellipse_from_cone(h, fov_deg, tilt_deg, yaw_deg=0.0):
    """
    Apex at (0,0,h). Floor is z=0. Cone axis points downward, tilted by tilt_deg
    toward yaw_deg in the xy-plane.

    Returns:
        center_xy: np.array([cx, cy])  (in the cone frame)
        axes_ab: (a, b) semi-axes (a >= b)
        angle_deg: major-axis angle in the floor plane (deg, CCW from +x)
    """
    a = np.array([0.0, 0.0, h])  # apex
    alpha = np.deg2rad(fov_deg / 2.0)

    # Axis direction (unit vector)
    theta = np.deg2rad(tilt_deg)
    psi   = np.deg2rad(yaw_deg)
    d = np.array([
        np.sin(theta)*np.cos(psi),
        np.sin(theta)*np.sin(psi),
        -np.cos(theta)
    ], dtype=float)
    d /= np.linalg.norm(d)

    # Quadric of cone
    M3 = np.outer(d, d) - (np.cos(alpha)**2) * np.eye(3)

    # Intersect with plane z=0
    M  = M3[:2, :2]
    q  = -2.0 * (a @ M3)[:2]
    k  =  (a @ M3 @ a)

    # Ellipse center (in plane)
    xc = -0.5 * np.linalg.solve(M, q)

    # Canonical form to get axes
    kprime = k + q @ xc + xc @ M @ xc
    evals, evecs = np.linalg.eigh(M)
    axes = np.sqrt(-kprime / evals)          # semi-axes, order unknown
    order = np.argsort(axes)[::-1]
    axes  = axes[order]
    evecs = evecs[:, order]
    angle_deg = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return xc, (axes[0], axes[1]), angle_deg


# ========= Model helpers (mirrors your SensorView logic) =========
def compute_sensor(sensorX, sensorY, height, min_side, fov, tilt, yaw, name):
    """Compute everything your matplotlib script exposes for one sensor."""
    alpha = fov / 2.0
    h = height - min_side

    # Guard (same as your code)
    if (alpha + tilt) >= 89.9:
        raise ValueError("Cone nearly parallel to floor: reduce tilt or opening angle.")

    # Side-view distances along tilt axis (same as your calcFloorCoverEllipse)
    leftFloor  = h * np.tan(np.deg2rad(alpha - tilt))
    rightFloor = h * np.tan(np.deg2rad(alpha + tilt))

    # Exact ellipse via quadric ∩ plane
    (cx, cy), (a, b), angle_deg = ellipse_from_cone(h=h, fov_deg=fov, tilt_deg=tilt, yaw_deg=yaw)

    # World coords (sensor at sensorX,sensorY)
    cx_w = sensorX + cx
    cy_w = sensorY + cy

    # Diameters
    ellipse_width  = 2.0 * a
    ellipse_height = 2.0 * b

    # Side-view convention in your code
    r0 = np.hypot(sensorX, sensorY)     # usually 0, kept for compatibility
    ellipse_start = r0 - leftFloor
    ellipse_end   = ellipse_start + (leftFloor + rightFloor)
    center_view   = height * np.tan(np.deg2rad(tilt))

    return dict(
        name=name,
        # inputs
        sensorX=sensorX, sensorY=sensorY,
        height=height, min_side=min_side,
        fov=fov, tilt=tilt, yaw=yaw,
        alpha=alpha, h=h,
        # ellipse in world
        cx=cx_w, cy=cy_w,
        a=a, b=b, angle_deg=angle_deg,
        ellipse_width=ellipse_width, ellipse_height=ellipse_height,
        area=np.pi * a * b,
        # side-view distances / references
        leftFloor=leftFloor, rightFloor=rightFloor,
        ellipse_start=ellipse_start, ellipse_end=ellipse_end,
        center_view=center_view,
        # compat
        ellipse_origindistance=np.hypot(cx, cy),
    )


def ellipse_poly(cx, cy, a, b, angle_deg, n=200):
    """Parametric ellipse for Plotly."""
    t = np.linspace(0, 2*np.pi, n)
    ca, sa = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    R = np.array([[ca, -sa], [sa, ca]])
    pts = R @ np.vstack([a*np.cos(t), b*np.sin(t)])
    return cx + pts[0], cy + pts[1]


# ========= UI =========
st.set_page_config(page_title="Sensor Coverage — Cone ∩ Floor", layout="wide")
st.title("Sensor Coverage — Cone ∩ Floor")

# --- Defaults (copied from your script) ---
DEFAULTS = dict(
    sensorHeight       = 2.7,
    sensorX            = 0.0,
    sensorY            = 0.0,
    sensorRoomRotation = 0.0,
    minimumSideHeight  = 1.3,
    sensorTiltAngle    = 10.0,
    sensorOpeningAngle = 116.0,
)

# --- Controls (sliders + advanced) ---
c1, c2, c3, c4 = st.columns(4)
height = c1.slider("Height [m]", 1.3, 4.3, float(DEFAULTS["sensorHeight"]), 0.01)
fov    = c2.slider("Opening Angle [°]", 10.0, 130.0, float(DEFAULTS["sensorOpeningAngle"]), 0.5)
tilt   = c3.slider("Tilt [°]", 0.1, 30.0, float(DEFAULTS["sensorTiltAngle"]), 0.1)
yaw    = c4.slider("Rotation (Yaw) [°]", 0.0, 90.0, float(DEFAULTS["sensorRoomRotation"]), 0.1)

with st.expander("Advanced", expanded=False):
    colA, colB, colC = st.columns(3)
    sensorX   = colA.number_input("Sensor X [m]", value=float(DEFAULTS["sensorX"]), step=0.1, format="%.3f")
    sensorY   = colB.number_input("Sensor Y [m]", value=float(DEFAULTS["sensorY"]), step=0.1, format="%.3f")
    min_side  = colC.number_input("Minimum side height [m]", value=float(DEFAULTS["minimumSideHeight"]),
                                  min_value=0.0, max_value=float(height), step=0.05, format="%.2f")

col_btn1, col_btn2, _ = st.columns([1,1,6])
autoscale = col_btn1.checkbox("Autoscale plots", value=True)
rescale_click = col_btn2.button("Rescale now")

# --- Compute both sensors (non-tilted vs tilted) ---
try:
    base = compute_sensor(sensorX, sensorY, height, min_side, fov, tilt=0.0, yaw=yaw, name="non-tilted")
    til  = compute_sensor(sensorX, sensorY, height, min_side, fov, tilt=tilt,  yaw=yaw, name="tilted")
except ValueError as e:
    st.error(str(e))
    st.stop()

# --- KPIs (same semantics as your GUI header) ---
right_gain = til["ellipse_end"]   - base["ellipse_end"]
left_delta = til["ellipse_start"] - base["ellipse_start"]
st.markdown(
    f"**Tilt:** {tilt:.1f}°  |  **Right coverage**: +{right_gain:.2f} m  |  "
    f"**Left coverage**: {left_delta:.2f} m"
)

# --- Tables (matches your UI table content + one extra with side-view metrics) ---
def table_rows(sensor):
    return {
        "Name": sensor["name"],
        "Center X [m]": f"{sensor['cx']:.3f}",
        "Center Y [m]": f"{sensor['cy']:.3f}",
        "Center distance [m]": f"{sensor['ellipse_origindistance']:.3f}",
        "Major diameter (2a) [m]": f"{2*sensor['a']:.3f}",
        "Minor diameter (2b) [m]": f"{2*sensor['b']:.3f}",
        "Area (πab) [m²]": f"{sensor['area']:.3f}",
    }

def table_rows_side(sensor):
    return {
        "Name": sensor["name"],
        "Left floor [m]": f"{sensor['leftFloor']:.3f}",
        "Right floor [m]": f"{sensor['rightFloor']:.3f}",
        "Ellipse start [m]": f"{sensor['ellipse_start']:.3f}",
        "Ellipse end [m]": f"{sensor['ellipse_end']:.3f}",
        "Axis hit (center_view) [m]": f"{sensor['center_view']:.3f}",
    }

st.subheader("Ellipse dimensions")
df_dim = pd.DataFrame([table_rows(base), table_rows(til)]).set_index("Name")
st.dataframe(df_dim, use_container_width=True)

with st.expander("Side-view metrics", expanded=False):
    df_side = pd.DataFrame([table_rows_side(base), table_rows_side(til)]).set_index("Name")
    st.dataframe(df_side, use_container_width=True)

col_dl1, col_dl2, _ = st.columns([1,1,6])
col_dl1.download_button("Download dimensions (CSV)", data=df_dim.to_csv().encode("utf-8"),
                        file_name="ellipse_dimensions.csv", mime="text/csv")
col_dl2.download_button("Download side-view metrics (CSV)", data=df_side.to_csv().encode("utf-8"),
                        file_name="side_view_metrics.csv", mime="text/csv")

# --- Plot helpers (apply your autoscale logic when requested) ---
def side_view_figure():
    fig = go.Figure()
    colors = {"non-tilted": "#3b82f6", "tilted": "#22c55e"}

    for s in (base, til):
        col = colors[s["name"]]
        # slanted edges to min_side
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
        # vertical drops to floor
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
    )
    # equal aspect
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # optional autoscale following your _rescale()
    if autoscale or rescale_click:
        far = max(abs(base["ellipse_end"]), abs(til["ellipse_end"]))
        x0, x1 = -0.1*far, 1.05*far
        y1 = max(base["height"], til["height"]) * 1.1
        fig.update_xaxes(range=[x0, x1])
        fig.update_yaxes(range=[-0.2, y1])
    return fig


def top_view_figure():
    fig = go.Figure()
    colors = {"non-tilted": "#3b82f6", "tilted": "#22c55e"}

    # Non-tilted footprint
    x0, y0 = ellipse_poly(base["cx"], base["cy"], base["a"], base["b"], base["angle_deg"])
    fig.add_trace(go.Scatter(
        x=x0, y=y0, mode="lines", fill="toself",
        name="non-tilted footprint",
        line=dict(width=2, color=colors["non-tilted"]),
        fillcolor="rgba(59,130,246,0.15)"
    ))

    # Tilted footprint
    x1, y1 = ellipse_poly(til["cx"], til["cy"], til["a"], til["b"], til["angle_deg"])
    fig.add_trace(go.Scatter(
        x=x1, y=y1, mode="lines", fill="toself",
        name="tilted footprint",
        line=dict(width=2, color=colors["tilted"]),
        fillcolor="rgba(34,197,94,0.15)"
    ))

    # Orientation / coverage guide (yaw-based) using tilted distances
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
    )

    if autoscale or rescale_click:
        # mirror your _rescale(): center around both ellipses
        max_r = max(base["ellipse_width"], til["ellipse_width"]) * 0.55
        cx = (base["cx"] + til["cx"]) / 2.0
        cy = (base["cy"] + til["cy"]) / 2.0
        pad = 0.15 * max_r
        fig.update_xaxes(range=[cx - max_r - pad, cx + max_r + pad])
        fig.update_yaxes(range=[cy - max_r - pad, cy + max_r + pad])
    return fig


# --- Layout: side-by-side like your figure ---
cL, cR = st.columns(2)
cL.plotly_chart(side_view_figure(), use_container_width=True, theme=None)
cR.plotly_chart(top_view_figure(),  use_container_width=True, theme=None)
