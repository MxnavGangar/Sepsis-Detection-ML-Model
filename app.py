import customtkinter as ctk
import numpy as np
import joblib
import datetime
import math
from tkinter import filedialog, Canvas

# ── Model ─────────────────────────────────────────────────────────────────────
model = joblib.load("model/xgb_model.pkl")

# ── Dark Palette ──────────────────────────────────────────────────────────────
C = {
    # Backgrounds
    "win":          "#0D1117",
    "sidebar":      "#161B22",
    "topbar":       "#161B22",
    "card":         "#1C2333",
    "card_inner":   "#21293A",
    "input_bg":     "#0D1117",

    # Borders
    "border":       "#2A3547",
    "border_light": "#344060",

    # Brand blue
    "blue":         "#2F81F7",
    "blue_dark":    "#1A6EE0",
    "blue_muted":   "#1C3A6B",

    # Text — WCAG AA contrast on dark backgrounds
    "t1":           "#E6EDF3",   # primary — near white
    "t2":           "#8B949E",   # secondary
    "t3":           "#484F58",   # muted / placeholders
    "t_sidebar":    "#CDD5DF",
    "t_on_blue":    "#FFFFFF",

    # Severity
    "green":        "#3FB950",
    "green_dim":    "#1A3A27",
    "green_border": "#2A6B3A",
    "amber":        "#D29922",
    "amber_dim":    "#3A2A0A",
    "amber_border": "#8A6A10",
    "red":          "#F85149",
    "red_dim":      "#3A1010",
    "red_border":   "#8A2020",

    # Divider
    "div":          "#21293A",
}

# ── Fonts ─────────────────────────────────────────────────────────────────────
F = {
    "brand":        ("Segoe UI Semibold", 14),
    "brand_sub":    ("Segoe UI", 9),
    "nav":          ("Segoe UI Semibold", 11),
    "nav_sec":      ("Segoe UI Semibold", 9),
    "topbar_title": ("Segoe UI Semibold", 15),
    "topbar_sub":   ("Segoe UI", 10),
    "stat_val":     ("Segoe UI Semibold", 22),
    "stat_lbl":     ("Segoe UI", 10),
    "card_title":   ("Segoe UI Semibold", 10),
    "field_key":    ("Segoe UI Semibold", 12),
    "field_name":   ("Segoe UI", 10),
    "field_unit":   ("Segoe UI", 9),
    "entry":        ("Segoe UI", 13),
    "verdict":      ("Segoe UI Bold", 20),
    "verdict_prob": ("Segoe UI Semibold", 13),
    "verdict_note": ("Segoe UI", 10),
    "xai_name":     ("Segoe UI Semibold", 11),
    "xai_val":      ("Segoe UI", 10),
    "btn_primary":  ("Segoe UI Semibold", 12),
    "btn_sec":      ("Segoe UI", 11),
    "log":          ("Consolas", 10),
    "disclaimer":   ("Segoe UI", 9),
}

# ── Window ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Sepsis Clinical Decision Support")
app.geometry("1100x760")
app.configure(fg_color=C["win"])
app.resizable(True, True)
app.minsize(960, 680)

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
sidebar = ctk.CTkFrame(app, fg_color=C["sidebar"], width=200, corner_radius=0)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

# Brand
brand_block = ctk.CTkFrame(sidebar, fg_color=C["win"], corner_radius=0, height=64)
brand_block.pack(fill="x")
brand_block.pack_propagate(False)

ctk.CTkLabel(
    brand_block, text="SepsisGuard",
    font=F["brand"], text_color=C["t1"], anchor="w"
).place(x=18, y=14)

ctk.CTkLabel(
    brand_block, text="Clinical Decision Support v2.1",
    font=F["brand_sub"], text_color=C["t2"], anchor="w"
).place(x=18, y=40)

# Separator
Canvas(sidebar, height=1, bg=C["border"], highlightthickness=0).pack(fill="x")

# Nav section
ctk.CTkLabel(
    sidebar, text="WORKSPACE",
    font=F["nav_sec"], text_color=C["t3"], anchor="w"
).pack(anchor="w", padx=18, pady=(18, 6))

def _nav(label, active=False):
    bg = C["blue_muted"] if active else "transparent"
    tc = C["t1"]         if active else C["t_sidebar"]
    f  = ctk.CTkFrame(sidebar, fg_color=bg, corner_radius=6, height=34)
    f.pack(fill="x", padx=10, pady=2)
    f.pack_propagate(False)
    ctk.CTkLabel(f, text=label, font=F["nav"], text_color=tc, anchor="w").pack(
        side="left", padx=12)

_nav("Patient Analysis", active=True)
_nav("Risk Assessment")
_nav("Audit Log")

# Bottom info
Canvas(sidebar, height=1, bg=C["border"], highlightthickness=0).pack(
    side="bottom", fill="x", pady=(0, 0))

info_frame = ctk.CTkFrame(sidebar, fg_color=C["card"], corner_radius=8)
info_frame.pack(side="bottom", fill="x", padx=12, pady=12)

for label, val in [("Model", "XGBoost Classifier"),
                   ("Features", "8 inputs + 3 derived")]:
    ctk.CTkLabel(info_frame, text=label,
                 font=("Segoe UI Semibold", 9), text_color=C["t3"],
                 anchor="w").pack(anchor="w", padx=12, pady=(8, 0))
    ctk.CTkLabel(info_frame, text=val,
                 font=("Segoe UI", 10), text_color=C["t2"],
                 anchor="w").pack(anchor="w", padx=12, pady=(0, 4))
ctk.CTkLabel(info_frame, text="", height=2).pack()  # bottom padding

# Status
status_row = ctk.CTkFrame(sidebar, fg_color="transparent")
status_row.pack(side="bottom", anchor="w", padx=18, pady=(0, 10))

dot_cv = Canvas(status_row, width=8, height=8, bg=C["sidebar"], highlightthickness=0)
dot_cv.pack(side="left")
dot_cv.create_oval(0, 0, 8, 8, fill="#3FB950", outline="")

ctk.CTkLabel(status_row, text="  System Online",
             font=("Segoe UI", 10), text_color=C["t2"]).pack(side="left")

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═════════════════════════════════════════════════════════════════════════════
main = ctk.CTkFrame(app, fg_color=C["win"], corner_radius=0)
main.pack(side="left", fill="both", expand=True)

# ── Topbar ────────────────────────────────────────────────────────────────────
topbar = ctk.CTkFrame(main, fg_color=C["topbar"], corner_radius=0, height=58)
topbar.pack(fill="x")
topbar.pack_propagate(False)

ctk.CTkLabel(
    topbar, text="Patient Analysis",
    font=F["topbar_title"], text_color=C["t1"], anchor="w"
).place(x=20, y=10)

ctk.CTkLabel(
    topbar,
    text="Enter patient parameters below and press Run Analysis to generate a sepsis risk score.",
    font=F["topbar_sub"], text_color=C["t2"], anchor="w"
).place(x=20, y=36)

# Buttons — right side of topbar
btn_container = ctk.CTkFrame(topbar, fg_color="transparent")
btn_container.place(relx=1.0, rely=0.5, anchor="e", x=-16)

ctk.CTkButton(
    btn_container, text="Run Analysis",
    command=lambda: predict(),
    font=F["btn_primary"], height=34, width=140,
    fg_color=C["blue"], hover_color=C["blue_dark"],
    text_color=C["t_on_blue"], corner_radius=6
).pack(side="right", padx=(8, 0))

ctk.CTkButton(
    btn_container, text="Export Report",
    command=lambda: save_report(),
    font=F["btn_sec"], height=34, width=120,
    fg_color=C["card"], hover_color=C["card_inner"],
    text_color=C["t2"],
    border_color=C["border_light"], border_width=1,
    corner_radius=6
).pack(side="right", padx=(8, 0))

ctk.CTkButton(
    btn_container, text="Clear Fields",
    command=lambda: reset(),
    font=F["btn_sec"], height=34, width=110,
    fg_color=C["card"], hover_color=C["card_inner"],
    text_color=C["t2"],
    border_color=C["border_light"], border_width=1,
    corner_radius=6
).pack(side="right")

Canvas(main, height=1, bg=C["border"], highlightthickness=0).pack(fill="x")

# ── Stat strip ────────────────────────────────────────────────────────────────
stat_outer = ctk.CTkFrame(main, fg_color="transparent")
stat_outer.pack(fill="x", padx=16, pady=(14, 0))

STAT_DEFS = [
    ("Risk Score",       "—", "Sepsis probability",  C["blue"]),
    ("Risk Level",       "—", "Classification tier", C["t2"]),
    ("Primary Factor",   "—", "Top XAI contributor", C["t2"]),
    ("Confidence",       "—", "Prediction certainty",C["t2"]),
]

stat_refs = []  # (val_label, sub_label, accent_canvas)

for i, (title, val, sub, ac_col) in enumerate(STAT_DEFS):
    stat_outer.grid_columnconfigure(i, weight=1)

    cell = ctk.CTkFrame(stat_outer, fg_color=C["card"],
                        border_color=C["border"], border_width=1, corner_radius=8)
    cell.grid(row=0, column=i, sticky="ew",
              padx=(0, 12) if i < 3 else (0, 0), pady=0)

    # Top accent bar
    top_bar = Canvas(cell, height=3, bg=C["card"], highlightthickness=0)
    top_bar.pack(fill="x")
    top_bar.create_rectangle(0, 0, 600, 3, fill=ac_col, outline="")

    inner = ctk.CTkFrame(cell, fg_color="transparent")
    inner.pack(fill="x", padx=14, pady=(10, 12))

    ctk.CTkLabel(inner, text=title, font=F["stat_lbl"],
                 text_color=C["t2"], anchor="w").pack(anchor="w")

    val_lbl = ctk.CTkLabel(inner, text=val, font=F["stat_val"],
                           text_color=ac_col, anchor="w")
    val_lbl.pack(anchor="w", pady=(2, 0))

    sub_lbl = ctk.CTkLabel(inner, text=sub, font=("Segoe UI", 9),
                           text_color=C["t3"], anchor="w")
    sub_lbl.pack(anchor="w")

    stat_refs.append((val_lbl, sub_lbl, top_bar, ac_col))

# ── Two-column layout ─────────────────────────────────────────────────────────
cols_frame = ctk.CTkFrame(main, fg_color="transparent")
cols_frame.pack(fill="both", expand=True, padx=16, pady=14)
cols_frame.grid_columnconfigure(0, weight=55)
cols_frame.grid_columnconfigure(1, weight=45)
cols_frame.grid_rowconfigure(0, weight=1)

# ── LEFT: scrollable ──────────────────────────────────────────────────────────
left = ctk.CTkScrollableFrame(
    cols_frame, fg_color="transparent",
    scrollbar_button_color=C["border"],
    scrollbar_button_hover_color=C["border_light"]
)
left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

# ── RIGHT: scrollable ─────────────────────────────────────────────────────────
right = ctk.CTkScrollableFrame(
    cols_frame, fg_color="transparent",
    scrollbar_button_color=C["border"],
    scrollbar_button_hover_color=C["border_light"]
)
right.grid(row=0, column=1, sticky="nsew")

# ─── Card factory ──────────────────────────────────────────────────────────────
def make_card(parent, title):
    outer = ctk.CTkFrame(parent, fg_color=C["card"],
                         border_color=C["border"], border_width=1, corner_radius=8)
    outer.pack(fill="x", pady=(0, 12))

    # Title row
    title_row = ctk.CTkFrame(outer, fg_color=C["card_inner"], corner_radius=0, height=36)
    title_row.pack(fill="x")
    title_row.pack_propagate(False)

    # Blue left stripe
    stripe = Canvas(title_row, width=3, height=36, bg=C["card_inner"], highlightthickness=0)
    stripe.pack(side="left")
    stripe.create_rectangle(0, 8, 3, 28, fill=C["blue"], outline="")

    ctk.CTkLabel(
        title_row, text=title,
        font=F["card_title"], text_color=C["t2"], anchor="w"
    ).pack(side="left", padx=(10, 0))

    # Divider
    Canvas(outer, height=1, bg=C["border"], highlightthickness=0).pack(fill="x")

    body = ctk.CTkFrame(outer, fg_color=C["card"], corner_radius=0)
    body.pack(fill="both")
    return body

# ═════════════════════════════════════════════════════════════════════════════
#  LEFT — Patient Parameters
# ═════════════════════════════════════════════════════════════════════════════
inp_body = make_card(left, "PATIENT PARAMETERS")

feature_info = {
    "PRG": ("Pregnancies",         "count",  "0 – 17"),
    "PL":  ("Plasma Glucose",      "mg/dL",  "0 – 199"),
    "PR":  ("Diastolic BP",        "mm Hg",  "0 – 122"),
    "SK":  ("Skin Fold Thickness", "mm",     "0 – 99"),
    "TS":  ("Serum Insulin",       "μU/mL",  "0 – 846"),
    "M11": ("BMI",                 "kg/m²",  "0 – 67.1"),
    "BD2": ("Diabetes Pedigree",   "score",  "0.08 – 2.42"),
    "Age": ("Age",                 "years",  "21 – 81"),
}

entries = {}

grid = ctk.CTkFrame(inp_body, fg_color="transparent")
grid.pack(fill="x", padx=14, pady=14)
grid.grid_columnconfigure(0, weight=1)
grid.grid_columnconfigure(1, weight=1)

for i, (key, (label_text, unit, ref)) in enumerate(feature_info.items()):
    col_i = i % 2
    row_i = i // 2

    cell = ctk.CTkFrame(grid, fg_color=C["card_inner"],
                        border_color=C["border"], border_width=1, corner_radius=7)
    px = (0, 8) if col_i == 0 else (0, 0)
    cell.grid(row=row_i, column=col_i, padx=px, pady=5, sticky="ew")

    # Key + unit row
    top = ctk.CTkFrame(cell, fg_color="transparent")
    top.pack(fill="x", padx=12, pady=(10, 0))

    ctk.CTkLabel(
        top, text=key,
        font=F["field_key"], text_color=C["t1"], anchor="w"
    ).pack(side="left")

    ctk.CTkLabel(
        top, text=unit,
        font=F["field_unit"], text_color=C["blue"], anchor="e"
    ).pack(side="right")

    # Full name
    ctk.CTkLabel(
        cell, text=label_text,
        font=F["field_name"], text_color=C["t2"], anchor="w"
    ).pack(anchor="w", padx=12, pady=(2, 0))

    # Ref range
    ctk.CTkLabel(
        cell, text=f"Range: {ref}",
        font=("Segoe UI", 9), text_color=C["t3"], anchor="w"
    ).pack(anchor="w", padx=12, pady=(0, 4))

    # Entry
    ent = ctk.CTkEntry(
        cell,
        height=38,
        font=F["entry"],
        fg_color=C["input_bg"],
        border_color=C["border"],
        border_width=1,
        text_color=C["t1"],
        placeholder_text_color=C["t3"],
        placeholder_text=f"e.g. {ref.split('–')[0].strip()}",
        corner_radius=5
    )
    ent.pack(fill="x", padx=12, pady=(0, 12))

    def _fi(e, w=ent): w.configure(border_color=C["blue"])
    def _fo(e, w=ent): w.configure(border_color=C["border"])
    ent.bind("<FocusIn>",  _fi)
    ent.bind("<FocusOut>", _fo)
    entries[key] = ent

# ═════════════════════════════════════════════════════════════════════════════
#  LEFT — Audit Log
# ═════════════════════════════════════════════════════════════════════════════
log_body = make_card(left, "AUDIT LOG")
log_box = ctk.CTkTextbox(
    log_body, height=96,
    font=F["log"],
    fg_color=C["card_inner"],
    text_color=C["t2"],
    border_width=0, corner_radius=0
)
log_box.pack(fill="x")

def _log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    log_box.configure(state="normal")
    log_box.insert("end", f"  {ts}  [{level:5s}]  {msg}\n")
    log_box.see("end")

_log("System ready. Model xgb_model.pkl loaded successfully.")

# ═════════════════════════════════════════════════════════════════════════════
#  RIGHT — Risk Assessment
# ═════════════════════════════════════════════════════════════════════════════
risk_body = make_card(right, "RISK ASSESSMENT")

# Verdict block
verdict_block = ctk.CTkFrame(
    risk_body, fg_color=C["card_inner"],
    border_color=C["border"], border_width=1, corner_radius=7
)
verdict_block.pack(fill="x", padx=14, pady=(12, 8))

verdict_level_lbl = ctk.CTkLabel(
    verdict_block, text="Awaiting Input",
    font=F["verdict"], text_color=C["t2"]
)
verdict_level_lbl.pack(pady=(16, 4))

verdict_prob_lbl = ctk.CTkLabel(
    verdict_block, text="Probability:  —",
    font=F["verdict_prob"], text_color=C["t2"]
)
verdict_prob_lbl.pack(pady=(0, 8))

verdict_bar = ctk.CTkProgressBar(
    verdict_block, height=10, corner_radius=5,
    progress_color=C["t3"], fg_color=C["border"]
)
verdict_bar.pack(fill="x", padx=20, pady=(0, 8))
verdict_bar.set(0)

Canvas(verdict_block, height=1, bg=C["card_inner"], highlightthickness=0).pack(
    fill="x", padx=14)

verdict_note_lbl = ctk.CTkLabel(
    verdict_block,
    text="Enter patient data and run analysis to generate a risk score.",
    font=F["verdict_note"], text_color=C["t2"],
    wraplength=310, justify="center"
)
verdict_note_lbl.pack(pady=(8, 16))

# ── Gauge ─────────────────────────────────────────────────────────────────────
gauge_cv = Canvas(risk_body, width=340, height=160,
                  bg=C["card"], highlightthickness=0)
gauge_cv.pack(pady=(0, 12))

_pc = [0.0]; _pt = [0.0]; _an = [False]

def _draw_gauge(p):
    gauge_cv.delete("all")
    cx, cy, r = 170, 148, 112

    # Grey track
    gauge_cv.create_arc(cx-r, cy-r, cx+r, cy+r,
        start=0, extent=180, style="arc", outline=C["border_light"], width=22)

    # Zone shading (subtle)
    gauge_cv.create_arc(cx-r, cy-r, cx+r, cy+r,
        start=0,   extent=72,  style="arc", outline="#1A3A27", width=22)
    gauge_cv.create_arc(cx-r, cy-r, cx+r, cy+r,
        start=72,  extent=54,  style="arc", outline="#3A2A0A", width=22)
    gauge_cv.create_arc(cx-r, cy-r, cx+r, cy+r,
        start=126, extent=54,  style="arc", outline="#3A1010", width=22)

    # Active filled arc
    sweep = max(int(p * 180), 0)
    if sweep > 0:
        col = C["green"] if p < 0.40 else (C["amber"] if p < 0.70 else C["red"])
        gauge_cv.create_arc(cx-r, cy-r, cx+r, cy+r,
            start=0, extent=sweep, style="arc", outline=col, width=22)

    # Needle
    ang = math.radians(p * 180)
    nx  = cx + (r - 30) * math.cos(ang)
    ny  = cy - (r - 30) * math.sin(ang)
    gauge_cv.create_line(cx, cy, nx, ny, fill=C["t1"], width=2, capstyle="round")
    gauge_cv.create_oval(cx-5, cy-5, cx+5, cy+5,
                         fill=C["t1"], outline=C["card"], width=2)

    # Zone labels
    gauge_cv.create_text(cx - r + 16, cy + 20, text="LOW",
                         font=("Segoe UI", 8), fill=C["green"],  anchor="center")
    gauge_cv.create_text(cx,           cy + 26, text="MODERATE",
                         font=("Segoe UI", 8), fill=C["amber"],  anchor="center")
    gauge_cv.create_text(cx + r - 16, cy + 20, text="HIGH",
                         font=("Segoe UI", 8), fill=C["red"],    anchor="center")

    # Probability label inside gauge
    pct_text = f"{p*100:.1f}%" if p > 0.01 else "—"
    gauge_cv.create_text(cx, cy - 36, text=pct_text,
                         font=("Segoe UI Semibold", 18),
                         fill=C["t1"], anchor="center")

_draw_gauge(0)

def _anim():
    if not _an[0]: return
    diff = _pt[0] - _pc[0]
    if abs(diff) < 0.003:
        _pc[0] = _pt[0]; _an[0] = False
    else:
        _pc[0] += diff * 0.13
    _draw_gauge(_pc[0])
    app.after(16, _anim)

def _set_gauge(p):
    _pt[0] = p; _an[0] = True; _anim()

# ═════════════════════════════════════════════════════════════════════════════
#  RIGHT — Contributing Factors
# ═════════════════════════════════════════════════════════════════════════════
xai_body = make_card(right, "CONTRIBUTING FACTORS")

xai_hdr = ctk.CTkFrame(xai_body, fg_color="transparent")
xai_hdr.pack(fill="x", padx=14, pady=(10, 4))
ctk.CTkLabel(xai_hdr, text="Feature",   font=("Segoe UI Semibold", 9),
             text_color=C["t3"], width=100, anchor="w").pack(side="left")
ctk.CTkLabel(xai_hdr, text="Influence", font=("Segoe UI Semibold", 9),
             text_color=C["t3"]).pack(side="left", expand=True, fill="x", padx=(0, 8))
ctk.CTkLabel(xai_hdr, text="Score",     font=("Segoe UI Semibold", 9),
             text_color=C["t3"], width=64, anchor="e").pack(side="right")

Canvas(xai_body, height=1, bg=C["border"], highlightthickness=0).pack(
    fill="x", padx=14, pady=(0, 6))

xai_rows = []
for _ in range(5):
    row = ctk.CTkFrame(xai_body, fg_color="transparent", height=36)
    row.pack(fill="x", padx=14, pady=3)
    row.pack_propagate(False)

    nl = ctk.CTkLabel(row, text="—", font=F["xai_name"],
                      text_color=C["t2"], width=100, anchor="w")
    nl.pack(side="left")

    bar = ctk.CTkProgressBar(row, height=8, corner_radius=3,
                              progress_color=C["t3"], fg_color=C["border"])
    bar.pack(side="left", expand=True, fill="x", padx=(0, 8))
    bar.set(0)

    vl = ctk.CTkLabel(row, text="", font=F["xai_val"],
                      text_color=C["t3"], width=64, anchor="e")
    vl.pack(side="right")
    xai_rows.append((nl, bar, vl))
ctk.CTkLabel(xai_body, text="", height=4).pack()  # bottom padding

def _update_xai(top5):
    mx = max(abs(v) for _, v in top5) if top5 else 1.0
    for i, (nl, bar, vl) in enumerate(xai_rows):
        if i < len(top5):
            name, val = top5[i]
            col  = C["red"] if val > 0 else C["green"]
            sign = "+" if val > 0 else ""
            nl.configure(text=name,  text_color=C["t1"])
            bar.configure(progress_color=col)
            bar.set(min(abs(val) / mx, 1.0))
            vl.configure(text=f"{sign}{val:.3f}", text_color=col)
        else:
            nl.configure(text="—", text_color=C["t2"])
            bar.configure(progress_color=C["t3"])
            bar.set(0)
            vl.configure(text="")

# ═════════════════════════════════════════════════════════════════════════════
#  RIGHT — Disclaimer
# ═════════════════════════════════════════════════════════════════════════════
disc = ctk.CTkFrame(right, fg_color=C["blue_muted"],
                    border_color=C["border_light"], border_width=1, corner_radius=8)
disc.pack(fill="x", pady=(0, 12))
ctk.CTkLabel(
    disc,
    text="This tool provides AI-generated clinical decision support only.\n"
         "It does not replace professional medical judgement or diagnosis.",
    font=F["disclaimer"], text_color="#8BBCE8",
    justify="left", anchor="w", wraplength=340
).pack(padx=14, pady=10)

# ═════════════════════════════════════════════════════════════════════════════
#  LOGIC
# ═════════════════════════════════════════════════════════════════════════════
def _explain(values):
    imps  = model.feature_importances_
    names = ["PRG", "PL", "PR", "SK", "TS", "M11", "BD2", "Age",
             "Glucose×BMI", "Age×BMI", "Glucose×Age"]
    return sorted(
        [(names[i], values[i] * imps[i]) for i in range(len(values))],
        key=lambda x: abs(x[1]), reverse=True
    )[:5]


def predict():
    try:
        raw = [float(entries[k].get()) for k in feature_info]
        prg, pl, pr, sk, ts, m11, bd2, age = raw
        values = raw + [pl * m11, age * m11, pl * age]
        prob   = model.predict_proba(np.array(values).reshape(1, -1))[0][1]

        _set_gauge(prob)

        if prob < 0.40:
            level = "Low Risk"
            lc, lbg, lbr = C["green"], C["green_dim"], C["green_border"]
            note = "No immediate action indicated. Continue routine monitoring."
        elif prob < 0.70:
            level = "Moderate Risk"
            lc, lbg, lbr = C["amber"], C["amber_dim"], C["amber_border"]
            note = "Heightened vigilance recommended. Reassess within 2 hours."
        else:
            level = "High Risk"
            lc, lbg, lbr = C["red"], C["red_dim"], C["red_border"]
            note = "Immediate clinical review recommended. Consider intervention."

        verdict_level_lbl.configure(text=level, text_color=lc)
        verdict_prob_lbl.configure(text=f"Probability:  {prob * 100:.1f}%", text_color=lc)
        verdict_bar.configure(progress_color=lc)
        verdict_bar.set(prob)
        verdict_block.configure(fg_color=lbg, border_color=lbr)
        verdict_note_lbl.configure(text=note, text_color=lc)

        top5 = _explain(values)
        _update_xai(top5)

        conf = (1.0 - 2.0 * abs(prob - 0.5)) * 100

        # Stat strip updates
        stat_refs[0][0].configure(text=f"{prob*100:.1f}%", text_color=lc)
        stat_refs[0][2].create_rectangle(0, 0, 600, 3, fill=lc, outline="")

        stat_refs[1][0].configure(text=level.split()[0], text_color=lc)
        stat_refs[1][2].create_rectangle(0, 0, 600, 3, fill=lc, outline="")

        stat_refs[2][0].configure(text=top5[0][0] if top5 else "—",
                                   text_color=C["t1"],
                                   font=("Segoe UI Semibold", 15))
        stat_refs[3][0].configure(text=f"{conf:.0f}%", text_color=C["t1"])

        _log(f"Analysis — {level} ({prob*100:.1f}%)  "
             f"[Glucose={pl}, BMI={m11}, Age={int(age)}]")

    except ValueError:
        _log("Input error: all fields require valid numeric values.", level="ERROR")
    except Exception as ex:
        _log(f"Unexpected error: {ex}", level="ERROR")


def reset():
    for e in entries.values():
        e.delete(0, "end")
        e.configure(border_color=C["border"])
    _set_gauge(0.0)
    verdict_level_lbl.configure(text="Awaiting Input", text_color=C["t2"])
    verdict_prob_lbl.configure(text="Probability:  —", text_color=C["t2"])
    verdict_bar.configure(progress_color=C["t3"])
    verdict_bar.set(0)
    verdict_block.configure(fg_color=C["card_inner"], border_color=C["border"])
    verdict_note_lbl.configure(
        text="Enter patient data and run analysis to generate a risk score.",
        text_color=C["t2"])
    _update_xai([])
    for vl, sl, top_bar, ac_col in stat_refs:
        vl.configure(text="—", text_color=ac_col, font=F["stat_val"])
        top_bar.create_rectangle(0, 0, 600, 3, fill=ac_col, outline="")
    _log("Fields cleared. Ready for new patient data.")


def save_report():
    try:
        p   = _pc[0]
        now = datetime.datetime.now()
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
            initialfile=f"sepsis_report_{now.strftime('%Y%m%d_%H%M%S')}"
        )
        if not path:
            return
        level = "Low Risk" if p < 0.40 else ("Moderate Risk" if p < 0.70 else "High Risk")
        with open(path, "w") as f:
            f.write("SEPSIS CLINICAL DECISION SUPPORT — PATIENT REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated   : {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model       : XGBoost Classifier (xgb_model.pkl)\n\n")
            f.write("PATIENT PARAMETERS\n" + "-" * 40 + "\n")
            for k, (lbl, unit, _) in feature_info.items():
                val = entries[k].get() or "N/A"
                f.write(f"  {k:<5}  {lbl:<25} {val:>10} {unit}\n")
            f.write(f"\nRISK ASSESSMENT\n" + "-" * 40 + "\n")
            f.write(f"  Risk Level  : {level}\n")
            f.write(f"  Probability : {p * 100:.1f}%\n\n")
            f.write("NOTE: AI-generated output. Not a substitute for clinical judgement.\n")
        _log(f"Report saved: {path}")
    except Exception as ex:
        _log(f"Save failed: {ex}", level="ERROR")


app.mainloop()
