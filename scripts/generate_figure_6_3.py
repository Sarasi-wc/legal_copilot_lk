import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

FONT = "Times New Roman"

# --- Data ---
systems = [
    "LegalBot-EC\n(Rivas-Echeverría et al., 2025)\nEcuador",
    "Vietnamese Law QA\n(Nguyen et al., 2024)\nVietnam",
    "LexRAG\n(Li et al., 2025)\nChina",
    "LegalBench-RAG\n(Pipitone & Alami, 2024)\nUS (contracts)",
    "This work\n(Sri Lankan Legal Copilot)\nSri Lanka",
]

columns = [
    "Jurisdiction-\nspecific corpus",
    "Hybrid\nretrieval",
    "Cross-encoder\nreranking",
    "NLI faithfulness\nverification",
    "Abstention\nmechanism",
    "Faithfulness\nreported",
    "MRR\nreported",
]

# True = present/yes, False = absent/no, string = literal value
data = [
    [True,  False, False, False, "—",  "—",     "—"],
    [True,  "—",   False, False, "—",  "—",     "—"],
    [True,  True,  False, False, "—",  "—",     "—"],
    [True,  "—",   False, False, "—",  "—",     "—"],
    [True,  True,  True,  True,  True, "0.900", "0.832"],
]

PRESENT_COLOR   = "#c8e6c9"
ABSENT_COLOR    = "#ffcdd2"
THIS_WORK_COLOR = "#bbdefb"
HEADER_COLOR    = "#37474f"
SYSTEM_BG       = "#eceff1"
LEGEND_ROW_BG   = "#f5f5f5"

n_rows     = len(systems)
n_cols     = len(columns)
sys_col_w  = 3.2
fig_w      = 18
header_h   = 1.10
row_h      = 1.00
legend_h   = 0.80
top_margin = 0.15
fig_h      = top_margin + header_h + n_rows * row_h + legend_h + 0.20

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

feat_col_w = (fig_w - sys_col_w) / n_cols
top_y      = fig_h - top_margin

SYMBOLS = {"✓", "✗", "—", "Yes", "No", "0.900", "0.832"}

def txt(x, y, s, fs=14, fw="bold", color="black", ha="center", va="center", style="normal"):
    # Use DejaVu Sans for symbols/values; Times New Roman for all other text
    font = FONT if s not in SYMBOLS else "DejaVu Sans"
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=color, fontfamily=font, fontstyle=style,
            multialignment="center")

# ── Column headers ────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0, top_y - header_h), sys_col_w, header_h,
                             boxstyle="square,pad=0", fc=HEADER_COLOR, ec="white", lw=1.5))
txt(sys_col_w / 2, top_y - header_h / 2, "System (Jurisdiction)", fs=14, color="white")

for j, col in enumerate(columns):
    x = sys_col_w + j * feat_col_w
    ax.add_patch(FancyBboxPatch((x, top_y - header_h), feat_col_w, header_h,
                                 boxstyle="square,pad=0", fc=HEADER_COLOR, ec="white", lw=1.5))
    txt(x + feat_col_w / 2, top_y - header_h / 2, col, fs=14, color="white")

# ── Data rows ─────────────────────────────────────────────────────────────────
for i, (system, row) in enumerate(zip(systems, data)):
    y = top_y - header_h - (i + 1) * row_h
    is_tw = i == len(systems) - 1

    ax.add_patch(FancyBboxPatch((0, y), sys_col_w, row_h,
                                 boxstyle="square,pad=0",
                                 fc=THIS_WORK_COLOR if is_tw else SYSTEM_BG,
                                 ec="white", lw=1.2))
    txt(sys_col_w / 2, y + row_h / 2, system, fs=14,
        color="#1a237e" if is_tw else "black")

    for j, val in enumerate(row):
        x = sys_col_w + j * feat_col_w
        if val == "—":
            bg, label, fc, fs = "#eeeeee", "—", "#424242", 18
        elif isinstance(val, str):
            bg, label, fc, fs = THIS_WORK_COLOR, val, "#1a237e", 14
        elif val:
            bg = THIS_WORK_COLOR if is_tw else PRESENT_COLOR
            label = "✓"
            fc    = "#1a237e" if is_tw else "#1b5e20"
            fs    = 20
        else:
            bg, label, fc, fs = ABSENT_COLOR, "✗", "#b71c1c", 20

        ax.add_patch(FancyBboxPatch((x, y), feat_col_w, row_h,
                                     boxstyle="square,pad=0", fc=bg, ec="white", lw=1.2))
        txt(x + feat_col_w / 2, y + row_h / 2, label, fs=fs, color=fc)

# ── Legend row ────────────────────────────────────────────────────────────────
leg_y = top_y - header_h - n_rows * row_h - legend_h

# background spanning full width
ax.add_patch(FancyBboxPatch((0, leg_y), fig_w, legend_h,
                             boxstyle="square,pad=0", fc=LEGEND_ROW_BG,
                             ec="#bdbdbd", lw=1.2))

# three equal sections
items = [
    ("✓", PRESENT_COLOR,   "#1b5e20", "feature present"),
    ("✗", ABSENT_COLOR,    "#b71c1c", "feature absent"),
    ("—", "#eeeeee",       "#424242", "not reported"),
    ("✓", THIS_WORK_COLOR, "#1a237e", "this work"),
]
seg_w = fig_w / 4
for k, (sym, bg, fc, label) in enumerate(items):
    cx = seg_w * k + seg_w / 2

    # coloured symbol box
    box_w, box_h = 0.55, 0.36
    bx = cx - 1.1
    by = leg_y + (legend_h - box_h) / 2
    ax.add_patch(FancyBboxPatch((bx, by), box_w, box_h,
                                 boxstyle="round,pad=0.04", fc=bg,
                                 ec="#9e9e9e", lw=0.8))
    txt(bx + box_w / 2, by + box_h / 2, sym, fs=13, color=fc)

    # "= label" text to the right of the box
    txt(bx + box_w + 0.15, leg_y + legend_h / 2,
        f"= {label}", fs=10.5, fw="bold", ha="left", color="#212121")

plt.tight_layout(pad=0.2)
out = "results/eval_post_qac_fixes/figure_6_3.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
