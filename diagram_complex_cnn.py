import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 1776, 880
FILL = "#BFE0FB"          # màu nền hộp
EDGE_TOP = "#1F4E79"      # viền hộp hàng trên
EDGE_IN = "#3B78B4"       # viền hộp trong các panel
PANEL_EDGE = "#5B7FA6"
PANEL_HEAD = "#DCEBFA"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                              "Nimbus Roman", "DejaVu Serif"]

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)         
ax.axis("off")

def box(cx, cy, w, h, text, fs=10, bold=False, edge=EDGE_IN, lw=1.2, fill=FILL):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,boxstyle="round,pad=0,rounding_size=4",fc=fill, ec=edge, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,fontweight="bold" if bold else "normal", zorder=3, linespacing=1.25)

def label(x, y, text, fs=10, bold=False, bg=False):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,fontweight="bold" if bold else "normal", zorder=4,bbox=dict(fc="white", ec="none", pad=1.5) if bg else None)

def arrow(x1, y1, x2, y2, head=True):
    style = "-|>" if head else "-"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),arrowprops=dict(arrowstyle=style, color="black", lw=1.0,shrinkA=0, shrinkB=0, mutation_scale=9),zorder=1)

def line(xs, ys):
    ax.plot(xs, ys, color="black", lw=1.0, zorder=1, solid_capstyle="butt")
ax.text(W / 2, 36, "Complex CNN Architecture (4 Residual Blocks)", ha="center",va="center", fontsize=27, fontweight="bold")

# flow chính của mạng
Y = 183
top = [
    (80, Y, 140, 86, "Input Image\n(160 × 160 × 3)", True),
    (240, Y, 136, 86, "Conv2D\n(3 × 3, 16 filters,\nsame)", False),
    (388, Y, 127, 86, "Batch\nNormalization", False),
    (513, Y, 90, 86, "ReLU", False),
    (650, Y, 130, 127, "Residual\nBlock 1\n(16 filters,\nstride 1)", True),
    (811, Y, 130, 127, "Residual\nBlock 2\n(32 filters,\nstride 2)", True),
    (973, Y, 130, 127, "Residual\nBlock 3\n(64 filters,\nstride 2)", True),
    (1136, Y, 130, 127, "Residual\nBlock 4\n(128 filters,\nstride 2)", True),
    (1258, Y, 73, 94, "Global\nAverage\nPooling", False),
    (1345, Y, 69, 94, "Dense\n(64,\nReLU)", False),
    (1438, Y, 84, 94, "Dropout\n(0.4)", True),
    (1541, Y, 91, 94, "Dense\n(1,\nSigmoid)", False),
    (1688, Y, 153, 94, "Output:\nP(AI-generated)", False),
]
for cx, cy, w, h, text, bold in top:
    box(cx, cy, w, h, text, fs=10.5, bold=bold, edge=EDGE_TOP, lw=1.6)

# mũi tên nối các hộp hàng trên
for a, b in zip(top[:-1], top[1:]):
    arrow(a[0] + a[2] / 2, Y, b[0] - b[2] / 2, Y)

# kích thước đầu ra dưới mỗi residual block
for cx, shape in [(650, "(160 × 160 × 16)"), (811, "(80 × 80 × 32)"),(973, "(40 × 40 × 64)"), (1136, "(20 × 20 × 128)")]:
    label(cx, 267, shape, fs=10.5)

blocks = [
    # x0 của panel, số filter, stride, shape đầu vào, shape đầu ra
    (37, 16, 1, "(160 × 160 × 16)", "(160 × 160 × 16)"),
    (470, 32, 2, "(160 × 160 × 16)", "(80 × 80 × 32)"),
    (900, 64, 2, "(80 × 80 × 32)", "(40 × 40 × 64)"),
    (1331, 128, 2, "(40 × 40 × 64)", "(20 × 20 × 128)"),
]
PANEL_W, PANEL_TOP, PANEL_BOT, HEAD_H = 408, 302, 831, 44
MAIN_OFF, SHORT_OFF, MID_OFF = 100, 308, 201 
MAIN_W, SHORT_W = 156, 152
FS = 9.2

for i, (x0, f, s, shape_in, shape_out) in enumerate(blocks, start=1):
    # khung nét đứt + dải tiêu đề
    ax.add_patch(FancyBboxPatch((x0, PANEL_TOP), PANEL_W, HEAD_H,boxstyle="round,pad=0,rounding_size=4",fc=PANEL_HEAD, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((x0, PANEL_TOP), PANEL_W, PANEL_BOT - PANEL_TOP,boxstyle="round,pad=0,rounding_size=6",fc="none", ec=PANEL_EDGE, lw=1.1,ls=(0, (5, 3)), zorder=0))
    label(x0 + PANEL_W / 2, PANEL_TOP + 23,"Residual Block %d (%d filters, stride %d)" % (i, f, s), fs=11.5)
    mx, sx, cx = x0 + MAIN_OFF, x0 + SHORT_OFF, x0 + MID_OFF

    # đầu vào + chỗ tách hai nhánh
    label(cx, 358, "Input", fs=10)
    label(cx, 379, shape_in, fs=10)
    drop_l, drop_r = mx - 55, sx + 62
    line([cx, cx], [388, 398])
    line([drop_l, drop_r], [398, 398])
    label(mx, 420, "Main branch", fs=10, bg=True)
    label(sx, 421, "Shortcut branch", fs=10, bg=True)

    # main branch
    main = [
        (457, 48, "Conv2D\n(3 × 3, %d, s=%d)" % (f, s)),
        (504, 30, "BatchNorm"),
        (543, 30, "ReLU"),
        (593, 48, "Conv2D\n(3 × 3, %d, s=1)" % f),
        (639, 30, "BatchNorm"),
    ]
    for cy, h, text in main:
        box(mx, cy, MAIN_W, h, text, fs=FS)
    arrow(drop_l, 398, drop_l, main[0][0] - main[0][1] / 2)      
    for (cy1, h1, _), (cy2, h2, _) in zip(main[:-1], main[1:]):
        arrow(mx, cy1 + h1 / 2, mx, cy2 - h2 / 2)

    # shortcut branch
    if s == 1:                               
        short = [(504, 68, "Identity\n(%d channels,\nsame size)" % f)]
    else:                                        # các block còn lại: Conv 1x1 + BN
        short = [(493, 52, "Conv2D\n(1 × 1, %d, s=%d)" % (f, s)),
                 (544, 30, "BatchNorm")]
    for cy, h, text in short:
        box(sx, cy, SHORT_W, h, text, fs=FS)
    arrow(drop_r, 398, drop_r, short[0][0] - short[0][1] / 2)
    for (cy1, h1, _), (cy2, h2, _) in zip(short[:-1], short[1:]):
        arrow(sx, cy1 + h1 / 2, sx, cy2 - h2 / 2)

    # Add -> ReLU -> Output
    add_w, add_cy = 105, 690
    box(cx, add_cy, add_w, 28, "Add", fs=FS)
    box(cx, 735, 107, 30, "ReLU", fs=FS)
    arrow(cx, add_cy + 14, cx, 735 - 15)
    # main branch đi xuống rồi sang phải vào Add
    line([mx, mx], [main[-1][0] + main[-1][1] / 2, add_cy])
    arrow(mx, add_cy, cx - add_w / 2, add_cy)
    # shortcut branch đi xuống rồi sang trái vào Add
    last_s = short[-1]
    line([sx, sx], [last_s[0] + last_s[1] / 2, add_cy])
    arrow(sx, add_cy, cx + add_w / 2, add_cy)
    # output
    arrow(cx, 735 + 15, cx, 768)
    label(cx, 783, "Output", fs=10, bold=True)
    label(cx, 803, shape_out, fs=10)

out = sys.argv[1] if len(sys.argv) > 1 else "complex_cnn_architecture.png"
fig.savefig(out, dpi=100, facecolor="white")
print("Đã lưu:", out)