"""Open-layout vector schematic of the study's evaluation design.

All counts, periods and architecture relationships are drawn from the current
Word manuscript and model code. Tensor glyphs denote structural objects rather
than observed values. Colour is confined to the input and model objects;
the open-layout canvas stays white.
"""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib import colors as mcolors
from PIL import Image

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
QA = HERE / 'qa'
QA.mkdir(exist_ok=True)
WIDTH_MM, HEIGHT_MM, DPI = 190, 138, 600
INK = '#222222'
RED = '#D62728'
BLUE = '#1F5FA8'
GREEN = '#578D75'
GOLD = '#B5862D'
PURPLE = '#806486'
TEAL = '#328E98'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.unicode_minus': True, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': DPI,
})
fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WIDTH_MM)
ax.set_ylim(0, HEIGHT_MM)
ax.axis('off')
texts = []


def pale(c, amount=.85):
    rgb = np.array(mcolors.to_rgb(c))
    return tuple(rgb * (1 - amount) + amount)


def txt(x, y, text, size=7.5, weight='normal', ha='center', va='center', color=INK, **kw):
    obj = ax.text(x, y, text, fontsize=size, fontweight=weight, ha=ha,
                  va=va, color=color, zorder=15, **kw)
    texts.append(obj)
    return obj


def rect(x, y, w, h, fc='white', ec=INK, lw=.65, z=3):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                      linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def path(points, color=INK, lw=.7, ls='-', z=5):
    ax.plot(*zip(*points), color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle='round', solid_joinstyle='round')


def arrow(points, color=INK, lw=.75, head=4.8, z=7):
    if len(points) > 2:
        path(points[:-1], color, lw, z=z)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=head, color=color, lw=lw,
                                zorder=z, shrinkA=0, shrinkB=0))


def tensor(x, y, w, h, c, rows=4, cols=4, depth=1.4, lw=.4):
    dx, dy = depth, depth * .57
    front = pale(c, .35)
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=pale(c, .70), ec=INK, lw=lw, zorder=5))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=pale(c, .14), ec=INK, lw=lw, zorder=5))
    rect(x, y, w, h, front, INK, lw, z=6)
    for j in range(1, cols):
        path([(x + w * j / cols, y), (x + w * j / cols, y + h)],
             INK, lw * .75, z=7)
    for j in range(1, rows):
        path([(x, y + h * j / rows), (x + w, y + h * j / rows)],
             INK, lw * .75, z=7)


def heading(y, letter, title):
    # No panel borders, title bands or shaded regions.
    txt(2, y, f'({letter}) {title}', 9.2, 'bold', ha='left')


def recurrent_cell(x, y, w, h, label, colour, size=8.2):
    # A boundary is used only for an actual model computation.
    rect(x, y, w, h, 'white', colour, .8)
    txt(x + w / 2, y + h / 2, label, size)


def contrast_brace(x, top, bottom):
    """A right brace joins exactly the two adjacent gain endpoints."""
    mid = (top + bottom) / 2
    verts = [(x, top),
             (x + 2.3, top), (x + 2.3, top - 1.3), (x + 2.3, top - 2.5),
             (x + 2.3, mid + 1.5), (x + 2.5, mid + .5), (x + 4.1, mid),
             (x + 2.5, mid - .5), (x + 2.3, mid - 1.5), (x + 2.3, bottom + 2.5),
             (x + 2.3, bottom + 1.3), (x + 2.3, bottom), (x, bottom)]
    codes = [MPath.MOVETO] + [MPath.CURVE4] * 12
    ax.add_patch(PathPatch(MPath(verts, codes), fc='none', ec=INK, lw=.75,
                           capstyle='round', joinstyle='round', zorder=7))
    arrow([(x + 4.3, mid), (x + 8.8, mid)], lw=.7, head=4.3)


# (a) An open three-row architecture comparison, with one shared input stream.
heading(134.7, 'a', 'Regional inputs and trained model library')
txt(17.4, 128.1, 'Meteorological forcings', 7.4, 'bold')
tensor(7.0, 110.2, 18.0, 13.3, BLUE, rows=7, cols=5, depth=1.8)
txt(16.4, 106.8, '365 days × 5 variables', 7.1)
txt(17.8, 102.2, '21 static attributes', 7.4, 'bold')
tensor(6.3, 92.4, 23.0, 6.1, GREEN, rows=3, cols=7, depth=1.5)

# Both input tensors feed each architecture; static routing is shown at the cells.
path([(26.8, 116.7), (35.0, 116.7), (35.0, 94.9), (30.8, 94.9)])
path([(35.0, 94.9), (38.0, 94.9), (38.0, 120.5)])
txt(83.6, 128.1, 'Three architectures · 64 hidden units', 7.7, 'bold')
txt(159.3, 128.1, '10 training runs per architecture', 7.7, 'bold')

for y, name, colour in [(120.5, 'B0', GOLD), (107.7, 'M1', RED), (94.9, 'EA', BLUE)]:
    arrow([(38.0, y), (54.0, y)], head=4.5)
    txt(45.5, y + 3.1, name, 8.3, 'bold', color=colour)
    txt(62.4, y, r'$[x_t,\,s]$' if name != 'EA' else r'$x_t$', 8.7)
    arrow([(70.0, y), (80.0, y)])
    label = {'B0': 'LSTM', 'M1': r'$\mathrm{LSTM}\; (\Delta f_t)$', 'EA': 'EA-LSTM'}[name]
    recurrent_cell(80.5, y - 3.1, 26.0, 6.2, label, colour, 8.0)
    if name == 'M1':
        txt(93.5, y - 5.8, 'Rank-4 forget-gate residual', 6.6, color=RED)
    if name == 'EA':
        txt(93.5, y - 5.8, r'$s\,\rightarrow\,i(s)$', 7.8, color=BLUE)
    arrow([(106.9, y), (113.4, y)])
    txt(118.2, y, r'$\widehat{Q}_t$', 10.1)
    arrow([(124.1, y), (132.5, y)])
    for j in range(10):
        tensor(134.0 + j * 5.15, y - 2.2, 3.7, 4.4, colour,
               rows=3, cols=2, depth=.75, lw=.3)

txt(136.2, 89.5, r'$\theta_1$', 7.2)
txt(159.3, 89.5, r'$\cdots$', 9)
txt(183.3, 89.5, r'$\theta_{10}$', 7.2)


# (b) A frozen historical choice feeds the two evaluation branches.
heading(84.9, 'b', 'Historical selection and dual-branch evaluation')
txt(35.5, 79.4, 'Selection · WY2000–2003', 7.9, 'bold')
txt(98.0, 79.4, 'Evaluation · WY2004–2008', 7.9, 'bold')
txt(131.0, 79.4, 'Gain over B0', 7.9, 'bold')

# Symbolic score matrix uses the same B0/candidate colours as the instance pools.
txt(17.0, 74.5, 'Historical NSE', 7.8, 'bold')
txt(15.7, 69.8, 'B0', 7.3, color=GOLD)
txt(25.5, 69.8, r'$C$', 8.1, color=PURPLE)
for label, y in [(r'$b_1$', 64.4), (r'$b_2$', 59.6), (r'$b_{107}$', 49.6)]:
    txt(4.6, y, label, 7.0)
    rect(11.5, y - 1.55, 8.0, 3.1, pale(GOLD, .35), 'none', 0)
    rect(21.5, y - 1.55, 8.0, 3.1, pale(PURPLE, .35), 'none', 0)
txt(15.5, 54.5, r'$\vdots$', 8.2)
txt(25.5, 54.5, r'$\vdots$', 8.2)
path([(10.5, 67.1), (9.3, 67.1), (9.3, 47.0), (10.5, 47.0)], lw=.6)
path([(30.6, 67.1), (31.8, 67.1), (31.8, 47.0), (30.6, 47.0)], lw=.6)
txt(19.8, 43.7, r'$C\in\{\mathrm{M1},\mathrm{EA}\}$', 7.6)

txt(52.1, 67.6, 'Catchment-wise choice', 7.3, 'bold')
txt(52.1, 63.0, r'$I_{b,s}$', 10.1)
txt(52.1, 46.6, 'Global choice', 7.7, 'bold')
txt(52.1, 42.0, r'$I_{global,s}$', 10.1)
arrow([(32.1, 56.0), (35.2, 56.0), (35.2, 63.0), (43.3, 63.0)])
arrow([(35.2, 56.0), (35.2, 42.0), (40.0, 42.0)])
ax.add_patch(Circle((35.2, 56.0), .38, fc=INK, ec='none', zorder=9))

for y, retained in [(70.0, True), (56.0, False), (42.0, False)]:
    tensor(81.1, y - 2.1, 5.5, 4.2, TEAL, rows=3, cols=3,
           depth=.8, lw=.35)
    txt(101.0, y + 4.6, 'Retained weights' if retained else 'Independent weights',
        7.4 if retained else 7.1, 'bold')
    txt(101.0, y, r'$\theta_s$' if retained else r'$\theta_t,\quad t\ne s$', 9.1)
    arrow([(114.5, y), (121.5, y)])

path([(61.0, 63.0), (72.2, 63.0)])
arrow([(72.2, 63.0), (72.2, 70.0), (79.6, 70.0)])
arrow([(72.2, 63.0), (72.2, 56.0), (79.6, 56.0)])
ax.add_patch(Circle((72.2, 63.0), .38, fc=INK, ec='none', zorder=9))
arrow([(63.6, 42.0), (79.6, 42.0)])
for y, symbol in [(70.0, r'$G_{same}$'), (56.0, r'$G_{new}$'),
                  (42.0, r'$G_{global,new}$')]:
    txt(131.0, y, symbol, 10.8)

# Pairwise braces replace a nest of boxes and intersecting endpoint arrows.
contrast_brace(143.5, 70.0, 56.0)
contrast_brace(143.5, 56.0, 42.0)
txt(170.0, 67.7, 'Retraining loss', 8.1, 'bold')
txt(170.0, 62.6, r'$D=G_{same}-G_{new}$', 9.1)
txt(170.0, 53.6, 'Local increment', 8.1, 'bold')
txt(170.0, 48.6, r'$\Delta G_{inc,new}$', 10.0)
txt(170.0, 43.6, r'$=G_{new}-G_{global,new}$', 8.4)


# (c) Paired architecture identities are grouped into disjoint instance pools.
heading(32.7, 'c', 'Selection from multiple training instances')
txt(27.7, 26.7, r'Source pool · $k=1,\ldots,5$', 7.7, 'bold')
txt(123.5, 26.7, 'Disjoint evaluation pool · 5', 7.7, 'bold')
for x0 in [4.4, 99.0]:
    txt(x0 + 1.0, 20.1, 'B0', 7.0, ha='left', color=GOLD)
    txt(x0 + 1.0, 12.8, r'$C$', 7.6, ha='left', color=PURPLE)
    for j in ([0, 1, 4] if x0 < 50 else range(5)):
        tensor(x0 + 9 + j * 7.9, 17.8, 5.3, 4.2, GOLD,
               rows=3, cols=3, depth=.8, lw=.32)
        tensor(x0 + 9 + j * 7.9, 10.5, 5.3, 4.2, PURPLE,
               rows=3, cols=3, depth=.8, lw=.32)
    if x0 < 50:
        txt(x0 + 31.4, 20.1, r'$\cdots$', 10)
        txt(x0 + 31.4, 12.8, r'$\cdots$', 10)
    path([(x0 + 8.7, 8.4), (x0 + 8.7, 6.9),
          (x0 + 47.9, 6.9), (x0 + 47.9, 8.4)], lw=.7)
    txt(x0 + 28.3, 3.9,
        r'$\{s_1,\ldots,s_k\}$' if x0 < 50 else r'$\{t_1,\ldots,t_5\}$', 8.0)

txt(75.7, 20.6, 'Mean early NSE', 7.8, 'bold')
txt(75.7, 15.3, 'Fix local / global choice', 7.0)
arrow([(54.0, 16.5), (60.4, 16.5)])
txt(172.5, 18.0, 'Mean individual', 7.6, 'bold')
txt(172.5, 13.6, 'late NSE', 7.6, 'bold')
txt(172.5, 8.0, r'$G_{new},\ G_{global,new}$', 9.0)
arrow([(149.0, 16.0), (157.5, 16.0)])
arrow([(90.2, 18.6), (94.0, 18.6), (94.0, 23.8),
       (172.5, 23.8), (172.5, 20.0)], head=8.5)


# Physical bounds and reproducible publication exports.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
extent_records, outside = [], []
for obj in texts:
    bb = obj.get_window_extent(renderer).transformed(ax.transData.inverted())
    record = {'text': obj.get_text(),
              'bounds': [float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1)]}
    extent_records.append(record)
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > WIDTH_MM or bb.y1 > HEIGHT_MM:
        outside.append(record)
assert not outside, outside
(QA / 'text_bounds.json').write_text(json.dumps(extent_records, ensure_ascii=False, indent=2),
                                    encoding='utf-8')
for suffix in ['png', 'svg', 'pdf']:
    fig.savefig(HERE / f'fig1.{suffix}', dpi=DPI)
with Image.open(HERE / 'fig1.png') as im:
    pixels = list(im.size)
    im.convert('RGB').save(HERE / 'fig1.tif', compression='tiff_lzw', dpi=(DPI, DPI))
    preview = im.convert('RGB')
    preview.thumbnail((1800, 1800))
    preview.save(QA / 'fig1_preview.png')
plt.close(fig)

sources = [ARTICLE / 'src' / 'model.py',
           ARTICLE / 'src' / 'ea_model.py']
manifest = {
    'figure': 1, 'version': 'design-4-white-open-colour-blocks',
    'width_mm': WIDTH_MM, 'height_mm': HEIGHT_MM,
    'raster_dpi': DPI, 'font': 'Times New Roman',
    'version_detail': 'Panel c: 3.8 mm terminal drop, with a separate downward arrowhead',
    'pixels': pixels, 'vector_formats': ['pdf', 'svg'],
    'no_explanatory_footer': True, 'outside_canvas_text': outside,
    'style': {'background': 'white', 'panel_frames': False, 'node_cards': False,
              'architecture_colours': {'B0': GOLD, 'M1': RED, 'EA': BLUE},
              'input_colours': {'meteorological': BLUE, 'static': GREEN},
              'candidate_colour': PURPLE, 'weight_glyph_colour': TEAL,
              'all_tensor_faces_coloured': True},
    'data_design': {'forcings': 5, 'sequence_days': 365, 'static_attributes': 21,
                    'hidden_units': 64, 'architectures': ['B0', 'M1', 'EA'],
                    'instances_per_architecture': 10, 'held_out_catchments': 107,
                    'selection': 'WY2000–2003', 'evaluation': 'WY2004–2008',
                    'source_pool_sizes': [1, 2, 3, 4, 5], 'evaluation_pool_size': 5},
    'glyphs': 'Tensor cells and score bars are structural symbols, not numerical observations.',
    'sources': {p.relative_to(ARTICLE).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sources},
    'files': {p.name: {'sha256': sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
              for p in [HERE / f'fig1.{x}' for x in ['png', 'tif', 'svg', 'pdf']]},
}
manifest['outputs'] = {ext: dict(filename=f'fig1.{ext}', **manifest['files'][f'fig1.{ext}'])
                       for ext in ['png', 'tif', 'svg', 'pdf']}
(HERE / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
print(json.dumps({'files': list(manifest['files']), 'dpi': DPI,
                  'size_mm': [WIDTH_MM, HEIGHT_MM]}, indent=2))
