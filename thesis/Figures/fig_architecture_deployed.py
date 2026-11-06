#!/usr/bin/env python3
"""Fig. 4.1 (deployed system): architecture and information flow.

Derived from fig1(mass_only=True) of ros2_ws_HJH/src/paper/figs/fig_gen.py
(2026-11-06). Differences from that figure, which drew the cohort
configuration:
  * the MHE is the reduced velocity--mass estimator, driven by T_phys only;
  * release is accepted by the force-balance detector (m_acc = T/f_z, arm D),
    which reads the IMU specific force; the estimator's gamma is not used;
  * the NMPC parameter vector shows the residual specific force d_hat.
Output: fig1_architecture_deployed.pdf (+ .png) next to this script.
Usage:  python3 fig_architecture_deployed.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
COL2 = 7.0
FONT = 'Tinos'
BLUE, ORANGE, AQUA = '#1769AA', '#B85A18', '#087F70'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [FONT, 'Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 240, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
})


def main():
    fig, ax = plt.subplots(figsize=(COL2, 2.4), facecolor='white')
    ax.set_xlim(0, 100); ax.set_ylim(1.2, 40); ax.axis('off')

    NAVY = '#18324A'
    BLUE_D, BLUE_M, BLUE_L = BLUE, '#8FB5D5', '#EAF3FA'
    TEAL_D, TEAL_L = AQUA, '#EDF8F5'
    ORANGE_D, ORANGE_L = ORANGE, '#FFF3EA'
    RED_D = '#B74343'
    GREY, PALE, SLATE = '#627487', '#F8FAFC', '#718096'
    SERIF = [FONT, 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
    boxed_text = []

    def txt(x, y, s, size=7.0, color=NAVY, weight='normal', ha='left', va='center',
            style='normal', z=5):
        return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                       fontfamily=SERIF, ha=ha, va=va, style=style, zorder=z)

    def arrow(p1, p2, color=NAVY, lw=1.1, rad=0.0, z=3):
        a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=8.5,
                            connectionstyle=f'arc3,rad={rad}', color=color, lw=lw,
                            shrinkA=1.2, shrinkB=1.2, zorder=z)
        ax.add_patch(a)
        return a

    def block(x, y, w, h, title, lines, edge=BLUE_D, fill='white', title_size=7.0,
              strong=False):
        body = FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0.10,rounding_size=0.45',
                              fc=fill, ec=edge, lw=1.35 if strong else 0.9, zorder=2)
        ax.add_patch(body)
        ax.add_patch(plt.Rectangle((x, y), 0.65, h, fc=edge, ec='none', zorder=2.2))
        title_artist = txt(x + w / 2 + 0.25, y + h - 1.9, title, title_size, edge,
                           'bold', ha='center')
        boxed_text.append((title_artist, body, 1.1))
        if lines:
            if len(lines) == 1:
                ys = [y + h * 0.32]
            elif len(lines) == 2:
                ys = [y + h * 0.46, y + h * 0.20]
            else:
                ys = np.linspace(y + h * 0.55, y + h * 0.18, len(lines))
            for yy, (value, size, color, weight) in zip(ys, lines):
                artist = txt(x + w / 2 + 0.25, yy, value, size, color, weight, ha='center')
                boxed_text.append((artist, body, 1.0))
        return body

    # Deployment boundary.
    outer = FancyBboxPatch((3.2, 8.1), 74.3, 29.5,
                           boxstyle='round,pad=0.12,rounding_size=0.75',
                           fc=PALE, ec=BLUE_M, lw=0.8, ls=(0, (3.0, 2.4)), zorder=0)
    ax.add_patch(outer)
    txt(46.5, 36.45, 'ROS 2 OUTER LOOP', 6.0, BLUE_D, 'bold', ha='center',
        z=1).set_bbox(dict(fc=PALE, ec='none', pad=0.7))

    # Estimation side: reduced MHE, collective thrust only.
    block(6.0, 11.6, 20.0, 6.2, 'ROTOR THRUST',
          [(r'$T_{phys}=k_f\sum\omega_i^2$', 6.0, GREY, 'normal')],
          edge=TEAL_D, fill='white', title_size=6.35)
    block(6.0, 23.0, 20.0, 9.0, 'ONLINE MHE',
          [(r'states $v,m$; attitude as input', 5.9, GREY, 'normal'),
           (r'estimate $m\;\rightarrow\;\Delta J=\mu(m)\,r_z^2$', 6.15, NAVY, 'normal')],
          edge=BLUE_D, fill='white', title_size=6.15)

    # The contract.
    block(33.0, 22.0, 19.0, 11.0, 'ATOMIC FRAME',
          [(r'$\mathcal{P}=\{m,\Delta J,\gamma,h,a\}$', 6.55, NAVY, 'bold'),
           ('atomic timestamp', 5.75, GREY, 'normal')],
          edge=BLUE_D, fill=BLUE_L, title_size=5.9, strong=True)

    # Two consumers of the frame.
    block(59.0, 28.2, 15.5, 8.2, 'NMPC',
          [(r'$p=[x_{ref},m,\Delta J,\hat d\,]$', 5.65, NAVY, 'normal'),
           (r'state $x\leftarrow$ odometry', 5.45, GREY, 'normal')],
          edge=BLUE_D, fill='white', title_size=6.45)
    block(59.0, 18.6, 15.5, 7.3, 'RATE SCHEDULER',
          [(r'$\Delta J\;\rightarrow\;$ rate scale', 5.75, NAVY, 'normal')],
          edge=BLUE_D, fill='white', title_size=5.85)

    # Release path: independent of the frame.
    block(33.0, 10.5, 19.0, 6.2, 'FORCE BALANCE',
          [(r'$m_{acc}=\bar T/\bar f_z\;\rightarrow\;\gamma_{acc}$', 5.9, NAVY, 'normal')],
          edge=SLATE, fill='white', title_size=5.7)
    block(55.5, 10.5, 19.0, 6.2, 'RELEASE LIFECYCLE',
          [(r'accept $\cdot$ UNRESOLVED brake', 5.6, GREY, 'normal')],
          edge=SLATE, fill='white', title_size=5.7)

    # Execution outside ROS 2.
    block(82.0, 24.0, 15.0, 8.5, 'PX4 RATE LOOP',
          [('fixed gains · 1 kHz', 5.9, GREY, 'normal')],
          edge=ORANGE_D, fill=ORANGE_L, title_size=5.9)
    block(82.0, 7.0, 15.0, 10.0, 'QUADROTOR',
          [('rigid payload', 6.1, NAVY, 'bold'),
           ('Gazebo · DART', 5.8, GREY, 'normal')],
          edge=TEAL_D, fill=TEAL_L, title_size=6.8)

    # Estimation flow.
    arrow((16.0, 17.8), (16.0, 23.0), TEAL_D, 1.0)
    txt(18.0, 20.3, r'$T_{phys}$', 5.8, TEAL_D, style='italic')
    arrow((26.0, 27.5), (33.0, 27.5), BLUE_D, 1.35)
    txt(29.5, 29.0, 'qualified', 5.45, BLUE_D, 'bold', ha='center').set_bbox(
        dict(fc=PALE, ec='none', pad=0.25))
    arrow((52.0, 29.4), (59.0, 32.0), BLUE_D, 1.1)
    txt(58.2, 34.4, 'model update', 5.25, BLUE_D, ha='right', style='italic')
    arrow((52.0, 25.4), (59.0, 22.25), BLUE_D, 1.1)
    txt(55.2, 21.5, 'rate scale', 5.25, BLUE_D, ha='center', style='italic')

    # Release flow: detector -> lifecycle.
    arrow((52.0, 13.6), (55.5, 13.6), SLATE, 0.95)
    txt(53.75, 15.4, r'$\gamma_{acc}$', 5.3, GREY, ha='center', style='italic')

    # Commands to the inner loop and the vehicle.
    arrow((74.5, 32.2), (82.0, 29.8), ORANGE_D, 1.05)
    txt(79.9, 32.3, '$T$', 6.0, ORANGE_D, ha='center')
    arrow((74.5, 22.25), (82.0, 26.6), ORANGE_D, 1.05)
    txt(79.6, 22.4, r'$\omega^\star$', 6.0, ORANGE_D, ha='center')
    arrow((89.5, 24.0), (89.5, 17.0), ORANGE_D, 1.05)
    txt(91.0, 20.5, 'motor cmd', 5.6, ORANGE_D, style='italic')

    # Measured feedback: rotor speeds + odometry to the estimator; rotor speeds +
    # IMU specific force to the detector.
    ax.plot([89.5, 89.5, 16.0], [7.0, 5.1, 5.1], color=TEAL_D, lw=0.9, zorder=1)
    arrow((16.0, 5.1), (16.0, 11.6), TEAL_D, 0.9, z=2)
    arrow((42.5, 5.1), (42.5, 10.5), SLATE, 0.9, z=2)
    txt(41.6, 6.55, r'$\omega_i,\,f_z$', 5.3, GREY, ha='right', style='italic')
    txt(51.5, 2.85, r'adaptive path: motor speeds $\omega_i$ + odometry $x$ only'
        r'$\quad\cdot\quad$release evidence: $\omega_i$ + IMU $f_z$',
        6.0, TEAL_D, 'bold', ha='center')

    # The forbidden input terminates before reaching the deployed graph.
    txt(6.2, 39.0, 'attach / drop notification', 5.9, GREY)
    ax.plot([23.9, 23.9], [38.7, 35.2], color=RED_D, lw=0.8,
            ls=(0, (2.2, 2.2)), zorder=2)
    ax.add_patch(Circle((23.9, 34.35), 0.78, fc='white', ec=RED_D, lw=1.0, zorder=4))
    ax.plot([23.40, 24.40], [33.85, 34.85], color=RED_D, lw=1.2, zorder=5)
    txt(25.3, 34.35, 'NOT CONSUMED', 5.85, RED_D, 'bold')

    # Generation-time guard: no text may escape its module.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, patch, margin_px in boxed_text:
        tb = artist.get_window_extent(renderer=renderer)
        pb = patch.get_window_extent(renderer=renderer)
        if (tb.x0 < pb.x0 + margin_px or tb.x1 > pb.x1 - margin_px or
                tb.y0 < pb.y0 + margin_px or tb.y1 > pb.y1 - margin_px):
            raise RuntimeError(f'text escaped module: {artist.get_text()!r}')

    stem = os.path.join(HERE, 'fig1_architecture_deployed')
    fig.savefig(stem + '.png', dpi=240, facecolor='white')
    fig.savefig(stem + '.pdf')
    plt.close(fig)
    print('wrote', stem + '.pdf')


if __name__ == '__main__':
    main()
