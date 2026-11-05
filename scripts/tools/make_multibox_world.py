#!/usr/bin/env python3
"""Replace the single <model name="box"> in gripper_test.sdf with N small boxes (2026-09-15).

Partial-loss experiment: four 0.08 m cubes in a 2x2 grid centred on (1.0, 0.0),
centre spacing 0.09 m (0.01 m gap), names box, box2, box3, box4, each with link
"box_link" so proximity_gripper_node's child_link stays unchanged.
Usage: make_multibox_world.py SRC DST NBOX MASS_EACH
"""
import re
import sys

SIZE = 0.08
PITCH = 0.09


def box_model(name, x, y, mass):
    i = mass * SIZE * SIZE / 6.0
    z = SIZE / 2.0
    return f"""    <model name="{name}">
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="box_link">
        <inertial>
          <mass>{mass}</mass>
          <inertia>
            <ixx>{i:.8f}</ixx><ixy>0</ixy><ixz>0</ixz>
            <iyy>{i:.8f}</iyy><iyz>0</iyz><izz>{i:.8f}</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry><box><size>{SIZE} {SIZE} {SIZE}</size></box></geometry>
          <material><ambient>0.8 0.2 0.1 1</ambient><diffuse>0.8 0.2 0.1 1</diffuse></material>
        </visual>
        <collision name="collision">
          <geometry><box><size>{SIZE} {SIZE} {SIZE}</size></box></geometry>
        </collision>
      </link>
    </model>
"""


def main():
    src, dst, nbox, mass = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    text = open(src).read()
    pat = re.compile(r'    <model name="box">.*?</model>\n', re.S)
    if len(pat.findall(text)) != 1:
        raise SystemExit('expected exactly one <model name="box"> block')
    h = PITCH / 2.0
    grid = [(1.0 - h, -h), (1.0 + h, -h), (1.0 - h, +h), (1.0 + h, +h)]
    if nbox != 4:
        raise SystemExit('only NBOX=4 layout is defined')
    names = ['box', 'box2', 'box3', 'box4']
    models = ''.join(box_model(nm, x, y, mass) for nm, (x, y) in zip(names, grid))
    open(dst, 'w').write(pat.sub(lambda _: models, text))
    print(f'wrote {dst}: ' + ', '.join(f'{nm}@({x:.3f},{y:.3f})' for nm, (x, y) in zip(names, grid)))


if __name__ == '__main__':
    main()
