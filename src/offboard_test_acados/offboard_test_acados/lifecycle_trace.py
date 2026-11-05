"""Single-line lifecycle event records for the experiment scorer (plan §13.1).

Every node that owns a lifecycle transition prints exactly one line of the form

    [EVT] name=<event> wall=<unix seconds> key=value key=value ...

through its ROS logger.  ``scripts/gripper/score_lifecycle.py`` parses these
lines; the ROS log header timestamp is not used, so the record stays correct if
the logger clock ever changes.  Values must not contain spaces (they are
replaced with ``_``) so that the format stays trivially splittable.

Logging only: nothing here reads or changes controller/estimator state.

Event vocabulary (who emits, meaning):
  proximity  attach          physical DetachableJoint created
  proximity  attach_offset   physical box - drone offset at attach (rx, ry, rz)
  proximity  sep             physical DetachableJoint removed  -> t_sep
  nmpc       cmd             release command issued            -> t_cmd
  nmpc       inject_loss     uncommanded physical release injected (no command)
  mhe        declare         payload-state transition (state=EMPTY|LOADED, src)  -> t_declare
  mhe        release_geom    estimator released payload geometry (src)
  mhe        moment_ref      loaded first-moment reference formed and frozen (s_ref)
  mhe        cmd_armed_open  command-armed baseline opened its confirmation window
  mhe        cmd_armed_confirm  command-armed baseline confirmed release
  nmpc       confirm         controller-side persistence on no-payload confidence met -> t_confirm
  nmpc       accept          controller switched to the empty target (src) -> t_accept
  nmpc       unresolved      release not evidenced before timeout; model kept
  nmpc       brake           stop-to-hover reference armed (armed=0|1)
  nmpc       resolved        late evidence arrived after UNRESOLVED
"""
import time

TAG = '[EVT]'


def _fmt(v):
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v).replace(' ', '_')


def evt_line(name, **fields):
    parts = [TAG, f'name={name}', f'wall={time.time():.3f}']
    parts += [f'{k}={_fmt(v)}' for k, v in fields.items()]
    return ' '.join(parts)
