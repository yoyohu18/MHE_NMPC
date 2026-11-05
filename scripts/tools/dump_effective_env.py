#!/usr/bin/env python3
"""记录运行时代码里每个环境变量开关的**生效值**(显式设置 or 代码默认)。

record_experiment_provenance.sh 的 experiment.env 只抓显式设置的变量;代码默认值
不会落盘,于是"改默认值"前后的批次在 experiment.env 里外观相同(2026-09-15 窗口
力平衡种子改默认开时暴露,见实验计划 §12.2 配置断代)。本脚本静态扫描(ast,
不 import,无副作用)运行时包里的 os.environ.get(NAME, DEFAULT) / os.getenv,
对照当前环境输出每个变量的生效值与来源。

输出 TSV 列: name  effective  source  code_default  location
  source=explicit  环境里设置了该变量(含空串),effective 为环境值
  source=default   未设置,effective 为代码里的字面量默认值
  source=default-expr  未设置且代码默认不是字面量(如 `get('MHE_N') else 20`),
                   effective 记为 <unset>,需查 location 处代码
同一变量多处读取且字面量默认不一致时,每处各出一行,便于发现口径分裂。

注意:只反映本脚本执行时的环境;启动脚本若在调用 provenance **之后**才 export
变量,节点实际环境会不同(MHE 节点启动日志另打 [effective-env] 行作最终凭据)。
"""
import ast
import os
import sys
from pathlib import Path


def _env_name_and_default(call):
    """匹配 os.environ.get(...) / os.getenv(...);返回 (name, default_node) 或 None。"""
    f = call.func
    if not isinstance(f, ast.Attribute):
        return None
    is_environ_get = (f.attr == 'get' and isinstance(f.value, ast.Attribute)
                      and f.value.attr == 'environ')
    is_getenv = (f.attr == 'getenv' and isinstance(f.value, ast.Name)
                 and f.value.id == 'os')
    if not (is_environ_get or is_getenv):
        return None
    if not call.args or not isinstance(call.args[0], ast.Constant) \
            or not isinstance(call.args[0].value, str):
        return None
    default = call.args[1] if len(call.args) > 1 else None
    return call.args[0].value, default


def scan(root):
    rows = []
    for path in sorted(Path(root).rglob('*.py')):
        if any(part in ('install', 'build', '__pycache__') for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            hit = _env_name_and_default(node)
            if hit is None:
                continue
            name, default = hit
            if isinstance(default, ast.Constant):
                code_default = repr(default.value)
                literal = str(default.value) if default.value is not None else None
            else:
                code_default = '<expr>' if default is not None else '<none>'
                literal = None
            if name in os.environ:
                effective, source = os.environ[name], 'explicit'
            elif literal is not None:
                effective, source = literal, 'default'
            else:
                effective, source = '<unset>', 'default-expr'
            loc = f'{path.relative_to(root)}:{node.lineno}'
            rows.append((name, effective, source, code_default, loc))
    # 去重同名同默认同来源的多处读取,只保留首处位置
    seen, out = set(), []
    for r in sorted(rows):
        key = r[:4]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'offboard_test_acados', 'offboard_test_acados')
    root = os.path.normpath(root)
    print('name\teffective\tsource\tcode_default\tlocation')
    for r in scan(root):
        print('\t'.join(x.replace('\t', ' ').replace('\n', ' ') for x in r))


if __name__ == '__main__':
    main()
