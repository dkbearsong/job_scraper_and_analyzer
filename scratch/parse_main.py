import ast
import json

with open('main.py', 'r') as f:
    tree = ast.parse(f.read())

funcs = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
        funcs.append({"name": node.name, "start": node.lineno, "end": node.end_lineno})

funcs.sort(key=lambda x: x['start'])
for f in funcs:
    print(f"{f['name']}: {f['start']} - {f['end']} ({f['end'] - f['start']} lines)")
