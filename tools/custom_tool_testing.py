"""Run restricted custom-tool code in a separate Python process for testing."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


_MAX_SOURCE_CHARS = 32_000
_MAX_ARGUMENTS_BYTES = 32_000
_TEST_TIMEOUT_SECONDS = 2.0

_WORKER = r'''
import ast
import contextlib
import io
import json
import math
import sys

MAX_RANGE_ITEMS = 10_000
MAX_SEQUENCE_ITEMS = 100_000
MAX_STRING_CHARS = 64_000
MAX_RESULT_BYTES = 64_000
MAX_NUMBER = 10 ** 100
MAX_PRINT_BYTES = 32_000


class CodeRejected(ValueError):
    pass


class BoundedOutput(io.TextIOBase):
    def __init__(self):
        self.parts = []
        self.size = 0

    def write(self, text):
        size = len(text.encode("utf-8"))
        if self.size + size > MAX_PRINT_BYTES:
            raise ValueError("Printed output exceeded the test limit.")
        self.parts.append(text)
        self.size += size
        return len(text)

    def getvalue(self):
        return "".join(self.parts)


def _check_number(value):
    if type(value) not in (int, float) or abs(value) > MAX_NUMBER:
        raise ValueError("Numeric result exceeded the supported range.")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Numeric result exceeded the supported range.")
    return value


def _safe_add(left, right):
    if type(left) is type(right) and type(left) in (str, list, tuple):
        result_size = len(left) + len(right)
        limit = MAX_STRING_CHARS if type(left) is str else MAX_SEQUENCE_ITEMS
        if result_size > limit:
            raise ValueError("Combined value exceeded the supported size.")
        return left + right
    return _check_number(left + right)


def _safe_subtract(left, right):
    return _check_number(left - right)


def _safe_multiply(left, right):
    if type(left) in (str, list, tuple) and type(right) is int:
        sequence, count = left, right
    elif type(right) in (str, list, tuple) and type(left) is int:
        sequence, count = right, left
    else:
        return _check_number(left * right)
    if count < 0:
        count = 0
    limit = MAX_STRING_CHARS if type(sequence) is str else MAX_SEQUENCE_ITEMS
    if len(sequence) * count > limit:
        raise ValueError("Repeated value exceeded the supported size.")
    return sequence * count


def _safe_divide(left, right):
    return _check_number(left / right)


def _safe_floor_divide(left, right):
    return _check_number(left // right)


def _safe_modulo(left, right):
    if type(left) is str or type(right) is str:
        raise ValueError("String formatting with '%' is not supported.")
    return _check_number(left % right)


def _safe_power(left, right):
    if type(right) not in (int, float) or abs(right) > 100:
        raise ValueError("Exponent must be between -100 and 100.")
    return _check_number(left ** right)


def _bounded_range(*values):
    result = range(*values)
    if len(result) > MAX_RANGE_ITEMS:
        raise ValueError("range() is limited to 10,000 items.")
    return result


class SafeCodeValidator(ast.NodeVisitor):
    _allowed_nodes = {
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
        ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign,
        ast.If, ast.For, ast.Expr, ast.Pass, ast.Break, ast.Continue,
        ast.Name, ast.Load, ast.Store, ast.Constant, ast.Dict, ast.List,
        ast.Tuple, ast.Set, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Call, ast.keyword, ast.Subscript, ast.Slice, ast.IfExp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    }

    def __init__(self, function_name):
        self.function_name = function_name
        self.function_seen = False

    def visit(self, node):
        if type(node) not in self._allowed_nodes:
            raise CodeRejected(f"{type(node).__name__} is not allowed in custom tool tests.")
        return super().visit(node)

    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.FunctionDef):
            raise CodeRejected("Define exactly one top-level function.")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if (
            self.function_seen
            or node.name != self.function_name
            or node.name.startswith("_")
            or node.decorator_list
            or node.args.posonlyargs
            or [argument.arg for argument in node.args.args] != ["arguments"]
            or node.args.vararg
            or node.args.kwarg
            or node.args.kwonlyargs
            or node.args.defaults
            or node.args.kw_defaults
        ):
            raise CodeRejected(
                f"Define '{self.function_name}(arguments)' without decorators or default arguments."
            )
        self.function_seen = True
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("_"):
            raise CodeRejected("Names beginning with '_' are not allowed.")

    def visit_Constant(self, node):
        if type(node.value) not in (type(None), bool, int, float, str):
            raise CodeRejected("Only JSON-compatible literal values are allowed.")
        if type(node.value) is int and abs(node.value) > MAX_NUMBER:
            raise CodeRejected("Numeric literals must be finite and within the supported range.")
        if type(node.value) is float and (
            not math.isfinite(node.value) or abs(node.value) > MAX_NUMBER
        ):
            raise CodeRejected("Numeric literals must be finite and within the supported range.")
        if type(node.value) is str and len(node.value) > MAX_STRING_CHARS:
            raise CodeRejected("String literals exceeded the supported size.")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise CodeRejected("Only direct calls to safe built-ins are allowed.")
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            raise CodeRejected("Starred arguments are not allowed.")
        if any(keyword.arg is None for keyword in node.keywords):
            raise CodeRejected("Expanded keyword arguments are not allowed.")
        self.generic_visit(node)

    def visit_Assign(self, node):
        if any(not isinstance(target, ast.Name) for target in node.targets):
            raise CodeRejected("Assignments may only target local names.")
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise CodeRejected("Assignments may only target local names.")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise CodeRejected("Assignments may only target local names.")
        self.generic_visit(node)

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            raise CodeRejected("Loops may only assign to a local name.")
        self.generic_visit(node)


class SafeOperationTransformer(ast.NodeTransformer):
    _operations = {
        ast.Add: "_safe_add",
        ast.Sub: "_safe_subtract",
        ast.Mult: "_safe_multiply",
        ast.Div: "_safe_divide",
        ast.FloorDiv: "_safe_floor_divide",
        ast.Mod: "_safe_modulo",
        ast.Pow: "_safe_power",
    }

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        operation = self._operations.get(type(node.op))
        if operation is None:
            raise CodeRejected(f"{type(node.op).__name__} is not an allowed operation.")
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=operation, ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            ),
            node,
        )

    def visit_AugAssign(self, node):
        operation = self._operations.get(type(node.op))
        if operation is None:
            raise CodeRejected(f"{type(node.op).__name__} is not an allowed operation.")
        return ast.copy_location(
            ast.Assign(
                targets=[node.target],
                value=ast.Call(
                    func=ast.Name(id=operation, ctx=ast.Load()),
                    args=[ast.Name(id=node.target.id, ctx=ast.Load()), node.value],
                    keywords=[],
                ),
            ),
            node,
        )


_safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "object": object,
    "print": print,
    "range": _bounded_range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_safe_operations = {
    "_safe_add": _safe_add,
    "_safe_subtract": _safe_subtract,
    "_safe_multiply": _safe_multiply,
    "_safe_divide": _safe_divide,
    "_safe_floor_divide": _safe_floor_divide,
    "_safe_modulo": _safe_modulo,
    "_safe_power": _safe_power,
}


def main():
    payload = json.loads(sys.stdin.read())
    source = payload["source"]
    function_name = payload["function_name"]
    arguments = payload["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("Test arguments must be a JSON object.")

    tree = ast.parse(source, mode="exec")
    SafeCodeValidator(function_name).visit(tree)
    tree = SafeOperationTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = {
        "__builtins__": _safe_builtins,
        **_safe_operations,
    }
    output_stream = BoundedOutput()
    with contextlib.redirect_stdout(output_stream):
        exec(compile(tree, "<custom-tool-test>", "exec"), namespace, namespace)
        result = namespace[function_name](arguments)
    result_json = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(result_json.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("Returned result exceeded the 64 KB test limit.")
    return {"ok": True, "result": result, "stdout": output_stream.getvalue()}


try:
    response = main()
except SyntaxError as error:
    response = {
        "ok": False,
        "error": f"SyntaxError on line {error.lineno}: {error.msg}",
        "stdout": "",
    }
except Exception as error:
    message = str(error).replace("\r", " ").replace("\n", " ")[:2_000]
    response = {
        "ok": False,
        "error": f"{type(error).__name__}: {message}",
        "stdout": "",
    }
sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False))
'''


@dataclass(frozen=True)
class CustomToolTestResult:
    success: bool
    output: Any = None
    stdout: str = ""
    error: str | None = None


def test_custom_tool(
    source: str,
    function_name: str,
    arguments: dict[str, Any],
) -> CustomToolTestResult:
    """Run a custom function in a restricted subprocess with bounded input and time."""
    if not isinstance(source, str) or not source.strip():
        return CustomToolTestResult(success=False, error="Custom tool source must not be empty.")
    if len(source) > _MAX_SOURCE_CHARS:
        return CustomToolTestResult(
            success=False,
            error=f"Custom tool source exceeds {_MAX_SOURCE_CHARS:,} characters.",
        )
    if not isinstance(arguments, dict):
        return CustomToolTestResult(success=False, error="Test arguments must be a JSON object.")

    try:
        payload = json.dumps(
            {"source": source, "function_name": function_name, "arguments": arguments},
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as error:
        return CustomToolTestResult(success=False, error=f"Test arguments are not valid JSON: {error}")
    if len(payload.encode("utf-8")) > _MAX_ARGUMENTS_BYTES:
        return CustomToolTestResult(
            success=False,
            error=f"Source and test arguments together exceed {_MAX_ARGUMENTS_BYTES:,} bytes.",
        )

    environment = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    for variable in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if value := os.environ.get(variable):
            environment[variable] = value
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _WORKER],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TEST_TIMEOUT_SECONDS,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return CustomToolTestResult(
            success=False,
            error=f"Custom tool test exceeded the {_TEST_TIMEOUT_SECONDS:g}-second limit.",
        )
    except OSError as error:
        return CustomToolTestResult(
            success=False,
            error=f"Could not start the isolated custom tool test process: {error}",
        )

    if process.returncode != 0:
        detail = process.stderr.strip()[:2_000]
        return CustomToolTestResult(
            success=False,
            error=f"Custom tool test process exited with status {process.returncode}."
            + (f" {detail}" if detail else ""),
        )
    try:
        response = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        return CustomToolTestResult(
            success=False,
            error=f"Custom tool test process returned invalid output: {error}",
        )
    if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
        return CustomToolTestResult(
            success=False,
            error="Custom tool test process returned an invalid response.",
        )
    if not response["ok"]:
        return CustomToolTestResult(
            success=False,
            stdout=str(response.get("stdout", "")),
            error=str(response.get("error", "Custom tool test failed.")),
        )
    return CustomToolTestResult(
        success=True,
        output=response.get("result"),
        stdout=str(response.get("stdout", "")),
    )
