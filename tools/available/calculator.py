"""Bounded arithmetic calculator; never evaluates Python source."""

import ast
import operator
from decimal import Decimal, DecimalException
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tools.base_tool import BaseTool


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=512)


class _ArithmeticEvaluator(ast.NodeVisitor):
    _binary_operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    _unary_operations = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(self, expression: str) -> Decimal:
        try:
            parsed = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(parsed)) > 64:
                raise ValueError("Expression is too complex")
            result = self.visit(parsed.body)
        except (SyntaxError, DecimalException, ZeroDivisionError, OverflowError) as error:
            raise ValueError("Invalid arithmetic expression") from error
        if not result.is_finite():
            raise ValueError("Result must be finite")
        if abs(result.adjusted()) > 1000 or len(result.as_tuple().digits) > 1000:
            raise ValueError("Result exceeds the supported numeric range")
        return result

    def visit_Constant(self, node: ast.Constant) -> Decimal:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        result = Decimal(str(node.value))
        if not result.is_finite():
            raise ValueError("Numeric constants must be finite")
        return result

    def visit_BinOp(self, node: ast.BinOp) -> Decimal:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or abs(right) > 100:
                raise ValueError("Exponent must be an integer between -100 and 100")
            result = left**int(right)
        else:
            operation = self._binary_operations.get(type(node.op))
            if operation is None:
                raise ValueError("Operator is not allowed")
            result = operation(left, right)
        if not result.is_finite() or abs(result.adjusted()) > 1000:
            raise ValueError("Result exceeds the supported numeric range")
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Decimal:
        operation = self._unary_operations.get(type(node.op))
        if operation is None:
            raise ValueError("Unary operator is not allowed")
        return operation(self.visit(node.operand))

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Expression element is not allowed: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a bounded arithmetic expression using numeric operators only."
    input_model = CalculatorInput
    timeout_seconds = 1.0

    def execute(self, arguments: BaseModel) -> dict[str, str]:
        parsed = CalculatorInput.model_validate(arguments)
        value = _ArithmeticEvaluator().evaluate(parsed.expression)
        result = format(value.normalize(), "f")
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return {"result": result or "0"}