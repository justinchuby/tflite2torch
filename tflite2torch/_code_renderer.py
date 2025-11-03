"""
Torch FX graph rendering to PyTorch code.

This module renders a PyTorch FX graph into readable PyTorch code that can
be executed independently.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.fx import GraphModule, Node


class CodeRenderer:
    """
    Renders PyTorch FX graph as executable PyTorch code.

    This class takes a FX GraphModule and generates readable Python code
    that implements the same computation.
    """

    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "

    def render(self, graph_module: GraphModule, class_name: str = "ConvertedModel") -> str:
        """
        Render FX GraphModule as PyTorch code.

        Args:
            graph_module: FX GraphModule to render
            class_name: Name for the generated model class

        Returns:
            String containing the generated PyTorch code
        """
        lines = []
        
        # Add imports
        lines.append("import torch")
        lines.append("import torch.nn as nn")
        lines.append("import torch.nn.functional as F")
        lines.append("")
        lines.append("")
        
        # Start class definition
        lines.append(f"class {class_name}(nn.Module):")
        self.indent_level = 1
        
        # Add docstring
        lines.append(self._indent("\"\"\""))
        lines.append(self._indent("Converted TFLite model to PyTorch."))
        lines.append(self._indent(""))
        lines.append(self._indent("This model was automatically generated from a TFLite model."))
        lines.append(self._indent("\"\"\""))
        lines.append("")
        
        # Add __init__ method
        lines.extend(self._render_init(graph_module))
        lines.append("")
        
        # Add forward method
        lines.extend(self._render_forward(graph_module))
        
        self.indent_level = 0
        
        return "\n".join(lines)

    def _render_init(self, graph_module: GraphModule) -> List[str]:
        """Render __init__ method."""
        lines = []
        lines.append(self._indent("def __init__(self):"))
        self.indent_level += 1
        lines.append(self._indent("super().__init__()"))
        lines.append("")
        
        # Extract modules and parameters from the graph module
        modules_added = set()
        
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                module_name = node.target
                if module_name not in modules_added:
                    module = graph_module.get_submodule(module_name)
                    module_type = type(module).__name__
                    
                    # Generate initialization code
                    init_str = self._get_module_init_string(module)
                    lines.append(self._indent(f"self.{module_name} = {init_str}"))
                    modules_added.add(module_name)
            
            elif node.op == "get_attr":
                param_name = node.target
                if param_name not in modules_added:
                    try:
                        param = graph_module.get_parameter(param_name)
                        shape_str = "x".join(str(s) for s in param.shape)
                        lines.append(
                            self._indent(
                                f"self.{param_name} = nn.Parameter("
                                f"torch.randn({list(param.shape)}))"
                            )
                        )
                        modules_added.add(param_name)
                    except AttributeError:
                        # Might be a buffer or other attribute
                        pass
        
        self.indent_level -= 1
        return lines

    def _render_forward(self, graph_module: GraphModule) -> List[str]:
        """Render forward method."""
        lines = []
        
        # Collect input placeholders
        input_names = []
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                input_names.append(node.name)
        
        # Generate method signature
        if len(input_names) == 0:
            signature = "def forward(self):"
        elif len(input_names) == 1:
            signature = f"def forward(self, {input_names[0]}):"
        else:
            inputs_str = ", ".join(input_names)
            signature = f"def forward(self, {inputs_str}):"
        
        lines.append(self._indent(signature))
        self.indent_level += 1
        
        # Generate code for each node
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                # Skip placeholders, they're in the signature
                continue
            elif node.op == "get_attr":
                # Parameters are accessed as self.param_name
                continue
            elif node.op == "call_module":
                code = self._render_call_module(node)
                if code:
                    lines.append(self._indent(code))
            elif node.op == "call_function":
                code = self._render_call_function(node)
                if code:
                    lines.append(self._indent(code))
            elif node.op == "call_method":
                code = self._render_call_method(node)
                if code:
                    lines.append(self._indent(code))
            elif node.op == "output":
                code = self._render_output(node)
                if code:
                    lines.append(self._indent(code))
        
        self.indent_level -= 1
        return lines

    def _render_call_module(self, node: Node) -> str:
        """Render a call_module node."""
        module_name = node.target
        args_str = self._format_args(node.args)
        return f"{node.name} = self.{module_name}({args_str})"

    def _render_call_function(self, node: Node) -> str:
        """Render a call_function node."""
        func = node.target
        
        # Get function name
        if hasattr(func, "__name__"):
            func_name = func.__name__
        else:
            func_name = str(func)
        
        # Handle torch functions
        if hasattr(torch, func_name):
            func_str = f"torch.{func_name}"
        elif hasattr(torch.nn.functional, func_name):
            func_str = f"F.{func_name}"
        else:
            func_str = func_name
        
        args_str = self._format_args(node.args)
        kwargs_str = self._format_kwargs(node.kwargs)
        
        if kwargs_str:
            return f"{node.name} = {func_str}({args_str}, {kwargs_str})"
        else:
            return f"{node.name} = {func_str}({args_str})"

    def _render_call_method(self, node: Node) -> str:
        """Render a call_method node."""
        if not node.args:
            return ""
        
        obj = node.args[0]
        method_name = node.target
        args_str = self._format_args(node.args[1:])
        
        obj_name = obj.name if isinstance(obj, Node) else str(obj)
        return f"{node.name} = {obj_name}.{method_name}({args_str})"

    def _render_output(self, node: Node) -> str:
        """Render an output node."""
        if not node.args:
            return "return None"
        
        output = node.args[0]
        if isinstance(output, (list, tuple)):
            outputs_str = ", ".join(
                item.name if isinstance(item, Node) else str(item)
                for item in output
            )
            return f"return ({outputs_str})"
        elif isinstance(output, Node):
            return f"return {output.name}"
        else:
            return f"return {output}"

    def _format_args(self, args) -> str:
        """Format arguments for code generation."""
        if not args:
            return ""
        
        formatted_args = []
        for arg in args:
            if isinstance(arg, Node):
                if arg.op == "get_attr":
                    formatted_args.append(f"self.{arg.target}")
                else:
                    formatted_args.append(arg.name)
            elif isinstance(arg, (list, tuple)):
                inner = ", ".join(
                    item.name if isinstance(item, Node) else repr(item)
                    for item in arg
                )
                formatted_args.append(f"[{inner}]")
            else:
                formatted_args.append(repr(arg))
        
        return ", ".join(formatted_args)

    def _format_kwargs(self, kwargs: Dict) -> str:
        """Format keyword arguments for code generation."""
        if not kwargs:
            return ""
        
        formatted_kwargs = []
        for key, value in kwargs.items():
            if isinstance(value, Node):
                if value.op == "get_attr":
                    formatted_kwargs.append(f"{key}=self.{value.target}")
                else:
                    formatted_kwargs.append(f"{key}={value.name}")
            else:
                formatted_kwargs.append(f"{key}={repr(value)}")
        
        return ", ".join(formatted_kwargs)

    def _get_module_init_string(self, module: nn.Module) -> str:
        """Generate initialization string for a module."""
        module_type = type(module).__name__
        
        # Handle common module types
        if isinstance(module, nn.Conv2d):
            return (
                f"nn.Conv2d("
                f"in_channels={module.in_channels}, "
                f"out_channels={module.out_channels}, "
                f"kernel_size={module.kernel_size}, "
                f"stride={module.stride}, "
                f"padding={module.padding})"
            )
        elif isinstance(module, nn.Linear):
            return (
                f"nn.Linear("
                f"in_features={module.in_features}, "
                f"out_features={module.out_features})"
            )
        elif isinstance(module, nn.BatchNorm2d):
            return f"nn.BatchNorm2d(num_features={module.num_features})"
        elif isinstance(module, nn.MaxPool2d):
            return (
                f"nn.MaxPool2d("
                f"kernel_size={module.kernel_size}, "
                f"stride={module.stride}, "
                f"padding={module.padding})"
            )
        elif isinstance(module, nn.AvgPool2d):
            return (
                f"nn.AvgPool2d("
                f"kernel_size={module.kernel_size}, "
                f"stride={module.stride}, "
                f"padding={module.padding})"
            )
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid)):
            return f"nn.{module_type}()"
        elif isinstance(module, nn.Softmax):
            return f"nn.Softmax(dim={module.dim})"
        else:
            # Generic fallback
            return f"nn.{module_type}()"

    def _indent(self, line: str) -> str:
        """Add indentation to a line."""
        return self.indent_str * self.indent_level + line

    def save_to_file(self, code: str, filepath: str):
        """
        Save generated code to a file.

        Args:
            code: Generated Python code
            filepath: Path where to save the file
        """
        with open(filepath, "w") as f:
            f.write(code)
