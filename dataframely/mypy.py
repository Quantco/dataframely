# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Callable
from typing import cast

from mypy.checker import TypeChecker
from mypy.nodes import (
    AssignmentStmt,
    CallExpr,
    Decorator,
    DictExpr,
    Expression,
    ListExpr,
    MemberExpr,
    NameExpr,
    StrExpr,
    TupleExpr,
    TypeInfo,
    Var,
)
from mypy.options import Options
from mypy.plugin import (
    ClassDefContext,
    MethodContext,
    MethodSigContext,
    Plugin,
    SemanticAnalyzerPluginInterface,
)
from mypy.types import (
    AnyType,
    CallableType,
    FunctionLike,
    Instance,
    LiteralType,
    TupleType,
    Type,
    TypedDictType,
    TypeOfAny,
    UnionType,
)

COLLECTION_FULLNAME = "dataframely.collection.Collection"
COLUMN_PACKAGE = "dataframely.column"
RULE_DECORATOR_FULLNAME = "dataframely._rule.rule"
SCHEMA_FULLNAME = "dataframely.schema.Schema"
TYPED_DATAFRAME_FULLNAME = "dataframely._typing.DataFrame"
TYPED_LAZYFRAME_FULLNAME = "dataframely._typing.LazyFrame"

# --------------------------------------- RULES -------------------------------------- #


def mark_rules_as_staticmethod(ctx: ClassDefContext) -> None:
    """Mark all methods decorated with `@rule` as `staticmethod`s."""
    info = ctx.cls.info
    for sym in info.names.values():
        if not isinstance(sym.node, Decorator):
            continue
        decorator = sym.node.original_decorators[0]
        if not isinstance(decorator, CallExpr):
            continue
        if not isinstance(decorator.callee, MemberExpr):
            continue
        if decorator.callee.fullname == RULE_DECORATOR_FULLNAME:
            sym.node.func.is_static = True


# -------------------------------- FILTER RETURN TYPE -------------------------------- #


def alter_collection_filter_return_type(ctx: MethodSigContext) -> FunctionLike:
    """Alter the return type for `dy.Collection.filter` to a `TypedDict` for the failure
    info."""
    signature = ctx.default_signature
    if not isinstance(ctx.api, TypeChecker) or not isinstance(ctx.type, CallableType):
        return signature

    # Get the type that the method is associated with and iterate over the annotations
    base_class = cast(TypeInfo, ctx.type.type_object())
    required_keys: set[str] = set()
    optional_keys: set[str] = set()
    for name, node in base_class.names.items():
        if isinstance(node.node, Var):
            if isinstance(node.node.type, Instance):
                if node.node.type.type.fullname == TYPED_LAZYFRAME_FULLNAME:
                    required_keys.add(name)
            elif isinstance(node.node.type, UnionType):
                optional_keys.add(name)

    # For the filter method, we also want to use a typed dictionary for the failure
    # info dictionary that we return
    if not isinstance(signature.ret_type, TupleType):
        return signature

    second_tuple_item = signature.ret_type.items[1]
    if not isinstance(second_tuple_item, Instance):
        return signature

    failure_info_type = second_tuple_item.args[1]
    return_typed_dict = TypedDictType(
        {k: failure_info_type for k in (required_keys | optional_keys)},
        required_keys=required_keys,
        readonly_keys=required_keys | optional_keys,
        fallback=ctx.api.named_generic_type("typing._TypedDict", []),
    )
    return signature.copy_modified(
        ret_type=signature.ret_type.copy_modified(
            items=[signature.ret_type.items[0], return_typed_dict]
        )
    )


# ------------------------------- ITER ROWS RETURN TYPE ------------------------------ #


def _convert_dy_column_to_dtype(
    api: SemanticAnalyzerPluginInterface,
    column_type: str,
    column_args: list[Expression],
) -> Type:
    """Convert a `dataframely.column` type to a regular Python type."""
    if column_type.startswith("Int") or column_type.startswith("UInt"):
        return api.named_type("builtins.int")
    if column_type.startswith("Float"):
        return api.named_type("builtins.float")
    if column_type == "Decimal":
        return api.named_type("decimal.Decimal")
    if column_type == "Bool":
        return api.named_type("builtins.bool")
    if column_type == "String":
        return api.named_type("builtins.str")
    if column_type == "Date":
        return api.named_type("datetime.date")
    if column_type == "Time":
        return api.named_type("datetime.time")
    if column_type == "Datetime":
        return api.named_type("datetime.datetime")
    if column_type == "Duration":
        return api.named_type("datetime.timedelta")
    if column_type == "Enum":
        if (
            len(column_args) == 1
            and isinstance(column_args[0], ListExpr)
            and len(column_args[0].items) > 0
        ):
            return UnionType(
                items=[
                    LiteralType(expr.value, fallback=api.named_type("builtins.str"))
                    for expr in column_args[0].items
                    if isinstance(expr, StrExpr)
                ]
            )
        return api.named_type("builtins.str")
    if column_type == "Struct":
        if (
            len(column_args) == 1
            and isinstance(column_args[0], DictExpr)
            and len(column_args[0].items) > 0
        ):
            fields = {
                key.value: _convert_dy_column_to_dtype(api, node.callee.name, node.args)
                for key, node in column_args[0].items
                if isinstance(key, StrExpr)
                and isinstance(node, CallExpr)
                and isinstance(node.callee, MemberExpr | NameExpr)
            }
            return TypedDictType(
                items=fields,
                required_keys=set(fields.keys()),
                readonly_keys=set(fields.keys()),
                fallback=api.named_type("typing._TypedDict", []),
            )
        return api.named_type(
            "builtins.dict",
            [api.named_type("builtins.str"), AnyType(TypeOfAny.unannotated)],
        )
    if column_type == "List":
        if (
            len(column_args) == 1
            and isinstance(column_args[0], CallExpr)
            and isinstance(column_args[0].callee, MemberExpr | NameExpr)
        ):
            return api.named_type(
                "builtins.list",
                [
                    _convert_dy_column_to_dtype(
                        api, column_args[0].callee.name, column_args[0].args
                    )
                ],
            )
        return api.named_type("builtins.list")
    if column_type == "Array":
        if isinstance(column_args[0], CallExpr) and isinstance(
            column_args[0].callee, MemberExpr | NameExpr
        ):
            inner_type = _convert_dy_column_to_dtype(
                api,
                column_args[0].callee.name,
                column_args[0].args,
            )
            # If the array has more than one dimension, return a list of lists of the inner type.
            if len(column_args) > 1 and isinstance(column_args[1], TupleExpr):
                for _ in range(len(column_args[1].items) - 1):
                    inner_type = api.named_type(
                        "builtins.list",
                        [inner_type],
                    )
            return api.named_type(
                "builtins.list",
                [inner_type],
            )
        return api.named_type("builtins.list")
    if column_type == "Any" or column_type == "Object":
        return AnyType(TypeOfAny.explicit)
    # If we can't infer the type, we default to `Any`.
    # This is, for example, the case for self-defined types, e.g., via `functools.partial`.
    return AnyType(TypeOfAny.from_error)


def store_typed_dict_type_for_schema(
    ctx: ClassDefContext,
    schema_registry: dict[str, TypedDictType],
) -> None:
    """Add `TypedDictType` inferred from the schema's columns to a given registry."""

    schema_type = ctx.cls.info

    def _get_field_types_and_names(
        schema_type: TypeInfo,
        include_parents: bool = True,
    ) -> tuple[list[Type], list[str]]:
        field_types: list[Type] = []
        field_names: list[str] = []

        if include_parents:
            # Get field types and names for parent schemas.
            # The first item in the mro list is the schema type itself.
            for base in schema_type.mro[1:]:
                if any(
                    base_parent.fullname == SCHEMA_FULLNAME
                    for base_parent in base.mro[1:]
                ):
                    if base.fullname in schema_registry:
                        # If the parent schema has already been analyzed, we can use the
                        # cached TypedDictType.
                        base_typed_dict = schema_registry[base.fullname]
                        field_types.extend(base_typed_dict.items.values())
                        field_names.extend(base_typed_dict.items.keys())
                    else:
                        # This should not happen. But in case we end up not having the base type
                        # analyzed yet, we visit the parent class to get the field types and names.
                        # Since mro is a flat list of all the base classes,
                        # we don't need to recursively visit the parent classes again.
                        base_field_types, base_field_names = _get_field_types_and_names(
                            base, include_parents=False
                        )
                        field_types.extend(base_field_types)
                        field_names.extend(base_field_names)

        # Get the field types and names from the schema definition.
        # We need to parse the AST nodes to extract the exact column types, i.e.,
        # to include parameters passed to columns. This way, we can properly analyze column types
        # such as `dy.List(dy.Struct({"a": dy.Integer()}))`.
        ast_available = len(schema_type.defn.defs.body) > 0
        if ast_available:
            for node in schema_type.defn.defs.body:
                if (
                    isinstance(node, AssignmentStmt)
                    and isinstance(node.rvalue, CallExpr)
                    and isinstance(node.rvalue.callee, NameExpr | MemberExpr)
                    and len(node.lvalues) >= 1
                    and isinstance(node.lvalues[0], NameExpr)
                ):
                    field_types.append(
                        _convert_dy_column_to_dtype(
                            ctx.api,
                            node.rvalue.callee.name,
                            node.rvalue.args,
                        )
                    )
                    field_names.append(node.lvalues[0].name)
        else:
            # This should not happen, but in case the AST is not available,
            # e.g., if the type is defined in another module that has not been analyzed yet,
            # we can still infer limited type information.
            for name, sym in schema_type.names.items():
                if isinstance(sym.node, Var):
                    field_names.append(name)
                    field_types.append(
                        _convert_dy_column_to_dtype(
                            ctx.api,
                            cast(Instance, sym.node.type).type.name,
                            [],  # We don't have access to the arguments here.
                        )
                    )

        return field_types, field_names

    field_types, field_names = _get_field_types_and_names(
        schema_type=schema_type,
        include_parents=True,
    )

    # Generate a typed dictionary type with the column names and types.
    typed_dict_type = TypedDictType(
        items=dict(zip(field_names, field_types)),
        required_keys=set(field_names),
        readonly_keys=set(field_names),
        fallback=ctx.api.named_type("typing._TypedDict", []),
        line=ctx.cls.line,
        column=ctx.cls.column,
    )

    schema_registry[schema_type.fullname] = typed_dict_type


def alter_dataframe_iter_rows_return_type(
    ctx: MethodContext,
    schema_registry: dict[str, TypedDictType],
) -> Type:
    """Alter the return type for `dy.DataFrame.iter_rows` to a `TypedDict` if
    `named=True`."""
    api = cast(TypeChecker, ctx.api)

    # Check if the method has been called with the `named=True` argument.
    if not any(
        arg_name == "named"
        and arg_type == LiteralType(True, fallback=api.named_type("builtins.bool"))
        for arg_names, arg_types in zip(ctx.arg_names, ctx.arg_types)
        for arg_name, arg_type in zip(arg_names, arg_types)
    ):
        return ctx.default_return_type

    # Extract the `Schema` from an `dataframely._typing.DataFrame[Schema]`.
    if (
        not isinstance(ctx.type, Instance)
        or len((wrapper_type := cast(Instance, ctx.type)).args) != 1
        or not isinstance(wrapper_type.args[0], Instance)
    ):
        return ctx.default_return_type

    # Get the associated schema for the dataframe.
    schema_type = cast(Instance, wrapper_type.args[0]).type

    # Lookup the `TypedDictType` in the registry.
    if schema_type.fullname not in schema_registry:
        return ctx.default_return_type
    typed_dict_type = schema_registry[schema_type.fullname]

    # Return an iterator of the typed dictionary type (we're iterating the rows of the dataframe).
    return api.named_generic_type("typing.Iterator", [typed_dict_type])


# ------------------------------------------------------------------------------------ #
#                                   PLUGIN DEFINITION                                  #
# ------------------------------------------------------------------------------------ #


class DataframelyPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)
        self.schema_registry: dict[str, TypedDictType] = {}

    def get_base_class_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        # Given a class, check whether it is a subclass of `dy.Schema`. If so, mark
        # all methods decorated with `@rule` as staticmethods.
        # Also, store the `TypedDictType` for the schema in a registry to allow downstream
        # hooks (e.g., the type checking pass) to make use of them.
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):
            if any(base.fullname == SCHEMA_FULLNAME for base in sym.node.mro):

                def _hook(ctx: ClassDefContext) -> None:
                    mark_rules_as_staticmethod(ctx)
                    store_typed_dict_type_for_schema(
                        ctx,
                        self.schema_registry,
                    )

                return _hook
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        # Given a method call called "iter_rows", check whether it is called
        # from a subclass of `dataframely._typing.DataFrame`.
        # If so, alter the return type to a `TypedDict` if `named=True` is passed to the method.
        # The information about the schema is stored in a registry that is shared between hooks.
        parts = fullname.split(".")
        if parts[-1] == "iter_rows":
            sym = self.lookup_fully_qualified(".".join(parts[:-1]))
            if sym and isinstance(sym.node, TypeInfo):
                if any(
                    base.fullname == TYPED_DATAFRAME_FULLNAME for base in sym.node.mro
                ):

                    def _hook(ctx: MethodContext) -> Type:
                        return alter_dataframe_iter_rows_return_type(
                            ctx,
                            self.schema_registry,
                        )

                    return _hook
        return None

    def get_method_signature_hook(
        self, fullname: str
    ) -> Callable[[MethodSigContext], FunctionLike] | None:
        # Given a method call called "filter", check whether it is called
        # from a subclass of `dy.Collection`. If so, alter the failure dict return type
        # to provide a typed dict.
        parts = fullname.split(".")
        if parts[-1] == "filter":
            sym = self.lookup_fully_qualified(".".join(parts[:-1]))
            if sym and isinstance(sym.node, TypeInfo):
                if any(base.fullname == COLLECTION_FULLNAME for base in sym.node.mro):
                    return alter_collection_filter_return_type
        return None


def plugin(version: str) -> type[Plugin]:
    return DataframelyPlugin
