# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Callable

from mypy.nodes import (
    CallExpr,
    Decorator,
    MemberExpr,
    TypeInfo,
)
from mypy.options import Options
from mypy.plugin import (
    ClassDefContext,
    Plugin,
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


# ------------------------------------------------------------------------------------ #
#                                   PLUGIN DEFINITION                                  #
# ------------------------------------------------------------------------------------ #


class DataframelyPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)

    def get_base_class_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        # Given a class, check whether it is a subclass of `dy.Schema`. If so, mark
        # all methods decorated with `@rule` as staticmethods.
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):
            if any(base.fullname == SCHEMA_FULLNAME for base in sym.node.mro):
                return mark_rules_as_staticmethod
        return None


def plugin(version: str) -> type[Plugin]:
    return DataframelyPlugin
