#!/usr/bin/env python

from .configuration_eo1 import EO1Config


def __getattr__(name: str):
    if name == "EO1Policy":
        from .modeling_eo1 import EO1Policy

        return EO1Policy
    if name == "make_eo1_pre_post_processors":
        from .processor_eo1 import make_eo1_pre_post_processors

        return make_eo1_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["EO1Config", "EO1Policy", "make_eo1_pre_post_processors"]
