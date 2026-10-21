from .ft import NODE_CLASS_MAPPINGS as ftn, NODE_DISPLAY_NAME_MAPPINGS as ftnames
from .fluid import NODE_CLASS_MAPPINGS as fluidn, NODE_DISPLAY_NAME_MAPPINGS as fluidnames

NODE_CLASS_MAPPINGS = { **ftn, **fluidn }
NODE_DISPLAY_NAME_MAPPINGS = { **ftnames, **fluidnames }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"