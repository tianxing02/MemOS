from memos.configs.mem_os import MOSConfig
from memos.mem_os.core import MOSCore


class MOS(MOSCore):
    """
    The MOS (Memory Operating System) class inherits from MOSCore.
    This class maintains backward compatibility with the original MOS interface.
    """

    def __init__(self, config: MOSConfig):
        super().__init__(config)
