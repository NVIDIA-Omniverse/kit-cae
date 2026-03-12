from omni.cae.schema import viz as cae_viz
from omni.kit.widget.stage import StageIcons
from omni.kit.widget.stage.delegates import NameColumnDelegate


class CaeNameColumnDelegate(NameColumnDelegate):

    # Overrides a private method; may stop working without notice!
    def _NameColumnDelegate__get_all_icons_to_draw(self, item, item_is_native):
        icons = super()._NameColumnDelegate__get_all_icons_to_draw(item, item_is_native)
        prim = item.prim
        if prim and prim.HasAPI(cae_viz.OperatorAPI):
            icons.append(StageIcons().get("CaeVizOperator"))
        return icons
