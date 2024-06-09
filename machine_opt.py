from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
from numerics import bisect, clamp

@dataclass
class Material:
    # N/mm^2 (same as MPa)
    ultimate_tensile_strength: float

# TODO: Add common materials
#   * 6061 aluminum
#   * soft, medium, and hard woods
#   * some mild steels
#   * cast iron
#   * 1215 steel
#   * other free machining steels
#   * 303 stainless
#   * A1 tool steel (not hardened)

@dataclass
class Tool:
    # mm
    tool_diameter: float

    # count (I guess the SI is moles? lol)
    number_of_flutes: int

    # unitless, [0, 1]
    tool_wear_factor: float = 1.1

    # unitless, [0, 1]
    machine_efficancy: float = 0.8



def chip_thining_factor(woc, tool_diameter):
    return np.sqrt(1 - (2*woc / tool_diameter)**2)


@dataclass
class OptimizerSettings:
    # These are set by the user but they may be supersceeded
    # by the calculated properties below
    min_axial_doc: float
    max_axial_doc: float
    min_radial_doc: float
    max_radial_doc: float
    max_feed_per_tooth: float
    min_feed_per_tooth: float
    min_chip_cross_sectional_area: float
    max_spindle_power: float
    max_cutting_force: float

    @property
    def axial_doc(self):
        range = self.max_axial_doc - self.min_axial_doc
        range = np.arange(self.min_axial_doc, self.max_axial_doc, range / 100)
        return range[:, None, None]

    @property
    def radial_doc(self):
        range = self.max_radial_doc - self.min_radial_doc
        range = np.arange(self.min_radial_doc, self.max_radial_doc, range / 100)
        return range[None, :, None]

    @property
    def feed_per_tooth(self):
        range = self.max_feed_per_tooth - self.min_feed_per_tooth
        range = np.arange(self.min_feed_per_tooth, self.max_feed_per_tooth, range / 100)
        return range[None, None, :]

@dataclass
class Recipe:
    tool: Tool

    # mm
    feed_per_tooth: np.ndarray

    # m/min
    surface_speed: float

    # mm
    radial_doc: np.ndarray

    # mm
    axial_doc: np.ndarray

    # mm/min
    @property
    def feed_rate(self) -> np.ndarray:
        return self.feed_per_tooth * self.rpm * self.tool.number_of_flutes

    # 1/min
    @property
    def rpm(self) -> float:
        return 1000 * self.surface_speed /( np.pi * self.tool.tool_diameter)

    # cm^3/min
    @property
    def mrr(self) -> np.ndarray:
        return self.feed_rate * self.axial_doc * self.radial_doc / 1000

    @property
    def chip_thining_factor(self) -> np.ndarray:
        return chip_thining_factor(self.radial_doc, self.tool.tool_diameter)

    # mm^2
    # This *does* account for chip thining
    @property
    def chip_cross_sectional_area(self) -> np.ndarray:
        return self.axial_doc * self.feed_per_tooth

    # unitless
    @property
    def doc_ratio(self):
        return self.radial_doc / self.axial_doc

@dataclass
class Optimizer:
    material: Material
    recipe: Recipe
    settings: OptimizerSettings

    @property
    def tool(self):
        return self.recipe.tool

    def avg_engaged_flutes(self, recipe=None):
        dia = self.tool.tool_diameter
        flutes = self.tool.number_of_flutes

        if recipe is None:
            recipe = self.recipe
            print(dia, flutes, recipe.radial_doc)

        rdoc = recipe.radial_doc
        out = flutes * (np.arcsin((2 * rdoc - dia) / dia) + np.arcsin(1)) / 2/ np.pi
        return out

    @property
    def cutting_force(self):
        sigma = self.material.ultimate_tensile_strength
        flutes = self.avg_engaged_flutes()
        wear = self.tool.tool_wear_factor
        area = self.recipe.chip_cross_sectional_area
        return  sigma * flutes * wear * area

    @property
    def score(self):
        mrr = self.recipe.mrr
        force = self.cutting_force
        cond = force > self.settings.max_cutting_force
        return ma.masked_array(mrr, mask=cond)

    def print_recipe(self, recipe, force):
        print("axial doc: ", recipe.axial_doc)
        print("radial doc: ", recipe.radial_doc)
        print("spindle rpm: ", recipe.rpm)
        print("feed_rate: ", recipe.feed_rate)
        print("surface speed: ", recipe.surface_speed)
        print("feed_per_tooth", recipe.feed_per_tooth)
        print("mrr: ", recipe.mrr)
        print("cross sectional: ", recipe.chip_cross_sectional_area)
        print("number of flutes: ", self.avg_engaged_flutes(recipe))
        print("force: ", force)

    def compute_best(self):
        score = self.score
        idx = np.argmax(score)
        idx = np.unravel_index(idx, score.shape)
        adoc_idx = idx[0]
        rdoc_idx = idx[1]
        fpt_idx = idx[2]
        feed_per_tooth = self.recipe.feed_per_tooth[0, 0, fpt_idx]
        adoc = self.recipe.axial_doc[adoc_idx, 0, 0]
        rdoc = self.recipe.radial_doc[0, rdoc_idx, 0]
        out = Recipe(tool=self.tool, axial_doc=adoc, radial_doc=rdoc, feed_per_tooth=feed_per_tooth, surface_speed=self.recipe.surface_speed)
        self.print_recipe(out, self.cutting_force[idx])
        return out

    @property
    def torque_at_cutter(self):
        raise NotImplementedError("todo")

    @property
    def spindle_power(self):
        raise NotImplementedError("todo")

Aluminum6061 = Material(210.0)
Steel1215 = Material(540.0)

my_tool = Tool(tool_diameter=6.35, number_of_flutes=3)

my_settings = OptimizerSettings(
    min_axial_doc = 0.5,
    max_axial_doc = 1.5 * my_tool.tool_diameter,
    min_radial_doc = 0.2,
    max_radial_doc = 0.5 * my_tool.tool_diameter,
    max_feed_per_tooth = 0.1,
    min_feed_per_tooth = 0.01,
    min_chip_cross_sectional_area = 0.02,
    max_spindle_power = 1.5,
    max_cutting_force = 17.0,)

my_recipe = Recipe(
    tool = my_tool,
    axial_doc=my_settings.axial_doc,
    radial_doc=my_settings.radial_doc,
    surface_speed=300,
    feed_per_tooth=my_settings.feed_per_tooth)

my_state = Optimizer(Aluminum6061, my_recipe, my_settings)
print(my_state.compute_best())
