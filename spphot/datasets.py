"""
spphot.datasets — the dataset swap seam (PLAN.md stage 2 / OUTLINE.md §1).

A DatasetSpec names everything that changes when the photometry source changes:
which magnitude columns feed the model (and in what order), which pair drives
the RJCE extinction proxy, whether an auxiliary photometry table (e.g. a
VVV/VIRAC2 crossmatch) overrides survey bands, and whether a survey-indicator
feature is appended so the model can absorb residual system differences.

Everything else in the pipeline is band-count-agnostic: spphot.data assembles
the photometry block from a spec, spphot.v2 slices features through a
FeatureLayout, and every checkpoint records its spec (name + band list) so
apply-time feature assembly always matches training — resolve_dataset() maps
old checkpoints (which stored only label_cols) back to a spec.

Adding a dataset = adding one REGISTRY entry. The intended VVV entry is
sketched below; it needs the VIRAC2 crossmatch parquet + the crowding-threshold
study before it becomes real (PLAN.md "open decisions").
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AuxPhot:
    """Auxiliary photometry joined onto the sample (e.g. VVV/VIRAC2 JHKs).

    columns maps band-in-spec -> column-in-aux-parquet. The aux value REPLACES
    the survey value where finite; the survey_ind feature is set to 1 only where
    every overridden band came from the aux table (so the model sees a clean
    0/1 system flag, not a per-band patchwork)."""
    join_key: str = "sdss_id"
    columns: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    bands: tuple[str, ...]                     # ordered mag columns == feature order
    rjce_pair: tuple[str, str] | None = None   # (H, W2) columns for RJCE A_Ks; None disables
    aux_phot: AuxPhot | None = None
    survey_indicator: bool = False             # append 0/1 'survey_ind' to the phot block

    @property
    def n_phot(self) -> int:
        """Width of the photometry block in the feature matrix."""
        return len(self.bands) + (1 if self.survey_indicator else 0)

    @property
    def phot_cols(self) -> tuple[str, ...]:
        """Metadata columns the photometry block is read from, in feature order."""
        return self.bands + (("survey_ind",) if self.survey_indicator else ())

    def rjce_indices(self):
        """(i_H, i_W2) positions of the RJCE pair in the phot block, or None."""
        if self.rjce_pair is None:
            return None
        return self.bands.index(self.rjce_pair[0]), self.bands.index(self.rjce_pair[1])


@dataclass(frozen=True)
class FeatureLayout:
    """Layout of a model feature matrix [phot(n_phot) | A_rjce? | spec].

    Plain frozen ints, closed over statically by the jitted v2 forward — one
    compile per layout, exactly like the old module-level N_PHOT constants."""
    n_phot: int
    has_aks: bool = False

    @property
    def i_aks(self) -> int:
        """Column index of the A_Ks feature (only meaningful if has_aks)."""
        return self.n_phot

    @property
    def spec_start(self) -> int:
        return self.n_phot + (1 if self.has_aks else 0)


REGISTRY = {
    # the Hogg+18 / DR17 baseline: Gaia DR3 + 2MASS + WISE, RJCE from H-W2
    "dr17": DatasetSpec(
        name="dr17",
        bands=("g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag",
               "w1_mag", "w2_mag"),
        rjce_pair=("h_mag", "w2_mag"),
    ),
    # PLAN.md stage 2 — enable once the VIRAC2 crossmatch parquet exists:
    # "dr17-virac2": DatasetSpec(
    #     name="dr17-virac2",
    #     bands=("g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag",
    #            "w1_mag", "w2_mag"),
    #     rjce_pair=("h_mag", "w2_mag"),
    #     aux_phot=AuxPhot(join_key="sdss_id",
    #                      columns=(("j_mag", "virac_j"), ("h_mag", "virac_h"),
    #                               ("k_mag", "virac_ks"))),
    #     survey_indicator=True,
    # ),
}


def get_dataset(name_or_spec):
    """Registry lookup that also accepts an already-built DatasetSpec."""
    if isinstance(name_or_spec, DatasetSpec):
        return name_or_spec
    try:
        return REGISTRY[name_or_spec]
    except KeyError:
        raise SystemExit(f"unknown dataset {name_or_spec!r}; "
                         f"available: {sorted(REGISTRY)}")


def resolve_dataset(name, label_cols):
    """Checkpoint back-compat: prefer the saved dataset name; else match the
    saved label_cols against the registry (old checkpoints stored only
    label_cols); else wrap them in an ad-hoc 'legacy' spec so apply-time
    feature assembly still works."""
    if name and name in REGISTRY:
        return REGISTRY[name]
    lc = tuple(str(c) for c in label_cols)
    for ds in REGISTRY.values():
        if ds.bands == lc:
            return ds
    return DatasetSpec(name="legacy", bands=lc)
