import numpy as np
import pytest

from pyxtal import pyxtal


def make_mixed_xtal():
    xtal = pyxtal(random_state=0)
    xtal.from_random(
        dim=3,
        group=131,
        species=["C"],
        numIons=[6],
        sites=[["2e", "4l"]],
        random_state=0,
    )

    assert xtal.valid
    assert len(xtal.atom_sites) == 2

    # These are target labels attached to independent Wyckoff orbits.
    xtal.set_orbit_coordination([4, 3])
    return xtal


def multiplicity_counts(xtal):
    return xtal.get_orbit_coordination_counts()


def test_copy_preserves_coordination():
    xtal = make_mixed_xtal()
    copied = xtal.copy()

    assert copied.get_orbit_coordination() == [4, 3]
    assert multiplicity_counts(copied) == multiplicity_counts(xtal)


def test_identity_transform_preserves_coordination():
    xtal = make_mixed_xtal()
    before = multiplicity_counts(xtal)

    xtal.transform(np.eye(3))

    assert xtal.get_orbit_coordination() == [4, 3]
    assert multiplicity_counts(xtal) == before


def test_json_roundtrip_preserves_coordination(tmp_path):
    xtal = make_mixed_xtal()
    filename = tmp_path / "mixed.json"

    xtal.to_json(filename)

    loaded = pyxtal()
    loaded.from_json(filename)

    assert loaded.get_orbit_coordination() == [4, 3]
    assert multiplicity_counts(loaded) == multiplicity_counts(xtal)


def test_t_subgroup_preserves_coordination():
    xtal = make_mixed_xtal()
    parent_counts = multiplicity_counts(xtal)

    child = None
    for _ in range(50):
        child = xtal.subgroup_once(
            eps=0.0,
            group_type="t",
            mut_lat=False,
        )
        if child is not None:
            break

    if child is None:
        pytest.skip("No valid t-subgroup found for this structure.")

    assert child.has_orbit_coordination()
    assert all(
        site.coordination in (3, 4)
        for site in child.atom_sites
    )
    assert multiplicity_counts(child) == parent_counts


def test_k_subgroup_coordination_scales():
    xtal = make_mixed_xtal()
    parent_counts = multiplicity_counts(xtal)
    parent_atoms = sum(xtal.numIons)

    child = None
    for _ in range(50):
        child = xtal.subgroup_once(
            eps=0.0,
            group_type="k",
            max_cell=4,
            mut_lat=False,
        )
        if child is not None:
            break

    if child is None:
        pytest.skip("No valid k-subgroup found for this structure.")

    multiplier = sum(child.numIons) / parent_atoms
    expected = {
        cn: int(round(count * multiplier))
        for cn, count in parent_counts.items()
    }

    assert child.has_orbit_coordination()
    assert multiplicity_counts(child) == expected

