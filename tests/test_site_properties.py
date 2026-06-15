import numpy as np
import pytest

from pyxtal import pyxtal
from pyxtal.wyckoff_site import atom_site


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
    xtal.set_orbit_coordination([4, 3])
    return xtal


def target_counts(xtal):
    return xtal.get_orbit_coordination_counts()


def test_target_and_actual_coordination_are_independent():
    xtal = make_mixed_xtal()
    site = xtal.atom_sites[0]

    assert site.target_coordination == 4
    assert site.coordination is None

    site.set_coordination(7)
    assert site.coordination == 7
    assert site.target_coordination == 4

    site.set_target_coordination(5)
    assert site.coordination == 7
    assert site.target_coordination == 5


def test_generic_site_property_is_deep_copied():
    xtal = make_mixed_xtal()
    site = xtal.atom_sites[0]
    site.set_property("lego", {"motif": ["tetrahedral"]})

    copied = xtal.copy()
    copied.atom_sites[0].property["lego"]["motif"].append("changed")

    assert site.property["lego"] == {"motif": ["tetrahedral"]}
    assert copied.atom_sites[0].property["lego"] == {
        "motif": ["tetrahedral", "changed"]
    }


def test_identity_transform_preserves_properties():
    xtal = make_mixed_xtal()
    xtal.atom_sites[0].set_property("tag", "sp3")
    xtal.atom_sites[1].set_property("tag", "sp2")

    xtal.transform(np.eye(3))

    assert xtal.get_orbit_coordination() == [4, 3]
    assert [site.get_property("tag") for site in xtal.atom_sites] == ["sp3", "sp2"]


def test_json_roundtrip_preserves_properties(tmp_path):
    xtal = make_mixed_xtal()
    xtal.atom_sites[0].set_property("tag", {"hybridization": "sp3"})
    xtal.atom_sites[0].set_coordination(6)

    filename = tmp_path / "mixed.json"
    xtal.to_json(filename)

    loaded = pyxtal()
    loaded.from_json(filename)

    assert loaded.get_orbit_coordination() == [4, 3]
    assert loaded.atom_sites[0].coordination == 6
    assert loaded.atom_sites[0].get_property("tag") == {"hybridization": "sp3"}


def test_legacy_coordination_dictionary_migrates_to_target_property():
    xtal = make_mixed_xtal()
    saved = xtal.atom_sites[0].save_dict()
    saved.pop("property")
    saved["coordination"] = 4

    loaded = atom_site.load_dict(saved)

    assert loaded.coordination is None
    assert loaded.target_coordination == 4


def test_t_subgroup_preserves_properties():
    xtal = make_mixed_xtal()
    xtal.atom_sites[0].set_property("tag", "sp3")
    xtal.atom_sites[1].set_property("tag", "sp2")
    parent_counts = target_counts(xtal)

    child = None
    for _ in range(50):
        child = xtal.subgroup_once(eps=0.0, group_type="t", mut_lat=False)
        if child is not None:
            break
    if child is None:
        pytest.skip("No valid t-subgroup found for this structure.")

    assert child.has_orbit_coordination()
    assert all(site.target_coordination in (3, 4) for site in child.atom_sites)
    assert all(site.get_property("tag") in ("sp2", "sp3") for site in child.atom_sites)
    assert target_counts(child) == parent_counts


def test_k_subgroup_target_counts_scale():
    xtal = make_mixed_xtal()
    parent_counts = target_counts(xtal)
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
        coordination: int(round(count * multiplier))
        for coordination, count in parent_counts.items()
    }
    assert child.has_orbit_coordination()
    assert target_counts(child) == expected

