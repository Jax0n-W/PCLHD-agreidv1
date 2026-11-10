"""Canonical AG-ReID.v1 identity parsing.

AG-ReID identity is the concatenation of the P, T and A fields. Some local
training copies rewrite ``A{a}C{c}`` as one ``C{a}{c}`` token; this module
handles both forms without collapsing identities that share only P.
"""
from __future__ import absolute_import

import os.path as osp
import re
from collections import namedtuple


AGREID_TRAIN_IDENTITIES = 199
AGREID_TEST_IDENTITIES = 189

AGReIDName = namedtuple(
    "AGReIDName", ("pid", "camid", "person", "tracklet", "angle", "source"))

_OFFICIAL_RE = re.compile(
    r"^P(?P<p>\d+)_?T(?P<t>\d+)_?A(?P<a>\d+)_?C(?P<c>\d+)",
    re.IGNORECASE)
_REWRITTEN_RE = re.compile(
    r"^P(?P<p>\d+)_T(?P<t>\d+)_C(?P<ac>\d+)_F\d+",
    re.IGNORECASE)


def compose_agreid_pid(person, tracklet, angle):
    """Return the official integer label formed by concatenating P/T/A."""
    fields = (str(person), str(tracklet), str(angle))
    if any(not field.isdigit() for field in fields):
        raise ValueError("AG-ReID P/T/A fields must contain digits: {}".format(
            fields))
    return int("".join(fields))


def parse_agreid_name(filename):
    """Parse an official or locally rewritten AG-ReID image filename.

    Official: ``P0001T04041A0C0F121.jpg``
    Rewritten: ``P0001_T04041_C00_F121.jpg`` where C00 stores A=0, C=0.

    For rewritten training files ``camid`` deliberately remains the complete
    encoded C token (for example C10 -> 10), preserving the historical camera
    metadata used by training samplers. Only the identity label is corrected.
    """
    name = osp.basename(str(filename))
    match = _OFFICIAL_RE.match(name)
    if match is not None:
        person = match.group("p")
        tracklet = match.group("t")
        angle = match.group("a")
        return AGReIDName(
            compose_agreid_pid(person, tracklet, angle),
            int(match.group("c")), person, tracklet, angle, "official")

    match = _REWRITTEN_RE.match(name)
    if match is not None:
        encoded_camera = match.group("ac")
        if len(encoded_camera) < 2:
            raise ValueError(
                "Rewritten AG-ReID C token must encode A and C: {}".format(
                    name))
        person = match.group("p")
        tracklet = match.group("t")
        angle = encoded_camera[0]
        return AGReIDName(
            compose_agreid_pid(person, tracklet, angle),
            int(encoded_camera), person, tracklet, angle, "rewritten")

    raise ValueError("Cannot parse AG-ReID P/T/A identity from: {}".format(
        name))


def validate_identity_count(samples, expected, split_name):
    """Fail loudly when a complete official split has the wrong ID count."""
    identities = {int(sample[1]) for sample in samples}
    if identities and len(identities) != int(expected):
        raise RuntimeError(
            "AG-ReID {} identity count mismatch: expected {}, found {}. "
            "Check P/T/A parsing and the dataset split.".format(
                split_name, expected, len(identities)))
    return len(identities)
