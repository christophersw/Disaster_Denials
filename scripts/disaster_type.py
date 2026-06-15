# scripts/disaster_type.py
"""
Title: Disaster Type Classifier
Description: Classifies a free-text FEMA incident_name into a single primary
    disaster type. incident_name values are compound (e.g. "Severe Storms,
    Tornadoes, and Flooding"), so classify() scans an ordered priority list of
    (type, keywords) and returns the first type whose keyword appears as a
    substring — most specific/severe hazards first, the generic "Severe Storm"
    last, and "Other" when nothing matches or the name is blank. Shared by the
    disaster-type Sankey and the bucketing chart so both use one classification.
Changelog:
    2026-06-14  Initial version.
"""

# Ordered most-specific/severe first; the first keyword hit wins. Keywords are
# matched case-insensitively as substrings of incident_name. "Severe Storm" is
# the generic catch-all and is checked last so compound names resolve to their
# more specific hazard (e.g. "Severe Storms and Flooding" -> Flooding).
DISASTER_TYPE_PRIORITY = [
    ("Hurricane", ["hurricane"]),
    ("Tropical Storm", ["tropical storm", "tropical depression", "typhoon"]),
    ("Earthquake", ["earthquake", "seismic"]),
    ("Wildfire/Fire", ["wildfire", "fire"]),
    ("Mudslide/Landslide", ["mudslide", "landslide", "mudflow", "debris flow"]),
    ("Tornado", ["tornado"]),
    ("Winter Storm", ["winter", "snow", "blizzard", "ice storm", "severe ice"]),
    ("Flooding", ["flood"]),
    ("Freeze/Cold", ["freez", "subfreez", "frost", "cold"]),
    ("Drought", ["drought"]),
    ("Severe Storm", [
        "severe storm", "severe weather", "severe thunderstorm",
        "straight-line wind", "high wind", "windstorm", "storm", "wind",
    ]),
]

OTHER_TYPE = "Other"

# Display order for charts: priority order, then Other last.
DISASTER_TYPE_ORDER = [name for name, _ in DISASTER_TYPE_PRIORITY] + [OTHER_TYPE]


def classify(incident_name):
    """Return the single primary disaster type for an incident name.

    Args:
        incident_name (str | None): the free-text FEMA incident description.
    Returns:
        str — the first matching type from DISASTER_TYPE_PRIORITY, or "Other"
        if the name is missing/blank or matches no keyword.
    """
    if not incident_name:
        return OTHER_TYPE
    text = incident_name.lower()
    for disaster_type, keywords in DISASTER_TYPE_PRIORITY:
        if any(keyword in text for keyword in keywords):
            return disaster_type
    return OTHER_TYPE
