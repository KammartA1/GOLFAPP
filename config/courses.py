"""
Course Fingerprint Database
Each course has a weight vector across SG categories + key attributes.
These weights define what skills the course rewards historically.

Data sourced from: PGA Tour historical SG splits, manual research, DataGolf course fit studies.
Expand this as you collect more historical data.

Format:
    sg_weights: how much each SG category matters at this venue
    distance_bonus: 0-1 scale — how much raw distance helps
    accuracy_penalty: 0-1 scale — how much missing fairways hurts
    bermuda_greens: bool — putting SG on bermuda vs bentgrass is different
    elevation: high elevation = ball flies further (adjust distance calc)
    typical_conditions: links/parkland/desert/mountain
    notes: your manual insight as a golf enthusiast
"""

COURSE_PROFILES = {

    # ── MAJOR VENUES ──────────────────────────────────────────────────────

    "Augusta National": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.42, "sg_atg": 0.22, "sg_putt": 0.16},
        "distance_bonus": 0.65,
        "accuracy_penalty": 0.30,
        "bermuda_greens": True,
        "elevation_ft": 330,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.45,
        "key_skills": ["distance", "precise_irons", "bermuda_putting", "creative_short_game"],
        "notes": "Second shot dominates — approach SG is king. Long hitters have big advantage par 5s. Bermuda putting is volatile. Amen Corner wind is unpredictable."
    },

    "Pinehurst No. 2": {
        "sg_weights": {"sg_ott": 0.15, "sg_app": 0.35, "sg_atg": 0.35, "sg_putt": 0.15},
        "distance_bonus": 0.30,
        "accuracy_penalty": 0.70,
        "bermuda_greens": True,
        "elevation_ft": 480,
        "typical_conditions": "parkland_links_hybrid",
        "wind_sensitivity": 0.60,
        "key_skills": ["scrambling", "bump_and_run", "sand_play", "patience"],
        "notes": "Turtle-back greens make ATG the most important skill by far. Short game scrambling around domed greens is make-or-break. Distance matters little."
    },

    "Pebble Beach": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.38, "sg_atg": 0.20, "sg_putt": 0.22},
        "distance_bonus": 0.40,
        "accuracy_penalty": 0.65,
        "bermuda_greens": False,
        "elevation_ft": 50,
        "typical_conditions": "links_coastal",
        "wind_sensitivity": 0.85,
        "key_skills": ["wind_management", "accuracy", "coastal_links_irons"],
        "notes": "Wind is the equalizer — check weather model carefully. Coastal holes 4-10 are wind-critical. Accuracy off tee matters more than distance here."
    },

    "TPC Sawgrass": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.40, "sg_atg": 0.25, "sg_putt": 0.17},
        "distance_bonus": 0.35,
        "accuracy_penalty": 0.60,
        "bermuda_greens": True,
        "elevation_ft": 20,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.70,
        "key_skills": ["precision_irons", "course_management", "bermuda_putting"],
        "notes": "Players Championship — precision over power. 17th island green is high-variance. Bermuda greens. Wind swings from Atlantic."
    },

    "Valhalla GC": {
        "sg_weights": {"sg_ott": 0.24, "sg_app": 0.36, "sg_atg": 0.18, "sg_putt": 0.22},
        "distance_bonus": 0.70,
        "accuracy_penalty": 0.35,
        "bermuda_greens": False,
        "elevation_ft": 750,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["driving_distance", "approach_accuracy", "bentgrass_putting"],
        "notes": "PGA Championship host — long, wide layout rewards power. Reachable par-5s separate the field."
    },

    "Bethpage Black": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.38, "sg_atg": 0.22, "sg_putt": 0.18},
        "distance_bonus": 0.55,
        "accuracy_penalty": 0.70,
        "bermuda_greens": False,
        "elevation_ft": 100,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.40,
        "key_skills": ["length", "rough_play", "accurate_irons", "scrambling"],
        "notes": "Brutal rough, long carries, demanding par-4s. Complete ball-striking test."
    },

    "Oakmont CC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.35, "sg_atg": 0.27, "sg_putt": 0.20},
        "distance_bonus": 0.35,
        "accuracy_penalty": 0.75,
        "bermuda_greens": False,
        "elevation_ft": 1100,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["church_pew_bunkers", "approach_accuracy", "scrambling", "speed_management"],
        "notes": "U.S. Open host — fastest greens in championship golf, 200+ bunkers, extreme precision required."
    },

    "Winged Foot GC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.38, "sg_atg": 0.24, "sg_putt": 0.20},
        "distance_bonus": 0.40,
        "accuracy_penalty": 0.70,
        "bermuda_greens": False,
        "elevation_ft": 300,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.30,
        "key_skills": ["approach_accuracy", "scrambling", "patience"],
        "notes": "U.S. Open host — narrow fairways, thick rough, elevated greens. Favors patient, accurate ball-strikers."
    },

    "Quail Hollow": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.38, "sg_atg": 0.20, "sg_putt": 0.20},
        "distance_bonus": 0.55,
        "accuracy_penalty": 0.50,
        "bermuda_greens": True,
        "elevation_ft": 700,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["driving_distance", "approach_precision", "bermuda_putting"],
        "notes": "Wells Fargo / PGA Championship host — Green Mile finish is iconic. Distance matters, bermuda greens."
    },

    "Colonial CC": {
        "sg_weights": {"sg_ott": 0.14, "sg_app": 0.40, "sg_atg": 0.26, "sg_putt": 0.20},
        "distance_bonus": 0.20,
        "accuracy_penalty": 0.75,
        "bermuda_greens": True,
        "elevation_ft": 600,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.45,
        "key_skills": ["accuracy", "iron_play", "scrambling", "patience"],
        "notes": "Charles Schwab Challenge — Hogan's Alley. Tight, tree-lined, accuracy paramount. Short hitters can win."
    },

    "TPC River Highlands": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.36, "sg_atg": 0.22, "sg_putt": 0.22},
        "distance_bonus": 0.55,
        "accuracy_penalty": 0.40,
        "bermuda_greens": False,
        "elevation_ft": 50,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["birdie_making", "putting", "par5_scoring"],
        "notes": "Travelers Championship — short course, birdie-fest. Putting and par-5 scoring separate the field."
    },

    "Sedgefield CC": {
        "sg_weights": {"sg_ott": 0.16, "sg_app": 0.38, "sg_atg": 0.24, "sg_putt": 0.22},
        "distance_bonus": 0.25,
        "accuracy_penalty": 0.65,
        "bermuda_greens": True,
        "elevation_ft": 800,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.30,
        "key_skills": ["accuracy", "approach_precision", "bermuda_putting"],
        "notes": "Wyndham Championship — short, tight. Accuracy > distance. Bermuda greens favor local knowledge."
    },

    "Detroit Golf Club": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.34, "sg_atg": 0.22, "sg_putt": 0.22},
        "distance_bonus": 0.55,
        "accuracy_penalty": 0.40,
        "bermuda_greens": False,
        "elevation_ft": 600,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.30,
        "key_skills": ["driving_distance", "birdie_making", "bentgrass_putting"],
        "notes": "Rocket Mortgage Classic — birdie-fest, wide fairways. Distance creates par-5 eagle chances."
    },

    "Kapalua Plantation": {
        "sg_weights": {"sg_ott": 0.28, "sg_app": 0.32, "sg_atg": 0.18, "sg_putt": 0.22},
        "distance_bonus": 0.85,
        "accuracy_penalty": 0.20,
        "bermuda_greens": True,
        "elevation_ft": 300,
        "typical_conditions": "coastal_tropical",
        "wind_sensitivity": 0.80,
        "key_skills": ["driving_distance", "wind_management", "bermuda_putting"],
        "notes": "The Sentry — massive fairways, bombers paradise. Wind from trade winds is constant. Wide open favors power."
    },

    "TPC Southwind": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.38, "sg_atg": 0.24, "sg_putt": 0.20},
        "distance_bonus": 0.35,
        "accuracy_penalty": 0.65,
        "bermuda_greens": True,
        "elevation_ft": 300,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["accuracy", "approach_precision", "scrambling", "bermuda_putting"],
        "notes": "FedEx St. Jude Championship — tight, demanding. Water on many holes. Accuracy and ATG critical."
    },

    "Muirfield Village": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.40, "sg_atg": 0.22, "sg_putt": 0.20},
        "distance_bonus": 0.45,
        "accuracy_penalty": 0.55,
        "bermuda_greens": False,
        "elevation_ft": 900,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.40,
        "key_skills": ["accurate_driving", "precise_irons", "bentgrass_putting"],
        "notes": "Memorial — Jack's design favors ball strikers. Bentgrass greens are more predictable for putting model. Elevation helps distance."
    },

    "Royal Troon": {
        "sg_weights": {"sg_ott": 0.25, "sg_app": 0.35, "sg_atg": 0.20, "sg_putt": 0.20},
        "distance_bonus": 0.70,
        "accuracy_penalty": 0.45,
        "bermuda_greens": False,
        "elevation_ft": 10,
        "typical_conditions": "links",
        "wind_sensitivity": 0.95,
        "key_skills": ["links_game", "wind_management", "bump_and_run", "distance"],
        "notes": "The Open — pure links. Wind model is critical, check direction vs. layout. Distance matters on exposed holes. Low ball flight players have an edge."
    },

    "St Andrews Old Course": {
        "sg_weights": {"sg_ott": 0.28, "sg_app": 0.30, "sg_atg": 0.22, "sg_putt": 0.20},
        "distance_bonus": 0.80,
        "accuracy_penalty": 0.25,
        "bermuda_greens": False,
        "elevation_ft": 5,
        "typical_conditions": "links",
        "wind_sensitivity": 0.90,
        "key_skills": ["distance", "low_trajectory", "wind_reading", "creativity"],
        "notes": "Home of Golf — wide fairways = OTT less punishing. Massive double greens, wind is dominant factor. Distance is a massive weapon here."
    },

    # ── REGULAR TOUR VENUES ───────────────────────────────────────────────

    "TPC Scottsdale": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.36, "sg_atg": 0.20, "sg_putt": 0.22},
        "distance_bonus": 0.75,
        "accuracy_penalty": 0.25,
        "bermuda_greens": True,
        "elevation_ft": 1500,
        "typical_conditions": "desert",
        "wind_sensitivity": 0.30,
        "key_skills": ["distance", "birdie_making", "hot_putting"],
        "notes": "WM Phoenix — BOMBER course. Very high scoring, need to make birdies. Hot putting week = winners. Elevation adds distance. Crowd on 16th is chaos."
    },

    "Bay Hill": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.42, "sg_atg": 0.20, "sg_putt": 0.18},
        "distance_bonus": 0.50,
        "accuracy_penalty": 0.50,
        "bermuda_greens": True,
        "elevation_ft": 80,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.55,
        "key_skills": ["accurate_irons", "wind_management", "bermuda_putting"],
        "notes": "Arnold Palmer Invitational — tough, windy. App SG is king. 18th is brutal approach over water. Wind from south/west is significant."
    },

    "Torrey Pines South": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.40, "sg_atg": 0.20, "sg_putt": 0.18},
        "distance_bonus": 0.55,
        "accuracy_penalty": 0.45,
        "bermuda_greens": False,
        "elevation_ft": 300,
        "typical_conditions": "parkland_coastal",
        "wind_sensitivity": 0.60,
        "key_skills": ["length", "approach_precision", "rough_escaping"],
        "notes": "Farmers Insurance / US Open — long rough penalizes wayward drives. Distance still matters. Coastal wind varies round to round."
    },

    "Riviera CC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.40, "sg_atg": 0.25, "sg_putt": 0.17},
        "distance_bonus": 0.35,
        "accuracy_penalty": 0.55,
        "bermuda_greens": False,
        "elevation_ft": 200,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["precision", "course_management", "bunker_play"],
        "notes": "Genesis Invitational — The Riviera. Classic parkland. Unique par 4 4th with bunker in fairway. ATG matters — lots of bunker escapes needed."
    },

    "Innisbrook (Copperhead)": {
        "sg_weights": {"sg_ott": 0.16, "sg_app": 0.38, "sg_atg": 0.28, "sg_putt": 0.18},
        "distance_bonus": 0.30,
        "accuracy_penalty": 0.65,
        "bermuda_greens": True,
        "elevation_ft": 50,
        "typical_conditions": "parkland_tropical",
        "wind_sensitivity": 0.30,
        "key_skills": ["scrambling", "tree_navigation", "bermuda_putting"],
        "notes": "Valspar Championship — The Snake Pit. Tight, tree-lined. Scrambling is essential. Distance is less important than accuracy. ATG is critical."
    },

    "Harbour Town": {
        "sg_weights": {"sg_ott": 0.12, "sg_app": 0.38, "sg_atg": 0.30, "sg_putt": 0.20},
        "distance_bonus": 0.15,
        "accuracy_penalty": 0.80,
        "bermuda_greens": True,
        "elevation_ft": 10,
        "typical_conditions": "coastal_parkland",
        "wind_sensitivity": 0.70,
        "key_skills": ["accuracy", "scrambling", "wind_management", "putting"],
        "notes": "RBC Heritage — short hitters' paradise. One of the most accuracy-dependent courses on tour. Distance is actually a slight negative here. Wind off Calibogue Sound matters."
    },

    "East Lake GC": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.40, "sg_atg": 0.22, "sg_putt": 0.20},
        "distance_bonus": 0.40,
        "accuracy_penalty": 0.55,
        "bermuda_greens": False,
        "elevation_ft": 1000,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["complete_ball_striker", "clutch_putting"],
        "notes": "Tour Championship — elite field only. Complete game wins. Historical performance at East Lake in particular matters for form model."
    },
    "Memorial Park GC": {
        "sg_weights": {"sg_ott": 0.22, "sg_app": 0.32, "sg_atg": 0.18, "sg_putt": 0.28},
        "distance_bonus": 0.45,
        "accuracy_penalty": 0.55,
        "bermuda_greens": True,
        "elevation_ft": 50,
        "typical_conditions": "parkland_tropical",
        "wind_sensitivity": 0.35,
        "key_skills": ["driving_distance", "approach_accuracy", "bermuda_putting"],
        "notes": "Texas Children's Houston Open — Tom Doak municipal redesign. Long par-4s, bermuda greens, tight corridors reward driving distance + accuracy."
    },
    "Aronimink GC": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.32, "sg_atg": 0.18, "sg_putt": 0.30},
        "distance_bonus": 0.40,
        "accuracy_penalty": 0.60,
        "bermuda_greens": False,
        "elevation_ft": 400,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.30,
        "key_skills": ["iron_play", "scrambling", "bentgrass_putting"],
        "notes": "2026 PGA Championship — classic Donald Ross design, tight tree-lined, bentgrass, precise iron play and scrambling rewarded."
    },
    "Shinnecock Hills": {
        "sg_weights": {"sg_ott": 0.17, "sg_app": 0.30, "sg_atg": 0.22, "sg_putt": 0.31},
        "distance_bonus": 0.25,
        "accuracy_penalty": 0.60,
        "bermuda_greens": False,
        "elevation_ft": 50,
        "typical_conditions": "links_coastal",
        "wind_sensitivity": 0.75,
        "key_skills": ["wind_play", "scrambling", "approach_accuracy"],
        "notes": "2026 U.S. Open — links-influenced, coastal wind, fescue, firm and fast, complete ball-striking test."
    },
    "Royal Birkdale": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.28, "sg_atg": 0.23, "sg_putt": 0.31},
        "distance_bonus": 0.30,
        "accuracy_penalty": 0.55,
        "bermuda_greens": False,
        "elevation_ft": 15,
        "typical_conditions": "links",
        "wind_sensitivity": 0.85,
        "key_skills": ["wind_play", "scrambling", "links_experience"],
        "notes": "2026 Open Championship — English links, dune-lined fairways, pot bunkers, heavy wind, scrambling critical."
    },
    "TPC Toronto": {
        "sg_weights": {"sg_ott": 0.18, "sg_app": 0.30, "sg_atg": 0.18, "sg_putt": 0.34},
        "distance_bonus": 0.35,
        "accuracy_penalty": 0.50,
        "bermuda_greens": False,
        "elevation_ft": 300,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.35,
        "key_skills": ["ball_striking", "bentgrass_putting"],
        "notes": "RBC Canadian Open — Osprey Valley, heathland-style layout, bentgrass, balanced test."
    },
    "Bellerive CC": {
        "sg_weights": {"sg_ott": 0.20, "sg_app": 0.30, "sg_atg": 0.17, "sg_putt": 0.33},
        "distance_bonus": 0.45,
        "accuracy_penalty": 0.55,
        "bermuda_greens": False,
        "elevation_ft": 500,
        "typical_conditions": "parkland",
        "wind_sensitivity": 0.30,
        "key_skills": ["driving_distance", "iron_play", "bentgrass_putting"],
        "notes": "2026 BMW Championship — long layout, bentgrass, Midwest humidity, rewards power and precision."
    },
}

# ─────────────────────────────────────────────────────────────────
# COURSE ALIAS MAP — normalize different name spellings
# ─────────────────────────────────────────────────────────────────
COURSE_ALIASES = {
    "TPC Sawgrass (Stadium)": "TPC Sawgrass",
    "Augusta National Golf Club": "Augusta National",
    "The Old Course at St Andrews": "St Andrews Old Course",
    "St. Andrews (Old Course)": "St Andrews Old Course",
    "Pinehurst Resort & CC (Course No. 2)": "Pinehurst No. 2",
    "Torrey Pines (South Course)": "Torrey Pines South",
    "Innisbrook Resort (Copperhead Course)": "Innisbrook (Copperhead)",
    "Harbour Town Golf Links": "Harbour Town",
    "Memorial Park Golf Course": "Memorial Park GC",
    "Memorial Park": "Memorial Park GC",
    "Aronimink Golf Club": "Aronimink GC",
    "Shinnecock Hills Golf Club": "Shinnecock Hills",
    "Royal Birkdale Golf Club": "Royal Birkdale",
    "TPC Toronto at Osprey Valley": "TPC Toronto",
    "Bellerive Country Club": "Bellerive CC",
    "Augusta National GC": "Augusta National",
    "Augusta National Golf Course": "Augusta National",
    "Valhalla Golf Club": "Valhalla GC",
    "Bethpage State Park (Black Course)": "Bethpage Black",
    "Oakmont Country Club": "Oakmont CC",
    "Winged Foot Golf Club": "Winged Foot GC",
    "Quail Hollow Club": "Quail Hollow",
    "Colonial Country Club": "Colonial CC",
    "TPC River Highlands": "TPC River Highlands",
    "Sedgefield Country Club": "Sedgefield CC",
    "Detroit Golf Club": "Detroit Golf Club",
    "Kapalua Resort (Plantation Course)": "Kapalua Plantation",
    "TPC Southwind": "TPC Southwind",
}

# ─────────────────────────────────────────────────────────────────
# SURFACE TYPES — for putting SG split (bentgrass vs bermuda)
# This matters — a great bermuda putter may be a weak bentgrass putter
# ─────────────────────────────────────────────────────────────────
BERMUDA_COURSES = {
    name for name, data in COURSE_PROFILES.items()
    if data.get("bermuda_greens", False)
}

BENTGRASS_COURSES = {
    name for name, data in COURSE_PROFILES.items()
    if not data.get("bermuda_greens", False)
}

def get_course_profile(course_name: str) -> dict | None:
    """Resolve course name (with alias handling) and return profile."""
    name = COURSE_ALIASES.get(course_name, course_name)
    return COURSE_PROFILES.get(name, None)

def get_dominant_skill(course_name: str) -> str:
    """Return the single most important SG skill for a course."""
    profile = get_course_profile(course_name)
    if not profile:
        return "sg_app"  # Default — approach is most predictive generally
    weights = profile["sg_weights"]
    return max(weights, key=weights.get)
