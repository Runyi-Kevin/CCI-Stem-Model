"""Central configuration: every assumption and hard-coded parameter lives here."""
from __future__ import annotations

DISCLAIMER = ("SYNTHETIC DATA - illustrative only. These are simulated records and "
              "NOT actual workshop results.")
FOOTER = "Synthetic illustrative data (n=450) - not actual workshop results"

SEED = 42
SCORE_DECIMALS = 8  # composite precision used for ranking/banding (tie safety)
EXPECTED_N = 450
TOTAL_SESSIONS = 10
SHEET_NAME = "Participant Data"
HEADER_ROW = 3  # zero-based row index of the header line in the source sheet

INDICATORS = ["attendance_punctuality", "knowledge_gain", "participation",
              "teamwork", "group_discussion"]
LABELS = {
    "attendance_punctuality": "Attendance & Punctuality",
    "knowledge_gain": "Knowledge Gain",
    "participation": "Participation",
    "teamwork": "Teamwork",
    "group_discussion": "Group Discussions",
}
SHORT = {"attendance_punctuality": "Attend.", "knowledge_gain": "Know. Gain",
         "participation": "Particip.", "teamwork": "Teamwork",
         "group_discussion": "Discuss."}

BASELINE_WEIGHTS = {"attendance_punctuality": 0.10, "knowledge_gain": 0.30,
                    "participation": 0.15, "teamwork": 0.15, "group_discussion": 0.30}

# Attendance & Punctuality score = 100 * (0.6 * attendance rate + 0.4 * punctuality rate)
ATTENDANCE_SUBWEIGHT = 0.6

# (lower bound, label) - evaluated top-down
BANDS = [(80.0, "Excellent"), (70.0, "Very Good"), (60.0, "Good"),
         (50.0, "Fair"), (float("-inf"), "Needs Improvement")]
BAND_ORDER = ["Needs Improvement", "Fair", "Good", "Very Good", "Excellent"]

# Source-sheet header -> internal column name
RAW_COLUMNS = {
    "Participant ID": "participant_id",
    "Team ID": "team_id",
    "Sessions Attended (of 10)": "sessions_attended",
    "Sessions On Time": "sessions_on_time",
    "Pre-test Score": "pre_test",
    "Post-test Score": "post_test",
    "In-Session Participation Score": "participation",
    "Teamwork & Collaboration Score": "teamwork",
    "Group Discussions Score": "group_discussion",
}
SCORE_COLUMNS = ["pre_test", "post_test", "participation", "teamwork", "group_discussion"]

# Alternative weighting scenarios for sensitivity analysis (each sums to 1)
SCENARIOS = {
    "Baseline (10/30/15/15/30)": BASELINE_WEIGHTS,
    "Equal weights (20 each)": {k: 0.20 for k in INDICATORS},
    "Knowledge-led (10/40/10/10/30)": dict(zip(INDICATORS, [0.10, 0.40, 0.10, 0.10, 0.30])),
    "Engagement-led (15/20/25/20/20)": dict(zip(INDICATORS, [0.15, 0.20, 0.25, 0.20, 0.20])),
    "Collaboration-led (10/20/10/30/30)": dict(zip(INDICATORS, [0.10, 0.20, 0.10, 0.30, 0.30])),
    "Attendance-heavy (25/25/15/15/20)": dict(zip(INDICATORS, [0.25, 0.25, 0.15, 0.15, 0.20])),
    "Discussion-light (10/35/20/20/15)": dict(zip(INDICATORS, [0.10, 0.35, 0.20, 0.20, 0.15])),
}
OAT_DELTAS = [-0.10, -0.05, 0.05, 0.10]
MC_DRAWS = 2000
MC_CONCENTRATION = 50.0   # Dirichlet concentration around baseline weights
NOISE_SDS = [1, 2, 3, 5, 8]
NOISE_REPS = 200
TOP_FRACTION = 0.10
K_RANGE = range(2, 9)
KMEANS_N_INIT = 25
BOOTSTRAPS = 100
PARALLEL_SIMS = 500
