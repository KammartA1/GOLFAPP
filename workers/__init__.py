"""Background workers — run independently of Streamlit.

All state ingestion, model evaluation, and reporting happens here.
Streamlit is ONLY a read-only frontend.

Workers:
    odds_worker     — Scrape PrizePicks/DraftKings lines every 15 min
    signal_worker   — Evaluate lines → generate signals → approve/reject
    closing_worker  — Capture closing lines + settle bets + compute CLV
    model_worker    — Weekly retrain + drift detection
    report_worker   — Daily edge reports + system health
"""
