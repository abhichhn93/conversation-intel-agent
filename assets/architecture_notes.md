# Architecture Notes

## Overview
Conversation Intel Agent is a CLI pipeline that turns raw transcripts into engagement metrics and an audit-ready report.

## Data flow
1) Load transcript lines from `data/sample_transcript.txt`.
2) Parse timestamp, speaker, and text with a regex.
3) Compute metrics: message counts, average length, dominance ratio, interruptions, sentiment.
4) Emit a Markdown report and a console summary.

## Design choices
- Pandas for transparent aggregation.
- Simple rule-based sentiment for explainability.
- Deterministic sample input for reproducible demos.
