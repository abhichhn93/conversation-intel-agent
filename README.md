# conversation-intel-agent

## What problem this solves
Conversation transcripts are messy to interpret at scale. This tool summarizes engagement, dominance, interruptions, and sentiment signals so teams can quickly understand how a discussion actually went.

## Business use cases
- Sales: review call dynamics and spot which topics drove engagement
- Support: identify frustration patterns and representative dominance
- Research: quantify participation balance and conversation health

## How to run
From the project root:

```bash
pip install -r requirements.txt
python analyzer.py
```

Output:
- `report.md` generated in the project root
- Console summary of key metrics
