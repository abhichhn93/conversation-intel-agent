from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd


LINE_PATTERN = re.compile(r"^(?P<timestamp>[^|]+)\|(?P<speaker>[^|]+)\|(?P<text>.+)$")
WORD_PATTERN = re.compile(r"[A-Za-z']+")

POSITIVE_WORDS = {
    "great",
    "confident",
    "happy",
    "ready",
    "well",
    "thanks",
    "clear",
    "help",
    "backup",
}
NEGATIVE_WORDS = {
    "bad",
    "delay",
    "risk",
    "risks",
    "blocker",
    "blockers",
    "disagree",
}


class TranscriptParseError(Exception):
    pass


def parse_transcript(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Transcript path is not a file: {path}")

    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if not match:
            raise TranscriptParseError(f"Malformed line at {line_number}: {line}")
        record = {k: v.strip() for k, v in match.groupdict().items()}
        records.append(record)

    if not records:
        raise TranscriptParseError("No valid transcript lines found.")

    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise TranscriptParseError("Unable to parse one or more timestamps.")
    return df


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def sentiment_label(text: str) -> str:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    score = sum(word in POSITIVE_WORDS for word in words) - sum(
        word in NEGATIVE_WORDS for word in words
    )
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def build_report(df: pd.DataFrame) -> str:
    df = df.copy()
    df["word_count"] = df["text"].apply(count_words)
    df["sentiment"] = df["text"].apply(sentiment_label)
    df["interruption"] = df["text"].str.contains("interrupt", case=False, regex=False)

    total_messages = len(df)
    total_words = int(df["word_count"].sum())
    avg_message_length = df["word_count"].mean()

    messages_per_speaker = df["speaker"].value_counts()
    avg_length_per_speaker = df.groupby("speaker")["word_count"].mean()
    dominance_ratio = df.groupby("speaker")["word_count"].sum() / total_words
    interruption_counts = df.groupby("speaker")["interruption"].sum()

    sentiment_counts = df["sentiment"].value_counts()
    sentiment_by_speaker = df.groupby("speaker")["sentiment"].value_counts().unstack(fill_value=0)

    flags: list[str] = []
    negative_count = int(sentiment_counts.get("negative", 0))
    if negative_count >= 4:
        flags.append("Friction detected")
    dominant_speaker = dominance_ratio.idxmax()
    dominant_share = float(dominance_ratio.max())
    if dominant_share > 0.45:
        flags.append("Dominance imbalance")
    total_interruptions = int(df["interruption"].sum())
    if total_interruptions >= 2:
        flags.append("High overlap / interruptions")

    if dominant_share > 0.45:
        dominance_note = f"{dominant_speaker} is leading the airtime."
    else:
        dominance_note = "Participation is fairly balanced."

    positive_count = int(sentiment_counts.get("positive", 0))
    if negative_count >= 4:
        sentiment_note = "Tone shows elevated friction."
    elif positive_count > negative_count:
        sentiment_note = "Tone is generally positive."
    elif negative_count > positive_count:
        sentiment_note = "Tone leans negative."
    else:
        sentiment_note = "Tone is mostly neutral."

    lines: list[str] = ["# Conversation Intel Report", "", "## Summary"]
    lines.append(f"- Total messages: {total_messages}")
    lines.append(f"- Total words: {total_words}")
    lines.append(f"- Average message length (words): {avg_message_length:.2f}")
    lines.append(f"- Key takeaway: {dominance_note} {sentiment_note}")
    lines.append(f"- Flags: {', '.join(flags) if flags else 'None'}")

    lines.extend(["", "## Messages Per Speaker"])
    for speaker, count in messages_per_speaker.items():
        lines.append(f"- {speaker}: {count}")

    lines.extend(["", "## Average Message Length (Words)"])
    for speaker, avg_len in avg_length_per_speaker.items():
        lines.append(f"- {speaker}: {avg_len:.2f}")

    lines.extend(["", "## Dominance Ratio (Word Share)"])
    for speaker, ratio in dominance_ratio.items():
        lines.append(f"- {speaker}: {ratio * 100:.1f}%")

    lines.extend(["", "## Interruptions"])
    lines.append(f"- Total interruptions: {total_interruptions}")
    for speaker, count in interruption_counts.items():
        if count:
            lines.append(f"- {speaker}: {int(count)}")

    lines.extend(["", "## Sentiment (Rule-based)"])
    for label in ("positive", "neutral", "negative"):
        lines.append(f"- {label.capitalize()}: {int(sentiment_counts.get(label, 0))}")

    lines.extend(["", "## Sentiment By Speaker"])
    for speaker in messages_per_speaker.index:
        pos = int(sentiment_by_speaker.get("positive", pd.Series()).get(speaker, 0))
        neu = int(sentiment_by_speaker.get("neutral", pd.Series()).get(speaker, 0))
        neg = int(sentiment_by_speaker.get("negative", pd.Series()).get(speaker, 0))
        lines.append(f"- {speaker}: {pos} positive, {neu} neutral, {neg} negative")

    lines.append("")
    return "\n".join(lines)


def print_summary(df: pd.DataFrame) -> None:
    df = df.copy()
    df["word_count"] = df["text"].apply(count_words)
    df["sentiment"] = df["text"].apply(sentiment_label)
    df["interruption"] = df["text"].str.contains("interrupt", case=False, regex=False)

    total_messages = len(df)
    total_words = int(df["word_count"].sum())
    avg_message_length = df["word_count"].mean()

    messages_per_speaker = df["speaker"].value_counts()
    dominance_ratio = df.groupby("speaker")["word_count"].sum() / total_words
    interruption_counts = df.groupby("speaker")["interruption"].sum()
    sentiment_counts = df["sentiment"].value_counts()

    flags: list[str] = []
    negative_count = int(sentiment_counts.get("negative", 0))
    if negative_count >= 4:
        flags.append("Friction detected")
    dominant_speaker = dominance_ratio.idxmax()
    dominant_share = float(dominance_ratio.max())
    if dominant_share > 0.45:
        flags.append("Dominance imbalance")
    total_interruptions = int(df["interruption"].sum())
    if total_interruptions >= 2:
        flags.append("High overlap / interruptions")

    if dominant_share > 0.45:
        dominance_note = f"{dominant_speaker} is leading the airtime."
    else:
        dominance_note = "Participation is fairly balanced."

    positive_count = int(sentiment_counts.get("positive", 0))
    if negative_count >= 4:
        sentiment_note = "Tone shows elevated friction."
    elif positive_count > negative_count:
        sentiment_note = "Tone is generally positive."
    elif negative_count > positive_count:
        sentiment_note = "Tone leans negative."
    else:
        sentiment_note = "Tone is mostly neutral."

    print("Conversation Intelligence Summary")
    print("-------------------------------")
    print(f"Total messages: {total_messages}")
    print(f"Total words: {total_words}")
    print(f"Average message length (words): {avg_message_length:.2f}")
    print(f"Key takeaway: {dominance_note} {sentiment_note}")
    print(f"Flags: {', '.join(flags) if flags else 'None'}")
    print("Messages per speaker:")
    for speaker, count in messages_per_speaker.items():
        print(f"- {speaker}: {count}")
    print("Dominance ratio (word share):")
    for speaker, ratio in dominance_ratio.items():
        print(f"- {speaker}: {ratio * 100:.1f}%")
    print("Interruptions:")
    print(f"- Total: {total_interruptions}")
    for speaker, count in interruption_counts.items():
        if count:
            print(f"- {speaker}: {int(count)}")
    print("Sentiment (rule-based):")
    for label in ("positive", "neutral", "negative"):
        print(f"- {label.capitalize()}: {int(sentiment_counts.get(label, 0))}")


def main() -> int:
    transcript_path = Path("data/sample_transcript.txt")
    try:
        df = parse_transcript(transcript_path)
    except (FileNotFoundError, TranscriptParseError) as exc:
        print(f"Error: {exc}")
        return 1

    report = build_report(df)
    Path("report.md").write_text(report, encoding="utf-8")
    print_summary(df)
    print("Report written to report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
