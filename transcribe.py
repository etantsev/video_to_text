from faster_whisper import WhisperModel

AUDIO_PATH = "audio.wav"
MODEL_SIZE = "small"

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
segments, info = model.transcribe(
    AUDIO_PATH,
    language="ru",
    beam_size=5,
    vad_filter=True,
)

print(f"Detected language: {info.language} ({info.language_probability:.2f})")

all_segments = list(segments)

with open("transcript.txt", "w", encoding="utf-8") as txt:
    for segment in all_segments:
        line = segment.text.strip()
        if line:
            txt.write(line + "\n")


def fmt_srt_time(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    secs = ms_total // 1000
    millis = ms_total % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


with open("transcript.srt", "w", encoding="utf-8") as srt:
    for idx, segment in enumerate(all_segments, 1):
        srt.write(f"{idx}\n")
        srt.write(f"{fmt_srt_time(segment.start)} --> {fmt_srt_time(segment.end)}\n")
        srt.write(segment.text.strip() + "\n\n")

print(f"Wrote {len(all_segments)} segments to transcript.txt and transcript.srt")
