"""Paths and machine settings in one place.

FFMPEG: the ffmpeg executable (the FFMPEG environment variable wins; then the full build on this machine; then
whatever 'ffmpeg' is on PATH). THREADS: ffmpeg threads (2 on this laptop: it gets graphics driver resets under load).
OUT: where make.py writes (out/<clip>/, never committed). LAB_REVIEW: the review folder the owner looks at.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
FFMPEG = os.environ.get('FFMPEG') or 'C:/Users/user/.codex/tools/ffmpeg-full/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe'
if not os.path.exists(FFMPEG):
    FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'
THREADS = 2
OUT = os.path.join(ROOT, 'out')
LAB_REVIEW = os.environ.get('LAB_REVIEW', 'D:/pythonProject4/lab-review')
FPS = 30
COPY_MB = 30.0            # the small copy must stay under this many megabytes
