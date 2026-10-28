"""Sound for a rendered clip, from its cue sheet:

    python clips/audio/build.py <cues.json> <out_dir> <track_name> <out.mp4>=<video.mp4> [<out2.mp4>=<video2.mp4> ...]

Writes <out_dir>/<track_name>.m4a (mixed and mastered to -14 LUFS, AAC 192 kbps) and every listed video with that
track muxed in (the video stream is copied, not re-encoded). See mix.py for the cue sheet format.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix  # noqa: E402

if __name__ == '__main__':
    cues, out_dir, name = sys.argv[1:4]
    videos = dict(a.split('=', 1) for a in sys.argv[4:])
    mix.build(cues, out_dir, videos, name)
