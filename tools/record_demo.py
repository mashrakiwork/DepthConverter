"""Record the walkthroughs the README shows.

Drives the real Qt app in-process, screenshots its window while a scripted tour
uses it, and encodes the frames as short H.264 MP4 clips - one per act.

MP4 rather than GIF: sharper at 1080p and a fraction of the size. The clips are
uploaded to GitHub and embedded by URL, not committed, so the repo never carries
their weight. (Pass `--format gif` for the old GIFs, or `both`.)

Acts rather than one film, so each piece lands beside the README section it
illustrates and is small enough that someone reading on a phone actually sees
it. They are filmed in one continuous session, so state carries from one act to
the next - the depth video that `depth` produces is the one `convert` turns into
3D.

**Nothing here is staged.** The tour drives the shipping widgets, and the jobs it
starts are the real pipeline running the real models on the GPU: the depth map
is real inference and the SBS output is the real stereo warp. What the tour does
do is *cut* - a job takes far longer than anyone will watch, so its middle is
either time-lapsed or skipped outright, and every cut is captioned on screen with
how much real time was taken out. A cut is honest; a fake progress bar would not
be.

Why drive the app in-process rather than a screen recorder: a hand-recorded clip
is stale the moment the interface changes, and re-recording one by hand is enough
work that nobody does it. This is a build step. Change the app, run it again,
re-upload the clips.

    uv run python tools/record_demo.py            # every act
    uv run python tools/record_demo.py depth      # just one
    uv run python tools/record_demo.py --format gif   # old GIFs instead of MP4
    uv run python tools/record_demo.py --keep     # leave the working files
    uv run python tools/record_demo.py result --work=<dir printed by --keep>
    uv run python tools/record_demo.py --out-dir=<dir>   # where clips are written

The acts are upscale, depth, convert, runall and result. Four of them film the
interface; `result` films the files instead, because the one thing a reader
most wants to see - what came out - is not something the interface shows. It
needs the outputs `depth` and `convert` write, so record those in the same
session.

The demo clip ships in the repo at assets/demo/source.mp4, committed rather
than fetched so a recording is reproducible offline and survives the source
going away; it is re-downloaded only if that file is missing. It came from
Pexels (CLIP_URL below) under the Pexels License, which allows commercial use
and requires no attribution.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is missing. Run DepthConverter.bat once, or: uv sync")

#: Where the finished GIFs land, beside the README that shows them.
MEDIA = ROOT / "docs" / "media"

#: The demo clip. Committed with the repo; re-fetched only if the file is gone.
CLIP_URL = (
    "https://videos.pexels.com/video-files/8496259/8496259-hd_1920_1080_25fps.mp4"
)
CLIP = ROOT / "assets" / "demo" / "source.mp4"

#: Default working directory for a recording. Deliberately a short path off the
#: drive root, NOT tempfile.mkdtemp(): that resolves under
#: C:\\Users\\<name>\\AppData\\Local\\Temp\\..., and the tour types the work dir
#: straight into the app's path fields, so a temp path would burn the operator's
#: personal name into every filmed clip. This one is anonymous and tidy.
DEMO_WORK = Path(Path.home().anchor or "C:\\") / "DepthDemo"

#: The app renders at this logical size; the whole tab's form and the log panel
#: are both in frame. Captured natively for MP4 (~1080 tall), scaled down for GIF.
WINDOW_SIZE = (1376, 1080)

#: H.264 output: normalise to 1080 tall, quality (lower is sharper), frame rate.
MP4_HEIGHT = 1080
CRF = 23
MP4_FPS = 30

#: Output width of the GIFs. Wide enough to read the field labels on a phone.
GIF_WIDTH = 820

#: Frames per second while filming. The app is a form, not a game - what moves
#: is a progress bar and a log, and 8 is plenty for both.
FPS = 8

#: Hard ceiling on frames in a finished act. Past this the act is thinned
#: uniformly: frame count is what a GIF costs, and nobody watches a README
#: animation for half a minute.
MAX_FRAMES = 80

#: The result composite: width, how many source frames to skip between frames,
#: and the playback delay. These defaults are for the GIF fallback, which is
#: narrower and sparser than the UI acts even though it carries four panels,
#: because photographs are far more expensive than a flat grey form: nothing
#: repeats between frames, so frame differencing has nothing to remove and every
#: frame costs close to a full picture. At 960 wide and every 4th frame this act
#: alone came to 13 MB. The MP4 build overrides all three in main(): it composes
#: from the real output pixels (not a screen capture), so it carries every frame
#: at the clip's native rate and plays as smoothly as the footage itself.
RESULT_WIDTH = 680
RESULT_STRIDE = 9
RESULT_DELAY = 170

#: The demo clip's native frame rate (see CLIP_URL): how many DISTINCT frames of
#: motion exist per second. Drives the MP4 result composite's per-frame hold so
#: it plays back in real time - every source frame, one after another.
RESULT_FPS = 25

#: Container frame rate for the MP4 result. Higher than RESULT_FPS so playback is
#: paced smoothly on 60/120 Hz displays, where 25 fps content otherwise lands on
#: an uneven 2.4-refresh cadence that reads as micro-judder. The extra frames are
#: duplicates of the 25 real ones (the source carries no more motion than that),
#: so they compress to almost nothing and the file barely grows.
RESULT_ENCODE_FPS = 60

#: Palette depth for the result composite. The full 256 because one of its four
#: panels is a smooth grey depth ramp: sharing a shallower palette with three
#: photographs starves the ramp of slots and posterises it into contour bands,
#: which reads as though the app produced a banded depth map.
RESULT_COLORS = 256

#: Playback delay bounds, in milliseconds. GIF stores delays in hundredths and
#: most viewers turn anything under 2 into 10, so 40 is the practical floor.
#: The ceiling keeps a slow capture from stalling the animation.
MIN_DELAY, MAX_DELAY = 40, 260


# --------------------------------------------------------------------------- #
# Capture.
# --------------------------------------------------------------------------- #


class Film:
    """Holds the frames of one act, and the real time each was taken at.

    The app runs on this thread, so filming and driving take turns: every
    capture pumps the Qt event loop first, which is what lets a job started on
    a worker thread paint its progress between two frames.
    """

    def __init__(self, app, window):
        self.app = app
        self.window = window
        #: Frames stream to disk (only paths held); the encoder reads them back.
        self.dir = Path(tempfile.mkdtemp(prefix="depthconverter-frames-"))
        self.frames: list[Path] = []
        self.stamps: list[float] = []
        self.forced: list[int | None] = []

    def add(
        self, image: Image.Image, hold_ms: int | None, stamp: float | None = None
    ) -> None:
        path = self.dir / f"frame-{len(self.frames):06d}.png"
        image.save(path)
        self.frames.append(path)
        self.stamps.append(time.monotonic() if stamp is None else stamp)
        self.forced.append(hold_ms)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def shoot(self, hold_ms: int | None = None) -> None:
        from PySide6.QtCore import QBuffer, QByteArray

        self.app.processEvents()
        pixmap = self.window.grab()
        # Through PNG rather than poking at the raw buffer: QImage pads every
        # scanline to a 4-byte boundary, and the padding lands in the middle of
        # the picture if you hand the bytes straight to Pillow.
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        buffer.close()
        # Kept at native resolution; the encoder scales and (for GIF) quantises.
        self.add(Image.open(io.BytesIO(bytes(data))).convert("RGB"), hold_ms)

    def hold(self, seconds: float) -> None:
        """Film in real time for a while, at roughly FPS.

        Roughly, because a capture costs whatever it costs. The real timestamps
        are kept and turned into per-frame delays at the end, so a slow capture
        stretches that frame rather than speeding the playback up.
        """
        interval = 1.0 / FPS
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            started = time.monotonic()
            self.shoot()
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self.app.processEvents()
                time.sleep(min(remaining, 0.05))

    def pump(self, seconds: float) -> None:
        """Keep the app responsive without filming any of it."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)

    def lapse(self, first: int) -> None:
        """Halve the frames captured since `first`, keeping every second one."""
        for path in self.frames[first + 1 :: 2]:
            path.unlink(missing_ok=True)
        self.frames[first:] = self.frames[first::2]
        self.stamps[first:] = self.stamps[first::2]
        self.forced[first:] = self.forced[first::2]


# --------------------------------------------------------------------------- #
# The tour's verbs.
#
# Driven through the widgets a user actually operates, so a control that is
# renamed or removed breaks the recording loudly instead of quietly filming the
# wrong thing.
# --------------------------------------------------------------------------- #


def _elapsed(seconds: float) -> str:
    """A caption reader should not have to divide. Past a minute and a half,
    "21 min" beats "1280s"."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f} min"


class Tour:
    def __init__(self, film: Film, window):
        self.film = film
        self.window = window
        self.tabs = window.centralWidget()

    # -- narration ---------------------------------------------------------- #

    def caption(self, text: str | None) -> None:
        """The two things a silent GIF cannot say for itself: which act this is,
        and that a cut just removed real elapsed time."""
        self.window.set_caption(text)
        self.film.app.processEvents()

    # -- navigation --------------------------------------------------------- #

    def open_tab(self, index: int, settle: float = 0.7) -> None:
        self.tabs.setCurrentIndex(index)
        self.film.hold(settle)

    def type_path(self, picker, value: str) -> None:
        """Type into a path field the way a person would, a chunk at a time, so
        the GIF shows the field being filled rather than blinking to full."""
        picker.edit.setFocus()
        picker.edit.clear()
        step = max(6, len(value) // 6)
        for end in range(step, len(value) + step, step):
            picker.edit.setText(value[:end])
            picker.edit.setCursorPosition(len(picker.edit.text()))
            self.film.shoot()
        picker.edit.setText(value)
        picker.edit.editingFinished.emit()
        self.film.hold(0.4)

    def choose(self, combo, index: int) -> None:
        combo.setCurrentIndex(index)
        self.film.hold(0.5)

    def tick(self, check, value: bool) -> None:
        check.setChecked(value)
        self.film.hold(0.45)

    # -- running a real job ------------------------------------------------- #

    def run_job(
        self,
        tab,
        label: str,
        *,
        lead: float = 2.5,
        tail: float = 1.6,
        timeout: float = 900.0,
    ) -> float:
        """Press Start, film the beginning, cut the middle, film the end.

        Returns the real wall-clock seconds the job took, which the caller
        captions so the cut is stated rather than hidden.
        """
        started = time.monotonic()
        tab.start_btn.click()
        self.film.hold(lead)

        # Time-lapse the middle: keep filming so progress and log lines are
        # visibly moving, then throw away every other frame each time the act
        # grows past what a README GIF should carry.
        mark = len(self.film.frames)
        budget = 40
        while tab.worker is not None and tab.worker.isRunning():
            if time.monotonic() - started > timeout:
                raise RuntimeError(f"{label}: still running after {timeout:.0f}s")
            self.film.hold(1.0)
            if len(self.film.frames) - mark > budget:
                self.film.lapse(mark)

        elapsed = time.monotonic() - started
        self.film.hold(tail)
        print(f"  · {label}: {_elapsed(elapsed)} of real work, time-lapsed")
        return elapsed


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #


def find_ffmpeg() -> str:
    bundled = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    found = shutil.which("ffmpeg")
    if found:
        return found
    sys.exit(
        "ffmpeg not found. Put it in tools/ffmpeg or on PATH, or use --format gif."
    )


def frame_durations(stamps: list[float], forced: list[int | None]) -> list[int]:
    """Per-frame on-screen time in ms: a forced hold, else real elapsed time to
    the next frame, clamped to MIN_DELAY..MAX_DELAY."""
    durations: list[int] = []
    for index in range(len(stamps)):
        override = forced[index]
        if override is not None:
            durations.append(max(MIN_DELAY, override))
        elif index + 1 < len(stamps):
            durations.append(
                min(
                    MAX_DELAY,
                    max(MIN_DELAY, round((stamps[index + 1] - stamps[index]) * 1000)),
                )
            )
        else:
            durations.append(durations[-1] if durations else 140)
    return durations


def build_mp4(film: Film, out: Path, ffmpeg: str, fps: int = MP4_FPS) -> None:
    """Encode the frames as H.264, holding each for its real duration via the
    concat demuxer, scaled to 1080 tall and resampled to a constant frame rate.

    `fps` defaults to the UI acts' rate; the result composite passes its own
    native rate so its full-frame-rate footage is encoded without a resample."""
    frames, stamps, forced = film.frames, film.stamps, film.forced
    if len(frames) < 2:
        print(f"  ! {out.name}: nothing captured", file=sys.stderr)
        return

    durations = frame_durations(stamps, forced)
    work = frames[0].parent
    lines: list[str] = []
    for path, ms in zip(frames, durations, strict=True):
        lines.append(f"file '{path.name}'")
        lines.append(f"duration {ms / 1000:.4f}")
    lines.append(f"file '{frames[-1].name}'")
    listing = work / f"{out.stem}.concat.txt"
    listing.write_text("\n".join(lines), encoding="utf-8")

    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            listing.name,
            "-vf",
            f"scale=-2:{MP4_HEIGHT},fps={fps},format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            str(CRF),
            "-preset",
            "veryslow",
            "-movflags",
            "+faststart",
            str(out),
        ],
        cwd=work,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    listing.unlink(missing_ok=True)
    if result.returncode != 0:
        sys.exit(
            f"ffmpeg failed on {out.name}:\n{result.stderr.decode(errors='replace')}"
        )
    span = stamps[-1] - stamps[0]
    print(
        f"wrote {out.name}: {len(frames)} frames, {span:.0f}s filmed, "
        f"{out.stat().st_size / 1_000_000:.1f} MB"
    )


def build_gif(
    film: Film,
    out: Path,
    colors: int = 64,
    cap: bool = True,
    width: int | None = GIF_WIDTH,
) -> None:
    """Quantise against one shared palette and write the GIF."""
    frames, stamps, forced = film.frames, film.stamps, film.forced
    if len(frames) < 2:
        print(f"  ! {out.name}: nothing captured", file=sys.stderr)
        return

    # Thin the whole act if it still runs long after the per-phase cuts.
    if cap and len(frames) > MAX_FRAMES:
        last = len(frames) - 1
        picks = [round(i * last / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
        frames = [frames[i] for i in picks]
        stamps = [stamps[i] for i in picks]
        forced = [forced[i] for i in picks]

    # Read the PNGs back, downscaling UI frames to the GIF width (result sheets
    # are already at their own size, so those pass width=None).
    images: list[Image.Image] = []
    for path in frames:
        image = Image.open(path).convert("RGB")
        if width and image.width != width:
            scale = width / image.width
            image = image.resize(
                (width, round(image.height * scale)), Image.Resampling.LANCZOS
            )
        images.append(image)

    # One shared palette, from a strip of sampled frames, so frame differencing
    # keeps the file small and the background does not shimmer.
    step = max(1, len(images) // 24)
    sample = images[::step]
    size = images[0].size
    strip = Image.new("RGB", (size[0], size[1] * len(sample)))
    for index, frame in enumerate(sample):
        strip.paste(frame, (0, index * size[1]))
    palette = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

    # dither=NONE: Floyd-Steinberg noise differs every frame, so nothing
    # compresses and the file roughly triples.
    quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in images]

    durations = frame_durations(stamps, forced)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(
        out,
        save_all=True,
        append_images=quantised[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    span = stamps[-1] - stamps[0]
    print(
        f"wrote {out.name}: {len(frames)} frames, {span:.0f}s filmed, "
        f"{sum(durations) / 1000:.0f}s playback, "
        f"{out.stat().st_size / 1_000_000:.1f} MB"
    )


# --------------------------------------------------------------------------- #
# The acts.
# --------------------------------------------------------------------------- #


def act_upscale(tour: Tour, work: Path) -> None:
    """Tab 1: a real Real-ESRGAN pass over the clip."""
    tab = tour.window.upscale_tab
    tour.caption("Upscale · run the clip through Real-ESRGAN first")
    tour.open_tab(0)
    tour.type_path(tab.input_pick, str(CLIP))
    tour.type_path(tab.output_pick, str(work / "upscaled"))
    tour.choose(tab.model_combo, 0)  # Real-ESRGAN x4plus
    tour.choose(tab.scale_combo, 0)  # 2x - 1080p in, 4K out
    tour.film.hold(0.8)
    elapsed = tour.run_job(tab, "upscale", timeout=1800.0)
    tour.caption(f"Upscale · {_elapsed(elapsed)} of real work, time-lapsed")
    tour.film.hold(2.2)


def act_depth(tour: Tour, work: Path) -> None:
    """Tab 2: a real depth map for the clip."""
    tab = tour.window.depth_tab
    tour.caption("Depth · estimate a depth map for the clip")
    tour.open_tab(1)
    tour.type_path(tab.input_pick, str(CLIP))
    tour.type_path(tab.output_pick, str(work))
    tour.choose(tab.model_combo, 4)  # Depth Anything V2 Small
    tour.film.hold(0.8)
    elapsed = tour.run_job(tab, "depth")
    tour.caption(f"Depth · {_elapsed(elapsed)} of real work, time-lapsed")
    tour.film.hold(2.2)


def act_convert(tour: Tour, work: Path) -> None:
    """Tab 3: the depth video from the last act, warped into side-by-side 3D."""
    tab = tour.window.converter_tab
    depth_video = work / f"{CLIP.stem}_depth.mp4"
    if not depth_video.exists():
        raise SystemExit(
            "convert needs the depth video the depth act produces. Record both "
            "in one session: uv run python tools/record_demo.py depth convert"
        )
    tour.caption("Converter · warp the clip into side-by-side 3D")
    tour.open_tab(2)
    tour.type_path(tab.orig_pick, str(CLIP))
    tour.type_path(tab.depth_pick, str(depth_video))
    tour.type_path(tab.output_pick, str(work))
    tour.film.hold(0.6)
    tour.caption("Converter · 3D strength and convergence set the comfort")
    tab.divergence.setValue(1.4)
    tour.film.hold(0.9)
    tour.tick(tab.aa_check, True)
    tour.caption(None)
    elapsed = tour.run_job(tab, "convert")
    tour.caption(f"Converter · {_elapsed(elapsed)} of real work, time-lapsed")
    tour.film.hold(2.2)


def act_runall(tour: Tour, work: Path) -> None:
    """Tab 4: the whole chain behind one Start button."""
    tab = tour.window.pipeline_tab
    tour.caption("Run all · Depth and Convert chained, one button")
    tour.open_tab(3)
    tour.type_path(tab.input_pick, str(CLIP))
    tour.type_path(tab.output_pick, str(work / "runall"))
    tour.tick(tab.upscale_check, False)
    tour.tick(tab.depth_check, True)
    tour.tick(tab.sbs_check, True)
    tour.caption(None)
    elapsed = tour.run_job(tab, "run all")
    tour.caption(f"Run all · {_elapsed(elapsed)} of real work, time-lapsed")
    tour.film.hold(2.2)


def act_result(tour: Tour, work: Path) -> None:
    """Not a recording of the app - a look at what it produced.

    Every other act films the interface; this one films the files, because the
    interface cannot show the one thing a reader most wants to see. Laid out as
    a 2x2 of equal 1920x1080 cells, which is exactly how the pieces line up:
    the two inputs on top, and below them the two halves of the full-SBS frame
    the converter actually wrote.
    """
    depth_video = work / f"{CLIP.stem}_depth.mp4"
    sbs_video = work / f"{CLIP.stem}_Full_SBS_LRF.mp4"
    missing = [p.name for p in (depth_video, sbs_video) if not p.exists()]
    if missing:
        raise SystemExit(
            f"result needs {', '.join(missing)}, which the depth and convert "
            f"acts produce. Record them in one session: "
            f"uv run python tools/record_demo.py depth convert result"
        )

    from app.core.video_io import VideoReader

    print("  · composing from the real output files")
    cell_w, cell_h = RESULT_WIDTH // 2, (RESULT_WIDTH * 9 // 16) // 2
    label_font = _font(max(11, cell_h // 11))
    originals = VideoReader(CLIP).frames()
    depths = VideoReader(depth_video).frames()
    stereo = VideoReader(sbs_video).frames()

    film = tour.film
    film.frames.clear()
    film.stamps.clear()
    film.forced.clear()

    # Sampled across the whole clip rather than taking its first seconds at full
    # rate, so the loop shows all ten seconds of what was actually converted.
    for index, (orig, depth, pair) in enumerate(zip(originals, depths, stereo)):
        if index % RESULT_STRIDE:
            continue
        half = pair.shape[1] // 2
        cells = [
            ("Original", orig),
            ("Depth", depth),
            ("Left eye", pair[:, :half]),
            ("Right eye", pair[:, half:]),
        ]
        sheet = Image.new("RGB", (cell_w * 2, cell_h * 2))
        for position, (name, array) in enumerate(cells):
            tile = Image.fromarray(array)
            if tile.mode != "RGB":
                tile = tile.convert("RGB")
            tile = tile.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            _label(tile, name, label_font)
            sheet.paste(tile, ((position % 2) * cell_w, (position // 2) * cell_h))
        film.add(sheet, RESULT_DELAY, stamp=index / 25.0)

    print(f"  · {len(film.frames)} frames from {CLIP.stem}")


def _font(size: int):
    from PIL import ImageFont

    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(tile: Image.Image, text: str, font) -> None:
    """A caption burnt into the corner of a panel, on a slab dark enough to stay
    readable over both the bright original and the near-black depth map."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(tile, "RGBA")
    box = draw.textbbox((0, 0), text, font=font)
    pad = 8
    draw.rectangle((0, 0, box[2] + pad * 2, box[3] + pad * 2), fill=(0, 0, 0, 190))
    draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))


ACTS = {
    "upscale": act_upscale,
    "depth": act_depth,
    "convert": act_convert,
    "runall": act_runall,
    "result": act_result,
}


# --------------------------------------------------------------------------- #
# Setup and entry point.
# --------------------------------------------------------------------------- #


def fetch_clip() -> None:
    if CLIP.exists():
        return
    CLIP.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching the demo clip from {CLIP_URL}")
    with urllib.request.urlopen(CLIP_URL, timeout=120) as response:
        CLIP.write_bytes(response.read())
    print(f"  · {CLIP.stat().st_size / 1_000_000:.1f} MB")


def build_window(work: Path):
    """The shipping window, pointed at a scratch config.

    A throwaway config matters: the tour types into every path field, and the
    app saves what it is given. Recording a demo must not overwrite the
    settings of whoever is running it.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel

    from app import config

    config.CONFIG_DIR = work / "config"
    config.CONFIG_FILE = config.CONFIG_DIR / "config.json"

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    qapp = QApplication(sys.argv)
    qapp.setStyle("Fusion")

    from app.ui.icon import app_icon
    from app.ui.theme import apply_scale

    qapp.setWindowIcon(app_icon())
    apply_scale(100)

    from app.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(*WINDOW_SIZE)

    # The tour needs to reach the tabs by name; the window builds them
    # anonymously, so pick them back out of the tab bar.
    widgets = [
        window.centralWidget().widget(i) for i in range(window.centralWidget().count())
    ]
    window.upscale_tab, window.depth_tab, window.converter_tab, window.pipeline_tab = (
        widgets
    )

    # A caption strip, drawn over the window in the app's own surface colours so
    # it reads as part of the recording rather than as a sticker on top of it.
    banner = QLabel(window)
    banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
    banner.setStyleSheet(
        "background: rgba(20,20,22,0.92); color: #f0f0f2;"
        "font-size: 15pt; font-weight: 600; padding: 10px;"
        "border-top: 1px solid rgba(255,255,255,0.18);"
    )
    banner.hide()

    def set_caption(text: str | None) -> None:
        if not text:
            banner.hide()
            return
        banner.setText(text)
        height = banner.sizeHint().height()
        banner.resize(window.width(), height)
        banner.move(0, window.height() - height)
        banner.raise_()
        banner.show()

    window.set_caption = set_caption
    window.show()
    return qapp, window


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    keep = "--keep" in sys.argv
    wanted = args or list(ACTS)
    unknown = [name for name in wanted if name not in ACTS]
    if unknown:
        sys.exit(
            f"unknown act(s): {', '.join(unknown)}. Choose from: {', '.join(ACTS)}"
        )

    fmt = next(
        (a.split("=", 1)[1] for a in sys.argv if a.startswith("--format=")), "mp4"
    )
    formats = {"mp4", "gif"} if fmt == "both" else {fmt}
    if formats - {"mp4", "gif"}:
        sys.exit(f"unknown format {fmt!r}. Pick mp4, gif or both.")
    ffmpeg = find_ffmpeg() if "mp4" in formats else ""

    out_dir = next(
        (a.split("=", 1)[1] for a in sys.argv if a.startswith("--out-dir=")), None
    )
    out = Path(out_dir) if out_dir else MEDIA

    # The MP4 result composite is built from the real output pixels, so it can
    # afford full 1920x1080 cells and, unlike the GIF, every frame at the clip's
    # native rate - which is what keeps it smooth instead of a slideshow.
    global RESULT_WIDTH, RESULT_STRIDE, RESULT_DELAY
    if "mp4" in formats:
        RESULT_WIDTH = 1920
        RESULT_STRIDE = 1
        RESULT_DELAY = round(1000 / RESULT_FPS)

    fetch_clip()
    # --work reuses an earlier --keep directory. `result` composes from files
    # the pipeline already wrote, so retuning its palette or frame rate should
    # not mean running the GPU through the whole clip again.
    reuse = next(
        (a.split("=", 1)[1] for a in sys.argv if a.startswith("--work=")), None
    )
    created = False
    if reuse:
        work = Path(reuse)
        if not work.is_dir():
            sys.exit(f"--work: no such directory: {work}")
        keep = True
    else:
        work = DEMO_WORK
        created = not work.exists()
        work.mkdir(parents=True, exist_ok=True)
    print(f"working in {work}")
    qapp, window = build_window(work)

    try:
        for name in wanted:
            print(f"filming {name}")
            film = Film(qapp, window)
            tour = Tour(film, window)
            if name != "result":
                film.hold(0.6)
            ACTS[name](tour, work)
            tour.caption(None)
            try:
                if "mp4" in formats:
                    fps = RESULT_ENCODE_FPS if name == "result" else MP4_FPS
                    build_mp4(film, out / f"demo-{name}.mp4", ffmpeg, fps)
                if "gif" in formats and name == "result":
                    # Deeper palette and its own size; the composite is pre-trimmed.
                    build_gif(
                        film,
                        out / "demo-result.gif",
                        colors=RESULT_COLORS,
                        cap=False,
                        width=None,
                    )
                elif "gif" in formats:
                    build_gif(film, out / f"demo-{name}.gif")
            finally:
                film.cleanup()
    finally:
        window.close()
        if keep:
            print(f"working files kept in {work}")
        elif created:
            # Only remove a directory this run created, never a pre-existing one.
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
