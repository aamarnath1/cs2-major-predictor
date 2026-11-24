"""Link-preview image (web/og.png) for Reddit, Discord and X: title odds plus the current Pick'em card."""

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WEB = Path(__file__).resolve().parent.parent / "web"
FONTS = [Path("C:/Windows/Fonts/bahnschrift.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
BG, PANEL, INK, MUTE, ACCENT = "#141412", "#1f1f1b", "#f1efe8", "#9a958a", "#ff7a2e"
SLOTS = [("3-0", "GO 3-0", "#51cf66"), ("3-1/3-2", "ADVANCE", "#4dabf7"), ("0-3", "GO 0-3", "#ff6b6b")]


def font(size: int) -> ImageFont.ImageFont:
    for f in FONTS:
        if f.exists():
            return ImageFont.truetype(str(f), size)
    return ImageFont.load_default(size)


def branding() -> dict[str, tuple[str, str]]:
    """team -> (logo asset, hex colour) parsed from team-branding.js."""
    text = (WEB / "team-branding.js").read_text(encoding="utf-8")
    rows = re.findall(r'^\s*"?([^":]+?)"?: \{ asset: "([^"]+)", color: "([^"]+)"', text, re.M)
    return {name: (asset, color) for name, asset, color in rows}


def logo(asset: str, size: int) -> Image.Image | None:
    path = WEB / "team-logos" / f"{asset}.webp"
    try:
        return Image.open(path).convert("RGBA").resize((size, size))
    except OSError:
        return None


def render(data: dict, out: Path = WEB / "og.png") -> Path:
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    brand = branding()
    d.text((48, 40), "PGL MAJOR SINGAPORE 2026 · CS2", font=font(26), fill=ACCENT)
    d.text((48, 74), "WE PLAYED THE MAJOR 5,000 TIMES", font=font(52), fill=INK)

    # left: title odds
    d.rounded_rectangle((48, 160, 548, 560), 18, fill=PANEL)
    d.text((72, 180), "CHANCE TO WIN IT ALL", font=font(24), fill=MUTE)
    top = data["major"][:5]
    best = max(r["champion"] for r in top) or 1
    for i, r in enumerate(top):
        y = 232 + i * 62
        asset, color = brand.get(r["team"], (None, "#888888"))
        color = color if color.startswith("#") else "#bdb7ae"  # CSS-variable colours: neutral grey
        lg = logo(asset, 36) if asset else None
        if lg:
            img.paste(lg, (72, y - 4), lg)
        d.text((120, y), r["team"], font=font(28), fill=INK)
        d.rounded_rectangle((300, y + 6, 300 + int(160 * r["champion"] / best), y + 26), 5, fill=color)
        d.text((470, y), f"{round(100 * r['champion'])}%", font=font(28), fill=INK)

    # right: the current stage card
    st = next((s for s in data["stages"] if s["status"] != "complete" and s.get("picks")), None)
    if st:
        d.rounded_rectangle((580, 160, 1152, 560), 18, fill=PANEL)
        d.text((604, 180), f"STAGE {st['stage']} PICK'EM CARD", font=font(24), fill=MUTE)
        y = 224
        for slot, label, color in SLOTS:
            d.rectangle((604, y + 4, 610, y + 30), fill=color)
            d.text((622, y), label, font=font(26), fill=INK)
            names = st["picks"].get(slot, [])
            status = (st.get("pick_status") or {}).get(slot, {})
            y = draw_names(d, names, status, 622, y + 38, 1130) + 18
        d.text((604, 520), f"{round(100 * st['p_at_least_5'])}% chance of 5+ correct", font=font(24), fill=ACCENT)

    d.text((48, 580), f"Updated {data['generated']} · aamarnath1.github.io/cs2-major-predictor", font=font(22), fill=MUTE)
    img.save(out, optimize=True)
    return out


def draw_names(d: ImageDraw.ImageDraw, names: list[str], status: dict, x0: int, y: int, right: int) -> int:
    """Write team names left to right, wrapping at `right`; wrong picks struck through, correct ones green.

    Returns the y below the last line."""
    f, x, line_h = font(24), x0, 30
    for i, t in enumerate(names):
        text = t + ("," if i < len(names) - 1 else "")
        w = d.textlength(text + " ", font=f)
        if x + w > right and x > x0:
            x, y = x0, y + line_h
        st = status.get(t)
        d.text((x, y), text, font=f, fill="#64d979" if st == "hit" else MUTE if st == "miss" else INK)
        if st == "miss":
            d.line((x, y + 15, x + d.textlength(t, font=f), y + 15), fill=MUTE, width=2)
        x += w
    return y + line_h
