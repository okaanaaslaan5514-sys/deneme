#!/usr/bin/env python3
"""
Micro script to read MP (mana) values from the screen and report percentage.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

try:
    from .bar_reader import (
        ROI,
        build_common_parser,
        compute_percentage,
        configure_tesseract,
        read_bar_from_image,
        read_bar_from_screen,
    )
except ImportError:  # pragma: no cover - script fallback
    from bar_reader import (  # type: ignore[assignment]
        ROI,
        build_common_parser,
        compute_percentage,
        configure_tesseract,
        read_bar_from_image,
        read_bar_from_screen,
    )

MP_ROI = ROI(left=108, top=72, width=118, height=25)


def format_output(current: int, maximum: int) -> str:
    percentage = compute_percentage(current, maximum)
    return f"MP: {current}/{maximum} ({percentage:.1f}%)"


def read_once(image: Optional[Path]) -> str:
    if image:
        current, maximum = read_bar_from_image(image)
    else:
        current, maximum = read_bar_from_screen(MP_ROI)
    return format_output(current, maximum)


def main() -> None:
    parser = build_common_parser("MP bar OCR okuyucusu")
    args = parser.parse_args()

    configure_tesseract(args.tesseract)

    if args.once or args.from_image:
        try:
            print(read_once(args.from_image))
        except Exception as exc:
            print(f"MP okunamadı: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    interval = max(args.interval, 0.1)
    limit = args.limit or sys.maxsize

    try:
        for idx in range(limit):
            try:
                print(read_once(args.from_image), flush=True)
            except Exception as exc:
                print(f"MP okunamadı: {exc}", file=sys.stderr, flush=True)
            time.sleep(interval)
            if args.from_image:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
