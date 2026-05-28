"""
Data fetching module - fully parameterized by cfg.

Provides functions to fetch lottery draw data from ZHCW (HTML scraping)
and 500.com (CSV download), then save as a unified CSV using cfg's
column naming conventions (cfg.main_cols, cfg.sub_cols).
"""
import csv
import io
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.helpers import get_logger

# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

#: User-agent to mimic a browser request
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_HEADERS = {"User-Agent": _DEFAULT_UA}

#: Requests timeout (seconds)
_TIMEOUT = 30

#: Max retries for transient failures
_MAX_RETRIES = 3


def _get_session() -> requests.Session:
    """Return a requests.Session with default headers."""
    sess = requests.Session()
    sess.headers.update(_HEADERS)
    return sess


# ---------------------------------------------------------------------------
# ZHCW scraper (通用爬虫)
# ---------------------------------------------------------------------------

# Each ball in ZHCW HTML is inside:  <em class="ball1|ball2|...">NUMBER</em>
# DLT:  5 front balls (ball1..ball5) + 2 back balls (ball2..ball2)
# SSQ:  6 red  balls                   + 1 blue ball
# We rely on cfg to know how many to expect.

_BALL_SELECTOR = "em.ball1, em.ball2, em.ball3"


def _parse_zhcw_numbers(
    soup: BeautifulSoup,
    cfg,
) -> List[Tuple[str, List[int], List[int]]]:
    """Parse ZHCW page soup into list of (period, main_nums, sub_nums).

    Strategy: find all ``<em class="ball*">NN</em>`` sequences grouped
    by draw row, then split by cfg.main_count / cfg.sub_count.
    """
    logger = get_logger(cfg)
    results: List[Tuple[str, List[int], List[int]]] = []

    # --- Find issue/period numbers ---
    # ZHCW typically puts issue number in <td> or <span> near the balls.
    # Look for all <em class="ball1"> ... </em> and count them.
    ball_ems = soup.select("em[class^='ball']")

    if not ball_ems:
        logger.warning("No ball elements found on ZHCW page (selector: em[class^='ball'])")
        return results

    # Try to group balls by draw
    # Method: each draw row has main_count + sub_count consecutive <em> elements.
    total_per_draw = cfg.main_count + cfg.sub_count

    # Also look for issue numbers in nearby table cells
    # ZHCW structure: <tr> with <td>issue</td> then <td>ball1 ball2 ...</td>
    rows = soup.select("tr")
    parsed = 0
    skipped = 0

    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        # Issue number: usually first td, text contains digits
        issue_text = tds[0].get_text(strip=True)
        # Extract numeric issue
        issue_match = re.search(r"(\d{5,})", issue_text)
        if not issue_match:
            continue
        issue = issue_match.group(1)

        # Collect all ball numbers from this row
        row_balls: List[int] = []
        for td in tds[1:]:
            for em in td.find_all("em", class_=re.compile(r"^ball")):
                try:
                    row_balls.append(int(em.get_text(strip=True)))
                except ValueError:
                    continue

        if len(row_balls) != total_per_draw:
            # Try fallback: parse all em in the entire tr
            all_ems = tr.find_all("em", class_=re.compile(r"^ball"))
            row_balls = []
            for em in all_ems:
                try:
                    row_balls.append(int(em.get_text(strip=True)))
                except ValueError:
                    continue
            if len(row_balls) != total_per_draw:
                skipped += 1
                continue

        main_nums = sorted(row_balls[: cfg.main_count])
        sub_nums = sorted(row_balls[cfg.main_count :])
        results.append((issue, main_nums, sub_nums))
        parsed += 1

    logger.info(
        "ZHCW parsed %d draws (skipped %d rows) for %s",
        parsed,
        skipped,
        cfg.name,
    )
    return results


def _zhcw_url_for_page(cfg, page: int = 1) -> str:
    """Build ZHCW URL for a specific page.

    ZHCW uses format like:
      https://www.zhcw.com/kjxx/dlt/?pageNo=1
      https://www.zhcw.com/kjxx/ssq/?pageNo=1
    """
    base = cfg.zhcw_path.rstrip("/")
    return f"{base}/?pageNo={page}"


def fetch_from_zhcw(
    cfg,
    max_pages: int = 10,
    max_draws: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch draw history from ZHCW (中国福利彩票官网).

    Scrapes the HTML pages and returns a DataFrame with columns
    based on cfg.main_cols and cfg.sub_cols.

    Parameters
    ----------
    cfg : LotteryConfig instance.
    max_pages : max pages to scrape (default 10, each page ~50 draws).
    max_draws : optional limit on total draws to return.

    Returns
    -------
    DataFrame with columns [period, cfg.main_cols..., cfg.sub_cols...]
    """
    logger = get_logger(cfg)
    logger.info(
        "Fetching %s data from ZHCW (%s) ...", cfg.name, cfg.zhcw_path
    )

    all_draws: List[Tuple[str, List[int], List[int]]] = []
    session = _get_session()

    for page in range(1, max_pages + 1):
        if max_draws and len(all_draws) >= max_draws:
            break

        url = _zhcw_url_for_page(cfg, page)
        logger.debug("Fetching ZHCW page %d: %s", page, url)

        try:
            resp = session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as exc:
            logger.warning("Failed to fetch ZHCW page %d: %s", page, exc)
            time.sleep(2)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        page_draws = _parse_zhcw_numbers(soup, cfg)
        all_draws.extend(page_draws)

        if len(page_draws) == 0:
            logger.info("No more draws found on ZHCW page %d, stopping", page)
            break

        # Be polite — don't hammer the server
        time.sleep(1.5)

    if not all_draws:
        logger.warning("No data fetched from ZHCW for %s", cfg.name)
        return pd.DataFrame()

    # Build DataFrame
    rows = []
    for period, main_nums, sub_nums in all_draws:
        row: dict = {"period": period}
        for i, col in enumerate(cfg.main_cols):
            row[col] = main_nums[i] if i < len(main_nums) else 0
        for i, col in enumerate(cfg.sub_cols):
            row[col] = sub_nums[i] if i < len(sub_nums) else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    if max_draws and len(df) > max_draws:
        df = df.head(max_draws)

    logger.info(
        "Fetched %d draws from ZHCW for %s",
        len(df),
        cfg.name,
    )
    return df


# ---------------------------------------------------------------------------
# 500.com CSV downloader
# ---------------------------------------------------------------------------


def fetch_from_500com(
    cfg,
    start: str = "",
    end: str = "",
    max_draws: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch draw data from 500.com CSV endpoint.

    500.com provides a CSV export with columns defined by the source.
    DLT vs SSQ CSV column structure differs, so we remap to cfg.main_cols/sub_cols.

    Parameters
    ----------
    cfg : LotteryConfig instance.
    start : start date string (e.g. ``"2024-01-01"``). Empty = earliest.
    end   : end date string. Empty = latest.
    max_draws : max rows to keep.

    Returns
    -------
    DataFrame with columns [period, cfg.main_cols..., cfg.sub_cols...]
    """
    logger = get_logger(cfg)
    url_template = cfg.data_sources.get("500com", "")

    if not url_template:
        logger.warning("No 500com URL configured for %s", cfg.name)
        return pd.DataFrame()

    # Build URL with start/end
    url = url_template.format(start=start, end=end)
    logger.info("Fetching %s data from 500com ...", cfg.name)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except requests.RequestException as exc:
        logger.error("Failed to fetch 500com data: %s", exc)
        return pd.DataFrame()

    # Parse CSV
    try:
        lines = resp.text.strip().splitlines()
        if not lines:
            logger.warning("Empty response from 500com")
            return pd.DataFrame()

        reader = csv.reader(lines)
        raw_rows = list(reader)
    except Exception as exc:
        logger.error("Failed to parse 500com CSV: %s", exc)
        return pd.DataFrame()

    if len(raw_rows) < 2:
        logger.warning("Not enough rows from 500com (got %d)", len(raw_rows))
        return pd.DataFrame()

    # Determine column mapping based on source format
    # 500.com DLT columns: 期号, 开奖日期, front_1..front_5, back_1..back_2, sales, pool, ...
    # 500.com SSQ columns: 期号, 开奖日期, red_1..red_6, blue_1, sales, pool, ...
    header = raw_rows[0]
    data_rows = raw_rows[1:]

    # Try to find period column and ball columns
    period_idx: Optional[int] = None
    ball_idxs: List[int] = []
    ball_count = cfg.main_count + cfg.sub_count

    for idx, col_name in enumerate(header):
        col_clean = col_name.strip()
        if col_clean in ("期号", "期数", "issue", "period"):
            period_idx = idx
        # Look for ball-number columns: anything that looks like a number column
        # after the date column, before sales/pool columns

    # Fallback: first column is usually the issue number
    if period_idx is None:
        period_idx = 0

    # Find ball columns: they should be numeric-only columns starting from column 2
    # (skip period + date) until we hit non-numeric or known label columns
    ball_start = period_idx + 1  # skip period column
    # Skip date column if present
    date_keywords = ("date", "日期", "开奖日期", "draw_date")
    if ball_start < len(header) and header[ball_start].strip() in date_keywords:
        ball_start += 1
    # Also skip second date column
    if (
        ball_start < len(header)
        and header[ball_start].strip() in date_keywords
    ):
        ball_start += 1

    # The next ball_count columns should be the ball numbers
    ball_idxs = list(range(ball_start, min(ball_start + ball_count, len(header))))

    # Build DataFrame
    parsed_rows: List[dict] = []
    for row in data_rows:
        if len(row) <= max(ball_idxs + [period_idx]) if period_idx is not None else False:
            continue

        period_val = row[period_idx].strip() if period_idx is not None else ""
        numbers = []
        for bi in ball_idxs:
            try:
                numbers.append(int(row[bi].strip()))
            except (ValueError, IndexError):
                numbers.append(0)

        if len(numbers) < ball_count:
            continue

        main_nums = sorted(numbers[: cfg.main_count])
        sub_nums = sorted(numbers[cfg.main_count :])

        entry: dict = {"period": period_val}
        for i, col in enumerate(cfg.main_cols):
            entry[col] = main_nums[i] if i < len(main_nums) else 0
        for i, col in enumerate(cfg.sub_cols):
            entry[col] = sub_nums[i] if i < len(sub_nums) else 0
        parsed_rows.append(entry)

    if not parsed_rows:
        logger.warning("No draw rows could be parsed from 500com CSV")
        return pd.DataFrame()

    df = pd.DataFrame(parsed_rows)
    if max_draws and len(df) > max_draws:
        df = df.head(max_draws)

    logger.info(
        "Fetched %d draws from 500com for %s",
        len(df),
        cfg.name,
    )
    return df


# ---------------------------------------------------------------------------
# Main update entry point
# ---------------------------------------------------------------------------


def update_data(
    cfg,
    force_refresh: bool = False,
    source: str = "auto",
    max_draws: Optional[int] = None,
) -> pd.DataFrame:
    """Main entry point: fetch lottery data and save to cfg.history_csv.

    Priority / workflow:
      1. If ``cfg.history_csv`` exists and ``force_refresh`` is False,
         load from file.
      2. If ``source="auto"``, try ZHCW first (more reliable), fall
         back to 500com.
      3. Save to ``cfg.history_csv`` and return the DataFrame.

    Parameters
    ----------
    cfg : LotteryConfig instance.
    force_refresh : if True, re-fetch from remote even if local file exists.
    source : ``"auto"`` (default), ``"zhcw"``, or ``"500com"``.
    max_draws : optional cap on number of draws to keep.

    Returns
    -------
    DataFrame with columns [period, cfg.main_cols..., cfg.sub_cols...]
    """
    logger = get_logger(cfg)
    csv_path = Path(cfg.history_csv)

    # ---- 1. Load from cache if available ----
    if not force_refresh and csv_path.exists():
        logger.info(
            "Loading cached %s data from %s",
            cfg.name,
            csv_path,
        )
        try:
            df = pd.read_csv(csv_path)
            # Ensure required columns exist
            required_cols = ["period"] + cfg.main_cols + cfg.sub_cols
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.warning(
                    "Missing columns in cached CSV: %s. Re-fetching.", missing
                )
            else:
                logger.info(
                    "Loaded %d draws from cache for %s",
                    len(df),
                    cfg.name,
                )
                return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
            logger.warning("Failed to read cached CSV: %s. Re-fetching.", exc)

    # ---- 2. Fetch from remote ----
    logger.info("Fetching %s data from remote source ...", cfg.name)
    df = pd.DataFrame()

    if source == "auto" or source == "zhcw":
        df = fetch_from_zhcw(cfg, max_draws=max_draws)
        if df.empty and source == "auto":
            logger.info("ZHCW returned no data, trying 500com fallback ...")
            df = fetch_from_500com(cfg, max_draws=max_draws)
    elif source == "500com":
        df = fetch_from_500com(cfg, max_draws=max_draws)
        if df.empty:
            logger.info("500com returned no data, trying ZHCW fallback ...")
            df = fetch_from_zhcw(cfg, max_draws=max_draws)
    else:
        logger.error("Unknown data source: %s", source)
        return pd.DataFrame()

    if df.empty:
        logger.error("Failed to fetch any data for %s from any source", cfg.name)
        return df

    # ---- 3. Ensure numeric types for ball columns ----
    for col in cfg.main_cols + cfg.sub_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ---- 4. Sort by period descending (newest first) ----
    if "period" in df.columns:
        df = df.sort_values("period", ascending=False).reset_index(drop=True)

    # ---- 5. Save to CSV ----
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(
            "Saved %d %s draws to %s",
            len(df),
            cfg.name,
            csv_path,
        )
    except OSError as exc:
        logger.error("Failed to save CSV %s: %s", csv_path, exc)

    return df
