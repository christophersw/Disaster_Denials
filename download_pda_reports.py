"""
Title: download_pda_reports.py — FEMA Preliminary Damage Assessment (PDA) report downloader
Description:
    Downloads every PDA report listed on FEMA's paginated "Reports" index page
    into a folder tree organized by report type and posted year:

        data/pdfs/<ReportType>/<Year>/<filename>.pdf

    The crawler works one index page at a time: it fetches a page, then
    immediately downloads all PDFs linked on that page before moving to the
    next. This matters because FEMA's *dynamic* index pages are slow and
    throttled (tens of seconds each), while the *static* PDFs download quickly;
    interleaving means every slow page fetch is immediately paid off, and a
    crash never loses more than the page in flight.

    Robustness:
      - Per-page checkpoint (data/.progress): completed pages are skipped on
        re-run, so the job is fully resumable.
      - Existing PDFs are skipped (size > 0), so downloads are resumable too.
      - The manifest (data/manifest.csv) is appended after each page, recording
        every report's metadata and download status.

Changelog:
    2026-05-24: Default scope is now ALL report types; denials-only is opt-in
                via --denials-only (was --all to broaden).
    2026-05-24: Added CLI (argparse), a tqdm progress bar, and structured
                logging to console + rotating-free file (data/download.log).
    2026-05-24: Redesigned to interleave per-page crawl+download with
                checkpointing; replaced the discover-all-then-download flow.
    2026-05-24: Initial version — index crawl, type/year foldering, manifest.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from urllib.parse import unquote

from curl_cffi import requests
from tqdm import tqdm

logger = logging.getLogger("pda_downloader")


# --- Configuration -----------------------------------------------------------

INDEX_URL = (
    "https://www.fema.gov/disaster/how-declared/"
    "preliminary-damage-assessments/reports"
)
FEMA_ORIGIN = "https://www.fema.gov"

OUTPUT_ROOT = Path("data/pdfs")
MANIFEST_PATH = Path("data/manifest.csv")
CHECKPOINT_PATH = Path("data/.progress")
LOG_PATH = Path("data/download.log")

IMPERSONATE_PROFILE = "chrome"

# Politeness / robustness knobs.
INDEX_TIMEOUT_SECONDS = 120      # dynamic index pages are slow
PDF_TIMEOUT_SECONDS = 90         # static PDFs are usually fast
DELAY_BETWEEN_PDFS_SECONDS = 0.5
DELAY_BETWEEN_PAGES_SECONDS = 1.0
MAX_RETRIES_PER_REQUEST = 4
RETRY_BACKOFF_SECONDS = 4

# Safety cap on index pages to crawl (FEMA currently has ~69 pages).
MAX_INDEX_PAGES = 200


# --- Data model --------------------------------------------------------------

@dataclass
class ReportRecord:
    """
    One PDA report discovered on the index page.

    Attributes:
        index_page: The index page number this report was found on.
        title: The bold title text, e.g. "Hawaii; FEMA-4909-DR".
        state: Best-effort state/territory/tribe name parsed from the title.
        declaration_string: Text after the state in the title (e.g. "FEMA-4909-DR"
            or an incident description ending in "- Denial").
        posted_date: ISO datetime string from the row's <time datetime> attribute.
        posted_year: Four-digit year derived from posted_date (folder bucket).
        report_type: Classified category used as the top-level folder
            (MajorDisaster, Emergency, Expedited, Denials, AppealDenials, Other).
        url: Absolute URL of the report PDF.
        filename: Basename of the PDF as stored on disk.
        local_path: Path the PDF was/will be saved to (relative to repo root).
        status: Download outcome ("downloaded", "skipped_exists", or "error: ...").
    """

    index_page: int
    title: str
    state: str
    declaration_string: str
    posted_date: str
    posted_year: str
    report_type: str
    url: str
    filename: str
    local_path: str
    status: str


# --- HTTP helpers ------------------------------------------------------------

def fetch(session: requests.Session, url: str, timeout: int) -> requests.Response:
    """
    GET a URL through the Chrome-impersonating session with retries.

    Args:
        session: A curl_cffi Session configured to impersonate Chrome.
        url: Absolute URL to fetch.
        timeout: Per-attempt timeout in seconds.

    Returns:
        The successful Response object (HTTP 200).

    Raises:
        RuntimeError: If all retry attempts fail or return a non-200 status.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES_PER_REQUEST + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            last_error = f"HTTP {response.status_code}"
        except Exception as exception:  # network/TLS/timeout errors
            last_error = repr(exception)
        if attempt < MAX_RETRIES_PER_REQUEST:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


# --- Index parsing -----------------------------------------------------------

# Each report row contains a bold title, a <time datetime=...> stamp, and a PDF
# link. We split the page on the bold-title marker so each chunk holds exactly
# one report, which is far more robust than one big greedy regex.
ROW_SPLIT_PATTERN = re.compile(r'<p class="bold-title">')
TITLE_PATTERN = re.compile(r"^(.*?)</p>", re.DOTALL)
DATETIME_PATTERN = re.compile(r'<time[^>]*datetime="([^"]+)"')
PDF_HREF_PATTERN = re.compile(r'href="([^"]+\.pdf[^"]*)"', re.IGNORECASE)


def discover_last_index_page(html: str) -> int:
    """
    Find the highest page number referenced by the index pager.

    Args:
        html: Raw HTML of an index page (the pager appears on every page).

    Returns:
        The largest ?page=N value found, or 0 if no pager is present.
    """
    page_numbers = [int(n) for n in re.findall(r"[?&]page=(\d+)", html)]
    return max(page_numbers) if page_numbers else 0


def strip_tags(text: str) -> str:
    """Remove any HTML tags from a fragment, returning plain text."""
    return re.sub(r"<[^>]+>", "", text)


def parse_index_page(html: str) -> list[tuple[str, str, str]]:
    """
    Extract (title, datetime, pdf_href) tuples from one index page.

    Args:
        html: Raw HTML of an index page.

    Returns:
        A list of tuples, one per report row that has a PDF link.
    """
    rows: list[tuple[str, str, str]] = []
    for chunk in ROW_SPLIT_PATTERN.split(html)[1:]:
        title_match = TITLE_PATTERN.search(chunk)
        datetime_match = DATETIME_PATTERN.search(chunk)
        href_match = PDF_HREF_PATTERN.search(chunk)
        if not (title_match and href_match):
            continue
        title = re.sub(r"\s+", " ", strip_tags(title_match.group(1))).strip()
        posted_date = datetime_match.group(1) if datetime_match else ""
        rows.append((title, posted_date, href_match.group(1)))
    return rows


# --- Classification ----------------------------------------------------------

def classify_report_type(filename: str, declaration_string: str) -> str:
    """
    Bucket a report into a top-level folder category.

    Classification is by priority: appeal denials and denials first (these are
    the research focus and may otherwise look like ordinary reports), then
    expedited declarations, then emergency vs. major-disaster from the
    declaration string.

    Args:
        filename: The PDF basename (e.g. "PDAReport_FEMA4900DRexpedited-LA.pdf").
        declaration_string: Text parsed from the title (e.g. "FEMA-4900-DR").

    Returns:
        One of: AppealDenials, Denials, Expedited, Emergency, MajorDisaster, Other.
    """
    haystack = f"{filename} {declaration_string}".lower()
    if "appealdenial" in haystack or "appeal denial" in haystack or "denial of appeal" in haystack:
        return "AppealDenials"
    if "denial" in haystack:
        return "Denials"
    if "expedited" in haystack:
        return "Expedited"
    # Declaration type code: "...-EM..." (emergency) vs "...-DR..." (major disaster).
    if re.search(r"\bEM\b|-EM-|\d+EM", declaration_string, re.IGNORECASE):
        return "Emergency"
    if re.search(r"\bDR\b|-DR-|\d+DR", declaration_string, re.IGNORECASE):
        return "MajorDisaster"
    return "Other"


# A FEMA disaster number in a filename (e.g. "FEMA4909DR", "FEMA4900DRexpedited")
# marks an approved/expedited declaration — not a denial. Denial and appeal-denial
# filenames carry no such number (e.g. "PDAReport_Denial-KS.pdf").
NUMBERED_DECLARATION_PATTERN = re.compile(r"FEMA\d{2,}", re.IGNORECASE)


def is_numbered_declaration(filename: str) -> bool:
    """
    Report whether a filename belongs to a numbered (non-denial) declaration.

    Args:
        filename: The PDF basename.

    Returns:
        True if the name contains a FEMA disaster number (so it should be
        skipped when collecting denials only); False otherwise.
    """
    return NUMBERED_DECLARATION_PATTERN.search(filename) is not None


def parse_title(title: str) -> tuple[str, str]:
    """
    Split a row title into state and declaration string.

    Titles look like "Hawaii; FEMA-4909-DR" or "Colorado; ... - Denial".

    Args:
        title: The cleaned title text.

    Returns:
        (state, declaration_string). The second element may be empty if the
        title has no semicolon separator.
    """
    if ";" in title:
        state, _, remainder = title.partition(";")
        return state.strip(), remainder.strip()
    return title.strip(), ""


def year_from_datetime(iso_datetime: str) -> str:
    """
    Extract a four-digit year from an ISO datetime, or 'unknown'.

    Args:
        iso_datetime: A string like "2026-05-06T12:00:00Z" (may be empty).

    Returns:
        The leading four-digit year, or "unknown" if none is present.
    """
    match = re.match(r"(\d{4})", iso_datetime)
    return match.group(1) if match else "unknown"


def normalize_url(url: str) -> str:
    """
    Undo accidental double percent-encoding in a URL.

    FEMA's index HTML sometimes publishes hrefs with an extra encoding layer
    for non-ASCII filenames — e.g. the "ñ" in "Luiseño" appears as "%25C3%25B1"
    (the "%" of "%C3%B1" itself encoded as "%25"). Requesting that literally
    404s, while one level of decoding ("%C3%B1") resolves correctly. We decode
    repeatedly only while the "%25" double-encoding signature is present, so
    correctly-encoded URLs are left untouched.

    Args:
        url: The URL as published in the page (possibly double-encoded).

    Returns:
        The URL with any extra "%25" encoding layer removed.
    """
    while "%25" in url:
        url = unquote(url)
    return url


def absolute_url(href: str) -> str:
    """Resolve a possibly-relative PDF href against the FEMA origin and
    normalize away any double percent-encoding."""
    absolute = href if href.startswith("http") else FEMA_ORIGIN + href
    return normalize_url(absolute)


def build_record(index_page: int, title: str, posted_date: str, href: str) -> ReportRecord:
    """
    Turn one parsed index row into a fully-populated ReportRecord.

    Args:
        index_page: The page number the row came from.
        title: Cleaned row title.
        posted_date: ISO datetime from the row's <time> element.
        href: PDF link (relative or absolute).

    Returns:
        A ReportRecord with status "pending".
    """
    url = absolute_url(href)
    state, declaration_string = parse_title(title)
    # Use a fully-decoded basename on disk so non-ASCII names (e.g. "Luiseño")
    # are human-readable; the URL keeps its working percent-encoding.
    filename = unquote(url.rsplit("/", 1)[-1])
    report_type = classify_report_type(filename, declaration_string)
    posted_year = year_from_datetime(posted_date)
    local_path = OUTPUT_ROOT / report_type / posted_year / filename
    return ReportRecord(
        index_page=index_page,
        title=title,
        state=state,
        declaration_string=declaration_string,
        posted_date=posted_date,
        posted_year=posted_year,
        report_type=report_type,
        url=url,
        filename=filename,
        local_path=str(local_path),
        status="pending",
    )


# --- Download ----------------------------------------------------------------

def download_report(session: requests.Session, record: ReportRecord) -> None:
    """
    Download a single report PDF, skipping it if already present.

    Mutates record.status in place to reflect the outcome.

    Args:
        session: The Chrome-impersonating session.
        record: The report to download; its local_path determines destination.
    """
    destination = Path(record.local_path)
    if destination.exists() and destination.stat().st_size > 0:
        record.status = "skipped_exists"
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = fetch(session, record.url, PDF_TIMEOUT_SECONDS)
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and response.content[:4] != b"%PDF":
            record.status = f"error: not a pdf (content-type={content_type})"
            return
        destination.write_bytes(response.content)
        record.status = "downloaded"
    except Exception as exception:
        record.status = f"error: {exception}"


# --- Checkpoint & manifest persistence ---------------------------------------

def load_completed_pages() -> set[int]:
    """Return the set of index page numbers already fully processed."""
    if not CHECKPOINT_PATH.exists():
        return set()
    return {
        int(line) for line in CHECKPOINT_PATH.read_text().split() if line.strip().isdigit()
    }


def mark_page_completed(page: int) -> None:
    """Append a page number to the checkpoint file."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{page}\n")


def append_manifest_rows(records: list[ReportRecord]) -> None:
    """
    Append report records to the manifest CSV, writing a header if new.

    Args:
        records: Records to append (final statuses set).
    """
    if not records:
        return
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(ReportRecord)]
    write_header = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def load_manifest_records() -> list[ReportRecord]:
    """
    Read all records back from the manifest CSV.

    Returns:
        The records in file order. Empty if no manifest exists.
    """
    if not MANIFEST_PATH.exists():
        return []
    field_names = [f.name for f in fields(ReportRecord)]
    records: list[ReportRecord] = []
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values = {name: row.get(name, "") for name in field_names}
            values["index_page"] = int(values["index_page"] or 0)
            records.append(ReportRecord(**values))
    return records


def rewrite_manifest(records: list[ReportRecord]) -> None:
    """
    Overwrite the manifest CSV with the given records.

    Used after a retry pass so updated statuses replace the originals rather
    than appending duplicate rows.

    Args:
        records: The complete record set to persist.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(ReportRecord)]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def retry_errors(session: requests.Session) -> int:
    """
    Retry only the manifest rows whose status is an error, in place.

    Index pages are *not* re-crawled — this reuses the URLs already recorded in
    the manifest, so it is fast even though the dynamic index pages are slow.
    Each errored URL is re-normalized (in case it was recorded before the
    double-encoding fix) and re-downloaded; the manifest is rewritten with the
    updated statuses.

    Args:
        session: The Chrome-impersonating session.

    Returns:
        Process exit code (0 if no errors remain, 1 otherwise).
    """
    records = load_manifest_records()
    failed = [r for r in records if r.status.startswith("error")]
    if not records:
        logger.warning("No manifest found; nothing to retry.")
        return 0
    logger.info("Retrying %d errored report(s) out of %d.", len(failed), len(records))

    progress_bar = tqdm(total=len(failed), unit="rpt", desc="Retrying", dynamic_ncols=True)
    healed = 0
    for record in failed:
        record.url = normalize_url(record.url)
        download_report(session, record)
        if not record.status.startswith("error"):
            healed += 1
            logger.info("recovered: %s [%s]", record.filename, record.status)
        else:
            logger.warning("still failing: %s — %s", record.filename, record.status)
        progress_bar.update(1)
        time.sleep(DELAY_BETWEEN_PDFS_SECONDS)
    progress_bar.close()

    rewrite_manifest(records)
    remaining = sum(1 for r in records if r.status.startswith("error"))
    logger.info("Retry complete: %d recovered, %d still failing.", healed, remaining)
    print(f"\nRetry done. Recovered {healed}, still failing {remaining}. "
          f"Manifest: {MANIFEST_PATH}")
    return 1 if remaining else 0


# --- Orchestration -----------------------------------------------------------

def process_page(
    session: requests.Session,
    page: int,
    html: str,
    denials_only: bool,
) -> tuple[list[ReportRecord], int]:
    """
    Download the relevant PDFs linked on one index page.

    When denials_only is True, files whose name contains a FEMA disaster number
    (approved/expedited declarations) are skipped entirely — not downloaded and
    not added to the manifest. Per-file outcomes are logged (DEBUG for success,
    WARNING for errors).

    Args:
        session: The Chrome-impersonating session.
        page: The index page number (for record provenance).
        html: Raw HTML of that index page.
        denials_only: When True, skip numbered (non-denial) declarations.

    Returns:
        A tuple of (records handled with statuses set, count of numbered
        declarations skipped).
    """
    rows = parse_index_page(html)
    records: list[ReportRecord] = []
    skipped_numbered = 0
    for title, posted_date, href in rows:
        filename = unquote(absolute_url(href).rsplit("/", 1)[-1])
        if denials_only and is_numbered_declaration(filename):
            skipped_numbered += 1
            logger.debug("page %s: skip numbered declaration %s", page, filename)
            continue

        record = build_record(page, title, posted_date, href)
        download_report(session, record)
        records.append(record)

        relative = f"{record.report_type}/{record.posted_year}/{record.filename}"
        if record.status.startswith("error"):
            logger.warning("page %s: %s — %s", page, relative, record.status)
        else:
            logger.debug("page %s: %s [%s]", page, relative, record.status)

        if record.status == "downloaded":
            time.sleep(DELAY_BETWEEN_PDFS_SECONDS)
    return records, skipped_numbered


def setup_logging(verbose: bool) -> None:
    """
    Configure logging to a detail file and a concise console stream.

    The file handler (data/download.log) records everything at DEBUG with
    timestamps — a full audit trail of every report. The console handler stays
    quiet so it does not fight the progress bar: WARNING and above by default,
    or INFO and above with --verbose. Console records are emitted via
    tqdm.write so they do not corrupt the bar.

    Args:
        verbose: When True, also show per-file INFO lines on the console.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(console_handler)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm.write so the bar stays intact."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write one formatted log record above the active progress bar."""
        try:
            tqdm.write(self.format(record))
        except Exception:  # pragma: no cover - never let logging crash a run
            self.handleError(record)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed arguments with .verbose, .no_progress, and .max_pages.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download all FEMA Preliminary Damage Assessment (PDA) report PDFs, "
            "organized into data/pdfs/<Type>/<Year>/. Resumable: re-running "
            "skips pages already completed (data/.progress) and files already "
            "present."
        )
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-file progress on the console (always written to the log).",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar (useful for non-interactive logs).",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, metavar="N",
        help="Only crawl the first N index pages (handy for testing).",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Skip crawling; re-download only the failed rows in the manifest "
             "and rewrite it with updated statuses.",
    )
    parser.add_argument(
        "--denials-only", action="store_true",
        help="Only download denials and appeal denials (skip any file whose "
             "name contains a FEMA disaster number). By default ALL report "
             "types are downloaded.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Entry point: crawl each index page and download its PDFs, page by page.

    Args:
        argv: Optional argument list for testing (defaults to sys.argv).

    Returns:
        Process exit code (0 if no download errors, 1 otherwise).
    """
    args = parse_args(argv)
    setup_logging(args.verbose)

    session = requests.Session(impersonate=IMPERSONATE_PROFILE)

    if args.retry_errors:
        return retry_errors(session)

    completed_pages = load_completed_pages()
    if completed_pages:
        logger.info("Resuming: %d pages already completed.", len(completed_pages))

    # Fetch page 0 once to learn how many pages exist.
    logger.info("Fetching index page 0 to detect total pages...")
    first_html = fetch(session, INDEX_URL, INDEX_TIMEOUT_SECONDS).text
    last_page = min(discover_last_index_page(first_html), MAX_INDEX_PAGES)
    if args.max_pages is not None:
        last_page = min(last_page, args.max_pages - 1)
    pages_to_do = [p for p in range(last_page + 1) if p not in completed_pages]
    denials_only = args.denials_only
    scope = "denials only" if denials_only else "all report types"
    logger.info(
        "Index has pages 0..%d; %d pages remaining (scope: %s).",
        last_page, len(pages_to_do), scope,
    )

    all_errors: list[ReportRecord] = []
    total_downloaded = 0
    total_skipped = 0
    total_filtered = 0

    # Progress is tracked per index page, since under denials-only most reports
    # on a page are filtered out and a per-report total would be meaningless.
    progress_bar = None if args.no_progress else tqdm(
        total=len(pages_to_do), unit="page", desc="Index pages", dynamic_ncols=True,
    )

    for page in pages_to_do:
        logger.info("page %d/%d: fetching index...", page, last_page)
        try:
            html = first_html if page == 0 else fetch(
                session, f"{INDEX_URL}?page={page}", INDEX_TIMEOUT_SECONDS
            ).text
        except RuntimeError as error:
            logger.warning(
                "could not fetch index page %d (%s); leaving un-checkpointed "
                "so a re-run retries it.", page, error,
            )
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        records, page_filtered = process_page(session, page, html, denials_only)
        append_manifest_rows(records)
        mark_page_completed(page)

        page_downloaded = sum(1 for r in records if r.status == "downloaded")
        page_skipped = sum(1 for r in records if r.status == "skipped_exists")
        page_errors = [r for r in records if r.status.startswith("error")]
        total_downloaded += page_downloaded
        total_skipped += page_skipped
        total_filtered += page_filtered
        all_errors.extend(page_errors)

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                dl=total_downloaded, exists=total_skipped,
                filtered=total_filtered, err=len(all_errors), refresh=False,
            )
        logger.info(
            "page %d done: +%d downloaded, %d already present, "
            "%d filtered (numbered), %d errors.",
            page, page_downloaded, page_skipped, page_filtered, len(page_errors),
        )
        time.sleep(DELAY_BETWEEN_PAGES_SECONDS)

    if progress_bar is not None:
        progress_bar.close()

    logger.info("--- Summary ---")
    logger.info("Downloaded this run: %d", total_downloaded)
    logger.info("Skipped (already present): %d", total_skipped)
    logger.info("Filtered out (numbered declarations): %d", total_filtered)
    logger.info("Errors: %d", len(all_errors))
    for record in all_errors:
        logger.warning("error: %s  %s", record.status, record.url)
    logger.info("Manifest: %s", MANIFEST_PATH)

    # Final summary always prints to console, even without --verbose.
    print(
        f"\nDone. Downloaded {total_downloaded}, already present {total_skipped}, "
        f"filtered {total_filtered}, errors {len(all_errors)}. "
        f"Log: {LOG_PATH} | Manifest: {MANIFEST_PATH}"
    )
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
