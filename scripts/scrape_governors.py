# scripts/scrape_governors.py
"""
Title: Governor Scrape (Wikipedia → data/governors.csv)
Description: For each US state and the five NGA-member territories, fetches the
    "List of governors of <state>" Wikipedia article (wikitext via the MediaWiki
    API) and extracts, via the Anthropic SDK with a fixed structured-output
    tool, the governors who served at any point during 2007–2026: name, party
    (Democratic/Republican/Independent/Other), and ISO term_start/term_end.
    Writes data/governors.csv (committed). This is the only networked,
    non-deterministic component; the load step (import_governors.py) is offline.
Changelog:
    2026-06-13  Initial version.
"""

import argparse
import csv
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import dotenv

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pda.governors_import import GOVERNOR_CSV_COLUMNS, governor_rows_from_extraction

DEFAULT_OUT = "data/governors.csv"
# Clean structured tables — a small, fast model is sufficient and cheap.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia policy asks for a descriptive User-Agent with contact info, and
# rate-limits rapid anonymous requests (HTTP 429). Be polite between calls.
WIKI_USER_AGENT = "disaster-denials-research/1.0 (chris@webster.family)"
REQUEST_DELAY_SECONDS = 1.0

# (state_abbr, Wikipedia article title) for 50 states + 5 NGA territories.
STATE_ARTICLES = [
    ("AL", "List of governors of Alabama"), ("AK", "List of governors of Alaska"),
    ("AZ", "List of governors of Arizona"), ("AR", "List of governors of Arkansas"),
    ("CA", "List of governors of California"), ("CO", "List of governors of Colorado"),
    ("CT", "List of governors of Connecticut"), ("DE", "List of governors of Delaware"),
    ("FL", "List of governors of Florida"), ("GA", "List of governors of Georgia"),
    ("HI", "List of governors of Hawaii"), ("ID", "List of governors of Idaho"),
    ("IL", "List of governors of Illinois"), ("IN", "List of governors of Indiana"),
    ("IA", "List of governors of Iowa"), ("KS", "List of governors of Kansas"),
    ("KY", "List of governors of Kentucky"), ("LA", "List of governors of Louisiana"),
    ("ME", "List of governors of Maine"), ("MD", "List of governors of Maryland"),
    ("MA", "List of governors of Massachusetts"), ("MI", "List of governors of Michigan"),
    ("MN", "List of governors of Minnesota"), ("MS", "List of governors of Mississippi"),
    ("MO", "List of governors of Missouri"), ("MT", "List of governors of Montana"),
    ("NE", "List of governors of Nebraska"), ("NV", "List of governors of Nevada"),
    ("NH", "List of governors of New Hampshire"), ("NJ", "List of governors of New Jersey"),
    ("NM", "List of governors of New Mexico"), ("NY", "List of governors of New York"),
    ("NC", "List of governors of North Carolina"), ("ND", "List of governors of North Dakota"),
    ("OH", "List of governors of Ohio"), ("OK", "List of governors of Oklahoma"),
    ("OR", "List of governors of Oregon"), ("PA", "List of governors of Pennsylvania"),
    ("RI", "List of governors of Rhode Island"), ("SC", "List of governors of South Carolina"),
    ("SD", "List of governors of South Dakota"), ("TN", "List of governors of Tennessee"),
    ("TX", "List of governors of Texas"), ("UT", "List of governors of Utah"),
    ("VT", "List of governors of Vermont"), ("VA", "List of governors of Virginia"),
    ("WA", "List of governors of Washington"), ("WV", "List of governors of West Virginia"),
    ("WI", "List of governors of Wisconsin"), ("WY", "List of governors of Wyoming"),
    ("AS", "List of governors of American Samoa"), ("GU", "List of governors of Guam"),
    ("MP", "List of governors of the Northern Mariana Islands"),
    ("PR", "List of governors of Puerto Rico"),
    ("VI", "List of governors of the United States Virgin Islands"),
]

EXTRACT_TOOL = {
    "name": "record_governors",
    "description": ("Record every governor who served at any point during "
                    "2007-01-01 through 2026-12-31, from the article."),
    "input_schema": {
        "type": "object",
        "properties": {
            "governors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "party": {"type": "string",
                                  "enum": ["Democratic", "Republican",
                                           "Independent", "Other"]},
                        "term_start": {"type": "string",
                                       "description": "ISO YYYY-MM-DD"},
                        "term_end": {"type": "string",
                                     "description": "ISO YYYY-MM-DD; empty string if still serving"},
                    },
                    "required": ["name", "party", "term_start", "term_end"],
                },
            },
        },
        "required": ["governors"],
    },
}

PROMPT = ("From the Wikipedia article wikitext below, list every person who held "
          "the governorship at any time during 2007-01-01 to 2026-12-31. Use one "
          "entry per continuous tenure, with full ISO dates. Map the party to "
          "Democratic, Republican, Independent, or Other. Leave term_end as an "
          "empty string for the person currently serving.\n\nWIKITEXT:\n")


def article_url(title):
    """Return the human Wikipedia URL for an article title."""
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def fetch_wikitext(title, max_retries=4):
    """Fetch an article's wikitext via the MediaWiki API, retrying on 429.

    Args:
        title: Wikipedia article title.
        max_retries: attempts before giving up on rate-limit responses.
    Returns:
        The wikitext string.
    Raises:
        urllib.error.HTTPError: on a non-429 error, or 429 after all retries.
    """
    query = urllib.parse.urlencode({
        "action": "parse", "page": title, "prop": "wikitext",
        "format": "json", "formatversion": "2", "redirects": "1"})
    request = urllib.request.Request(
        f"{WIKI_API}?{query}", headers={"User-Agent": WIKI_USER_AGENT})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 (fixed host)
                return json.load(response)["parse"]["wikitext"]
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))   # 5s, 10s, 15s backoff
                continue
            raise


def extract_governors(client, model, wikitext):
    """Call the model to extract governors from wikitext.

    Args:
        client: anthropic.Anthropic client.
        model: model id.
        wikitext: the article wikitext.
    Returns:
        list of governor dicts (name, party, term_start, term_end).
    """
    message = client.messages.create(
        model=model, max_tokens=4096, tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_governors"},
        messages=[{"role": "user", "content": PROMPT + wikitext}])
    for block in message.content:
        if block.type == "tool_use":
            return block.input.get("governors", [])
    return []


def write_csv(out_path, rows):
    """Write governor rows to a CSV constrained to the project tree.

    Args:
        out_path: output path (default data/governors.csv).
        rows: list of dicts keyed by GOVERNOR_CSV_COLUMNS.
    """
    base = pathlib.Path.cwd().resolve()
    path = (base / out_path).resolve()
    if base != path and base not in path.parents:
        raise ValueError(f"refusing to write outside the project: {out_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOVERNOR_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Scrape every state/territory list and write data/governors.csv."""
    import anthropic  # imported here so the offline load path needs no SDK

    parser = argparse.ArgumentParser(description="Scrape governors → CSV")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    dotenv.load_dotenv()
    client = anthropic.Anthropic()

    all_rows = []
    for index, (state_abbr, title) in enumerate(STATE_ARTICLES):
        if index:
            time.sleep(REQUEST_DELAY_SECONDS)   # be polite to the Wikipedia API
        wikitext = fetch_wikitext(title)
        extracted = extract_governors(client, args.model, wikitext)
        rows = governor_rows_from_extraction(state_abbr, article_url(title), extracted)
        all_rows.extend(rows)
        print(f"{state_abbr}: {len(rows)} tenures in window")

    write_csv(args.out, all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
