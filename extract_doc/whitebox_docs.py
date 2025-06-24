from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

BASE_URL = "https://www.whiteboxgeo.com/manual/wbt_book/"
OUTPUT_DIR = "whitebox_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    print(f"📦 Visiting homepage: {BASE_URL}")
    page.goto(BASE_URL, timeout=60000)
    page.wait_for_selector("div.sidebar-scrollbox", timeout=15000)

    soup = BeautifulSoup(page.content(), "html.parser")
    sidebar = soup.select_one("div.sidebar-scrollbox")

    links = {}

    # Collect all sidebar links
    for a in sidebar.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") or href.startswith("#"):
            continue
        full_url = urljoin(BASE_URL, href)
        filename = href.strip("/").split("/")[-1].replace(".html", "").replace("/", "_") or "index"
        filename = filename.split("#")[0]
        links[filename] = full_url

    print(f"🔗 Found {len(links)} documentation pages")

    # Fetch content from each page
    for name, full_url in links.items():
        print(f"📄 Fetching: {name} → {full_url}")
        try:
            page.goto(full_url, timeout=60000)
            page.wait_for_selector("div.content", timeout=15000)
        except Exception as e:
            print(f"❌ Skipped {name} due to error: {e}")
            continue

        soup = BeautifulSoup(page.content(), "html.parser")
        content = soup.find("div", class_="content")

        text_parts = []
        if content:
            for tag in content.find_all(["p", "h1", "h2", "h3", "ul", "ol", "li"]):
                text_parts.append(tag.get_text(strip=True))

            for pre in content.find_all("pre"):
                code = pre.get_text()
                text_parts.append("\n--- CODE BLOCK ---\n" + code + "\n------------------\n")

            with open(f"{OUTPUT_DIR}/{name}.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(text_parts))

    browser.close()

print(f"✅ Done: WhiteboxTools docs saved in '{OUTPUT_DIR}/'")
