from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

# Base settings
BASE_URL = "https://shapely.readthedocs.io/en/stable/"
OUTPUT_DIR = "shapely_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    print(f"📦 Visiting homepage: {BASE_URL}")
    page.goto(BASE_URL, timeout=60000)
    page.wait_for_selector(".bd-toc-item.navbar-nav.active", timeout=15000)
    soup = BeautifulSoup(page.content(), "html.parser")

    # Step 1: Collect sidebar links
    sidebar = soup.select_one(".bd-toc-item.navbar-nav.active")
    links = {}

    for a in sidebar.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") or href.startswith("#"):
            continue
        full_url = urljoin(BASE_URL, href)
        filename = href.split("/")[-1].replace(".html", "").replace("/", "_").strip() or "index"
        filename = filename.split("#")[0]
        links[filename] = full_url

    print(f"🔗 Found {len(links)} documentation pages")

    # Step 2: Visit each link and extract content
    for name, full_url in links.items():
        print(f"📄 Fetching: {name} → {full_url}")
        try:
            page.goto(full_url, timeout=60000)
            page.wait_for_selector("article.bd-article", timeout=15000)
        except Exception as e:
            print(f"❌ Skipped {name} due to error: {e}")
            continue

        soup = BeautifulSoup(page.content(), "html.parser")
        content = soup.find("article", class_="bd-article")  # ✅ Corrected

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

print(f"✅ Done: Shapely docs saved in '{OUTPUT_DIR}/'")
