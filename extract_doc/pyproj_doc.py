from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse

BASE_URL = "https://pyproj4.github.io/pyproj/stable/"
OUTPUT_FOLDER = "pyproj_docs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Step 1: Load homepage and extract sidebar links
    print(f"📦 Visiting homepage: {BASE_URL}")
    page.goto(BASE_URL, timeout=60000)
    page.wait_for_selector("div.sidebar-scroll", timeout=15000)
    soup = BeautifulSoup(page.content(), "html.parser")
    sidebar = soup.find("div", class_="sidebar-scroll")

    links = {}
    for a in sidebar.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") or href.startswith("#"):
            continue  # skip external and anchor links
        full_url = urljoin(BASE_URL, href)
        title = href.split("/")[-1].replace(".html", "").replace("/", "_").strip() or "index"
        title = title.split("#")[0]  # Remove fragment identifiers
        links[title] = full_url

    print(f"🔗 Found {len(links)} internal docs")

    # Step 2: Visit each page and extract content
    for name, full_url in links.items():
        print(f"📄 Fetching: {name} → {full_url}")
        try:
            page.goto(full_url, timeout=60000)
            page.wait_for_selector("div.main", timeout=15000)
        except Exception as e:
            print(f"❌ Skipped {name} due to error: {e}")
            continue

        soup = BeautifulSoup(page.content(), "html.parser")
        content = soup.find("div", class_="main")

        text_parts = []
        if content:
            for tag in content.find_all(["p", "h1", "h2", "h3", "ul", "ol", "li"]):
                text_parts.append(tag.get_text(strip=True))

            for pre in content.find_all("pre"):
                code_text = pre.get_text()
                text_parts.append("\n--- CODE BLOCK ---\n" + code_text + "\n------------------\n")

            with open(f"{OUTPUT_FOLDER}/{name}.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(text_parts))

    browser.close()

print("✅ All sections downloaded into 'pyproj_docs/'")
