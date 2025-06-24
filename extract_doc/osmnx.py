from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os, time

url = "https://osmnx.readthedocs.io/en/stable/user-reference.html"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, timeout=60000)

    try:
        # Preferred method
        page.wait_for_selector("div[role='main']", state="attached", timeout=15000)
    except:
        print("⚠️ Couldn't find selector. Waiting manually.")
        time.sleep(5)

    html = page.content()
    browser.close()

# Optional: Debug the saved HTML
with open("debug_osmnx.html", "w", encoding="utf-8") as f:
    f.write(html)

# Parse with BeautifulSoup
soup = BeautifulSoup(html, "html.parser")
main = soup.find("div", role="main")

text_parts = []

if main:
    for tag in main.find_all(["h1", "h2", "h3", "p", "pre", "ul", "ol", "li", "dl", "dt", "dd"]):
        if tag.name == "pre":
            text_parts.append("\n--- CODE BLOCK ---\n" + tag.get_text() + "\n------------------\n")
        elif tag.name == "dl":
            for dt in tag.find_all("dt"):
                text_parts.append(f"\n**{dt.get_text(strip=True)}**")
            for dd in tag.find_all("dd"):
                text_parts.append(dd.get_text(strip=True))
        else:
            text_parts.append(tag.get_text(strip=True))
else:
    print("❌ <div role='main'> not found. Check debug_osmnx.html")

# Save to file
os.makedirs("osmnx_docs", exist_ok=True)
with open("osmnx_docs/user_reference.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(text_parts))
    print("✅ Saved osmnx_docs/user_reference.txt")
