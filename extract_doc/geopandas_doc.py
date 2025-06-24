from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os

url = "https://osmnx.readthedocs.io/en/stable/user-reference.html"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, timeout=60000)
    
    # Wait for main content to appear
    page.wait_for_selector("div[role='main']", timeout=10000)
    
    html = page.content()
    browser.close()

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
    print("❌ Could not find main documentation block.")

# Save to .txt
os.makedirs("osmnx_docs", exist_ok=True)
with open("osmnx_docs/user_reference.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(text_parts))
