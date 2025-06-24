from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

url = "https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, timeout=60000)
    page.wait_for_timeout(5000)
    html = page.content()
    browser.close()

soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(separator="\n", strip=True)

with open("bhoonidhi_api_full.txt", "w", encoding="utf-8") as f:
    f.write(text)
