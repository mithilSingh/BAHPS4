import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

BASE_URL = "https://rasterio.readthedocs.io/en/stable/topics/index.html"
ROOT_URL = "https://rasterio.readthedocs.io/en/stable/topics/"

os.makedirs("rasterio_docs", exist_ok=True)

# Step 1: Get the index page with the list
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.text, "html.parser")

wrapper = soup.find("div", class_="toctree-wrapper compound")

docs = []
if wrapper:
    for li in wrapper.find_all("li", class_="toctree-l1"):
        a = li.find("a", class_="reference internal")
        if a and a.get("href"):
            title = a.text.strip().replace(" ", "_").replace("/", "_")
            full_url = urljoin(ROOT_URL, a["href"])
            docs.append((title, full_url))

# Step 2: Download each linked page
for title, url in docs:
    print(f"Fetching: {url}")
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    # Extract main body
    content = soup.find("div", class_="document")

    # Gather all normal text
    text_parts = []
    if content:
        for tag in content.find_all(["p", "h1", "h2", "h3", "ul", "ol", "li"]):
            text_parts.append(tag.get_text(strip=True))

        # Extract and preserve code blocks
        for pre in content.find_all("pre"):
            code_text = pre.get_text()  # get all text inside pre (including spans)
            text_parts.append("\n--- CODE BLOCK ---\n" + code_text + "\n------------------\n")

    # Save to file
    with open(f"rasterio_docs/{title}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_parts))
