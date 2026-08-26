import os
import urllib.request

def download_chart_js():
    # 1. Ensure the 'static' directory exists
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        print(f"📁 Created directory: {static_dir}")

    # 2. Target file path and CDN URL
    target_path = os.path.join(static_dir, "chart.min.js")
    cdn_url = "https://cdn.jsdelivr.net/npm/chart.js"

    # 3. Download the file
    print("⏳ Downloading chart.min.js for offline use...")
    try:
        urllib.request.urlretrieve(cdn_url, target_path)
        print(f"✅ Successfully downloaded to '{target_path}'!")
        print("Your application is now 100% offline ready.")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    download_chart_js()