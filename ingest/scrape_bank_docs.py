from pathlib import Path
import scrapy
from scrapy.crawler import CrawlerProcess


class InvestopediaSpider(scrapy.Spider):
    name = "investopedia"
    start_urls = [
        "https://www.investopedia.com/articles/personal-finance/050515/how-swift-system-works.asp",
    ]

    custom_settings = {
        "FEEDS": {"output/raw_pages.jl": {"format": "jsonlines", "encoding": "utf8"}},
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "ROBOTSTXT_OBEY": True,
        "HTTPERROR_ALLOWED_CODES": [460],
        "LOG_LEVEL": "INFO",
    }

    def parse(self, response):
        yield {
            "url": response.url,
            "title": response.css("title::text").get(),
            "body": response.text,
        }


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    process = CrawlerProcess()
    process.crawl(InvestopediaSpider)
    process.start()
    print("✅ Scraping completed.")
