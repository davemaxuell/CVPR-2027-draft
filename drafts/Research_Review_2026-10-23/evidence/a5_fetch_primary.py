from pathlib import Path
from urllib.request import Request, urlopen
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).parent
URLS = {
    'world4scorer': 'https://arxiv.org/html/2609.36438v1',
    'mcr': 'https://arxiv.org/html/2608.09073v1',
    'cvaml': 'https://arxiv.org/html/2505.22772v2',
    'pfd': 'https://arxiv.org/html/2604.25859v1',
    'trm': 'https://arxiv.org/html/2605.22164v2',
    'cem': 'https://arxiv.org/html/2610.00921v1',
}

class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'tr'): self.items.append('\n')
        if tag == 'math':
            self.items.append(dict(attrs).get('alttext', ''))
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip -= 1
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'tr'): self.items.append('\n')
    def handle_data(self, data):
        if not self.skip: self.items.append(data)

def fetch(entry):
    name, url = entry
    try:
        local = ROOT / ('a5_' + name + '.html')
        if local.exists():
            html = local.read_text(encoding='utf-8')
        else:
            with urlopen(Request(url, headers={'User-Agent': 'ResearchReview/1.0'}), timeout=45) as response:
                html = response.read().decode('utf-8')
        parser = TextParser()
        parser.feed(html)
        lines = [' '.join(line.split()) for line in ''.join(parser.items).splitlines()]
        result = '\n'.join(line for line in lines if line)
        (ROOT / ('a5_' + name + '.txt')).write_text('Source: ' + url + '\nAccessed: 2026-10-23\n\n' + result, encoding='utf-8')
        return name + ': saved ' + str(len(result)) + ' chars'
    except Exception as error:
        return name + ': ' + repr(error)

if __name__ == '__main__':
    for result in ThreadPoolExecutor(max_workers=6).map(fetch, URLS.items()): print(result)
