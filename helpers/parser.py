from pathlib import Path
import html2text
import requests
from bs4 import BeautifulSoup, NavigableString

EFS_DIR = Path("../")

h = html2text.HTML2Text()

# Ignore converting links from HTML
h.ignore_links = False
h.mark_code = True
h.reference_links = True

def remove_examples_using_section(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    sections_to_remove = soup.find_all("section", id=lambda x: x and x.startswith("examples-using"))
    for section in sections_to_remove:
        section.decompose()
    return str(soup)

def extract_text_from_section(section):
  response = h.handle(section.prettify())
  return response

def path_to_uri(path, scheme="https://", domain="scikit-learn.org/stable/"):
    return scheme + domain + str(path).split(domain)[-1]

def extract_sections(record):
    with open(record["path"], "r", encoding="utf-8") as html_file:
        html_content = remove_examples_using_section(html_file)
        soup = BeautifulSoup(html_content, "html.parser")

    sections = soup.find_all("section")
    section_list = []

    if len(sections) == 0:
        uri = path_to_uri(path=record["path"])
        section_text = h.handle(soup.prettify())
        if section_text:
            section_list.append({"source": f"{uri}", "text": section_text})
    else:
        for section in sections:
            section_id = section.get("id").strip()
            section_text = extract_text_from_section(section)
            if section_text:
                uri = path_to_uri(path=record["path"])
                section_list.append({"source": f"{uri}#{section_id}", "text": section_text})
    return section_list


#Example usage:
#sample_html_fp = Path(EFS_DIR, "scikit-learn.org/stable/install.html")
#sample_html_fp = Path(EFS_DIR, "scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_3_0.html")
#sample_html_fp = Path(EFS_DIR, "scikit-learn.org/stable/index.html")
# sample_html_fp = Path(EFS_DIR, "scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html")

# result = extract_sections({"path": sample_html_fp})
# print(result)


# for i, element in enumerate(result):
#     print(str(i), element['source'])
#     print("-------------------------------------------------")
#     print(element['text'][:10000])
#     print("-------------------------------------------------")
