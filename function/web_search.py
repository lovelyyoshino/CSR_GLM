import sys
import requests
import re
from bs4 import BeautifulSoup
sys.path.append("../llm_bridge")
from llm_bridge_all import model_type
from fuction_utils import input_clipping,request_gpt_model_in_new_thread
from pydork.engine import SearchEngine
sys.path.append("../tools")
from get_confs import get_conf
from color import print亮红, print亮绿, print亮蓝, print亮黄
from check_proxy import check_proxy

"""
通过搜索引擎搜索关键词，返回搜索结果
"""
def google(query, proxies):
    query = query # 在此处替换您要搜索的关键词
    url = f"https://www.google.com/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36'}
    response = requests.get(url, headers=headers, proxies=proxies)#设置代理，并返回response对象
    soup = BeautifulSoup(response.content, 'html.parser')#解析网页
    results = []
    for g in soup.find_all('div', class_='g'):#找到所有class为g的div标签
        anchors = g.find_all('a')#在每个div标签中找到所有的a标签
        if anchors:
            link = anchors[0]['href']#提取第一个a标签中的链接
            if link.startswith('/url?q='):#如果链接以/url?q=开头
                link = link[7:]
            if not link.startswith('http'):#如果链接不以http开头
                continue
            title = g.find('h3').text#提取h3标签中的文本
            item = {'title': title, 'link': link}
            results.append(item)

    for r in results:
        print(r['link'])#打印结果
    return results
        


def scrape_text(url, proxies) -> str:#根据url爬取网页内容
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36',
        'Content-Type': 'text/plain',
    }
    try: 
        response = requests.get(url, headers=headers, proxies=proxies, timeout=8)#设置代理，并返回response对象
        if response.encoding == "ISO-8859-1": response.encoding = response.apparent_encoding#解决乱码问题
    except: 
        return "无法连接到该网页"
    soup = BeautifulSoup(response.text, "html.parser")#解析网页
    for script in soup(["script", "style"]):#去除script和style标签
        script.extract()
    text = soup.get_text()#获取网页文本
    lines = (line.strip() for line in text.splitlines())#去除空行
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))#去除多余空格
    text = "\n".join(chunk for chunk in chunks if chunk)#去除空行
    return text

"""
0代表google搜索，1代表百度搜索
"""
def website_search(txt, proxies, se=1):
    # ------------- < 第1步：爬取搜索引擎的结果 > -------------
    history = []
    say_txt = f"从以上搜索结果中抽取信息，然后回答问题：{txt}"
    urls=[]
    search_engine = SearchEngine()
    if se==0:
        search_engine.set('google')
        search_engine.set_proxy(proxies)
        # urls = google(txt,proxies)
        urls = search_engine.search(txt, maximum=30)
    elif se==1:
        search_engine.set('yahoo')
        urls = search_engine.search(txt)
        print亮红(urls)

    # ------------- < 第2步：依次访问网页 > -------------
    max_search_result = 5   # 最多收纳多少个网页的结果
    if len(urls)>max_search_result:
        for index, url in enumerate(urls[:max_search_result]):
            json_url = url['link'].replace("/url?q=","")
            json_url = json_url.split("&sa=")[0]
            print(json_url)
            res = scrape_text(json_url, proxies)
            history.extend([f"第{index}份搜索结果：", res])

    return say_txt, history


if __name__ == "__main__":
    proxies, LLM_MODEL, API_KEY = get_conf('proxies', 'LLM_MODEL',  'API_KEY')
    history=[]
    say_txt="我现在的IP地址为"
    # if check_proxy(proxies):
    #     say_txt, history =website_search("我现在的IP地址为", proxies,0)
    say_txt, history =website_search("我现在的IP地址为", proxies,1)
    
    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': LLM_MODEL,
        'top_p': 1.0,
        'max_length': None,
        'temperature': 1.0,
    }
    say_txt, history = input_clipping(    # 裁剪输入，从最长的条目开始裁剪，防止爆token
        inputs=say_txt, 
        history=history, 
        max_token_limit=model_type[llm_kwargs['llm_model']]['max_token']*3//4
    )
    print亮蓝(f"裁剪后的输入：{say_txt},{history}")


