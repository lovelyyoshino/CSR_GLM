import sys
import requests
import random
import re
from bs4 import BeautifulSoup
from time import sleep
from fake_useragent import UserAgent
sys.path.append("../llm_bridge")
from llm_bridge_all import model_type
from fuction_utils import input_clipping,request_gpt_model_in_new_thread
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

def get_real_url(v_url, headers):
	"""
	获取百度链接真实地址
	:param v_url: 百度链接地址
	:return: 真实地址
	"""
	r = requests.get(v_url, headers=headers, allow_redirects=False)  # 不允许重定向
	if r.status_code == 302:  # 如果返回302，就从响应头获取真实地址
		real_url = r.headers.get('Location')
	else:  # 否则从返回内容中用正则表达式提取出来真实地址
		real_url = re.findall("URL='(.*?)'", r.text)[0]
	# print('real_url is:', real_url)
	return real_url
# https://github.com/chao325/baidu_getUrls
def baidu(v_keyword, proxies):
    """
    爬取百度搜素结果
    :param v_max_page: 爬取前几页  【必填】
    :param v_keyword: 搜索关键词   【必填】
    :param v_setCookie: 更新Cookie 【不必填】
    :return: None
    """
    setCookie="BIDUPSID=A26AE98583A318DE98C0F123DB07607D; PSTM=1650780390; BAIDUID=A26AE98583A318DED1F2DE82E249C426:FG=1; ZFY=3Y1VeFzr3VhARQLTGQIeZWQnAKiPSxJWjVhb11HhkAs:C; BAIDUID_BFESS=A26AE98583A318DED1F2DE82E249C426:FG=1; __bid_n=186ed6fad3897f84b94207; ZD_ENTRY=other; BCLID=11509748439451522012; BDSFRCVID=lckOJeC62iQ5_ovfQ5FEdDvnAej5cgOTH6_nK5-UJDUtYCUxmdZaEG0PWU8g0KuMzoX4ogKK5mOTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID_SF=JnujoI_htKI3jbjY5PQEb-_thMuX2tQJfKJ2Bh7F5l8-hRThjqno54Kujtj4KU3qae6-BD5J5h7xOKQ3hx4hqqLpQboJXxJPMIjbhJjN3KJmSUK9bT3v5fuhhnAJ2-biWbTL2MbdJqvP_IoG2Mn8M4bb3qOpBtQmJeTxoUJ25DnJhbLGe4bK-TrBDaDO3J; BCLID_BFESS=11509748439451522012; BDSFRCVID_BFESS=lckOJeC62iQ5_ovfQ5FEdDvnAej5cgOTH6_nK5-UJDUtYCUxmdZaEG0PWU8g0KuMzoX4ogKK5mOTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID_SF_BFESS=JnujoI_htKI3jbjY5PQEb-_thMuX2tQJfKJ2Bh7F5l8-hRThjqno54Kujtj4KU3qae6-BD5J5h7xOKQ3hx4hqqLpQboJXxJPMIjbhJjN3KJmSUK9bT3v5fuhhnAJ2-biWbTL2MbdJqvP_IoG2Mn8M4bb3qOpBtQmJeTxoUJ25DnJhbLGe4bK-TrBDaDO3J; BA_HECTOR=al258kahaga1aha00h818l9g1i1fd1a1m; PSINO=5; delPer=0; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDUSS=Vk1Sm1HNDg0UE41UTFDZUNURHBQbzBPVkZRaTVmWldYN25ZSkdhM2t0LTZtRDlrSUFBQUFBJCQAAAAAAAAAAAEAAABI4BxUyrLDtNPDu6fD-zAzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALoLGGS6CxhkZ; BDUSS_BFESS=Vk1Sm1HNDg0UE41UTFDZUNURHBQbzBPVkZRaTVmWldYN25ZSkdhM2t0LTZtRDlrSUFBQUFBJCQAAAAAAAAAAAEAAABI4BxUyrLDtNPDu6fD-zAzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALoLGGS6CxhkZ; H_PS_PSSID=38185_36543_38409_36677_38355_38368_38306_37862_38174_38290_38217_38262_37920_38312_38382_38284_26350_22157_38282_37881; BAIDU_WISE_UID=wapp_1679297819561_943; arialoadData=false; ab_sr=1.0.1_NzlhMzNkNGUxZWY1MDA5MWNkMDczNWIyOWZmYmFlYzgyNjRiNDU1MjlkNGFlZmRlODQ5ZDM4ODE0MzBlOWNjZjgyYmY5YjUzYjMwYjRmMTlhYTcyMzM5M2ZiN2U3MjQ2NDBkOTVlMzc5OWY2Y2U5MjhjMWE3ZmI2MjgxODhiODQ4ZTFjNmQyZThjOTU1Y2QxZDExYWRlODgxMWU4NDlhMzZiOWIwZTA1MzBkYjhkZTlkOWU0MTMzZDQwMmE2NGI3; FPTOKEN=HXFCQlrRn7oMtgFsr8UXdAbJki7qCoSUC6EqXsW8/EAgRfWPStZZI12UxTHhjEFJAEh3b/hxHukGB8s+8qcU4fB3Ufzre+gUcvZZ4L1oAgS0mjO0UA4CKr8ev2XFVk9zw/4p7wn8okHm/EjmYwlxjNzp6TXlqtifz01wSNJC4KkwhVFpV5ZmhRz8GR2DQTamBBSIFQ1wo5VwcAn6I2/0vG/z980WhE92/ih24h/UXIBEOeWRws+gAltdgVfQm+Adhw7kdW8s6gmVfBTlq0pFEaHaY06IF0AcJt+yF8wkMZO8LamxawsxXfEjxSdNh5PH76tAwlUwerDEDuNJshpYphZvAXW5/CV+mbasPCjqU9q+IVotp27Ez0yQ6W8aRqOEwd5P8V5eB1U4Yb/EeWJg1A==|TFoZxLEBUKzZkIlW2yaHBVHmUrbzKRdZukwLXPHc70g=|10|609e064d1dfa88f702b6f86ab8cd7139; RT='z=1&dm=baidu.com&si=dba700fb-0c2b-444d-95ac-b74e3352df3a&ss=lfgik8cv&sl=2&tt=8ik&bcn=https://fclog.baidu.com/log/weirwood?type=perf&ld=5qc&ul=6db&hd=6ek'"
    # 伪装浏览器请求头
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.baidu.com",
    # 需要更换Cookie
    "Cookie": setCookie
    }
    v_max_page =1
    # 获得每页搜索结果
    for page in range(v_max_page):
        # print('开始爬取第{}页'.format(page + 1))
        wait_seconds = random.uniform(1, 2)  # 等待时长秒
        # print('开始等待{}秒'.format(wait_seconds))
        sleep(wait_seconds)  # 随机等待
        url = 'http://www.baidu.com/s?wd=' + v_keyword + '&pn=' + str(page * 10)
        r = requests.get(url, headers=headers)
        html = r.text
        # print('响应码是:{}'.format(r.status_code))
        # print(html)
        soup = BeautifulSoup(html, 'html.parser')
        result_list_0 = soup.find_all(class_='result c-container new-pmd')
        result_list_1 = soup.find_all(class_='result c-container xpath-log new-pmd')
        result_list = result_list_0 + result_list_1
        url_over = []  # 百度的链接
        for result in result_list:
            title = result.find('a').text
            href = result.find('a')['href']
            real_url = get_real_url(v_url=href,headers=headers)
            url_over.append({'title': title, 'link': real_url})
        return url_over
        # for item in url_over:
        #     result_all.append(item)


def bing_search(query: str, proxies=None):
    ua = UserAgent()
    url = f"https://cn.bing.com/search?q={query}"
    res = requests.get(
        url,
        headers={
            "User-Agent": ua.random
        },
        proxies=proxies
    )

    results = []

    for titleContext, caption in re.findall(
        '<li class="b_algo"[^>]*><div class="tpcn"[^>]*>.*?</div><h2>(.*?)</h2><div class="b_caption">(.*?)</div></li>',
        res.text,
    ):
        url, title = re.findall('<a[^>]*href="(.*?)"[^>]*>(.*?)</a>', titleContext)[0]

        item = {'title': title, 'link': url}
        results.append(item)

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
    if se==0:
        urls = google(txt,proxies)
    elif se==1:
        urls = baidu(txt,proxies)
    elif se ==2:
        urls = bing_search(txt,proxies)

    # ------------- < 第2步：依次访问网页 > -------------
    max_search_result = 5   # 最多收纳多少个网页的结果
    if len(urls)>max_search_result:
        for index, url in enumerate(urls[:max_search_result]):
            res = scrape_text(url['link'], proxies)
            history.extend([f"第{index}份搜索结果：", res])

    return say_txt, history


if __name__ == "__main__":
    proxies, LLM_MODEL, API_KEY = get_conf('proxies', 'LLM_MODEL',  'API_KEY')
    history=[]
    say_txt="我现在的IP地址为"
    if check_proxy(proxies):
        say_txt, history =website_search("我现在的IP地址为", proxies,0)
        print亮蓝(f"使用google搜索，结果为：{history}")
        say_txt, history =website_search("我现在的IP地址为", proxies,2)
        print亮蓝(f"使用bing搜索，结果为：{history}")
    else:
        say_txt, history =website_search("我现在的IP地址为", proxies,1)
        print亮蓝(f"使用百度搜索，结果为：{history}")

    
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


