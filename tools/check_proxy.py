# -*- coding: utf-8 -*-
from get_confs import get_conf
from color import print亮红, print亮绿, print亮蓝
import requests

"""
检查代理是否可用
"""
def check_proxy(proxies):
    proxies_https = proxies['https'] if proxies is not None else '无'
    try:
        response = requests.get("https://ipapi.co/json/",
                                proxies=proxies, timeout=4)
        data = response.json()
        print(f'查询代理的地理位置，返回的结果是{data}')
        if 'country_name' in data:
            country = data['country_name']
            result = f"代理配置 {proxies_https}, 代理所在地：{country}"
        elif 'error' in data:
            result = f"代理配置 {proxies_https}, 代理所在地：未知，IP查询频率受限"
        print(result)
        response_google = requests.get("https://www.google.com/",
                                proxies=proxies, timeout=4)
        if response.status_code == 200:
            result = f"代理配置 {proxies_https}, 代理所在地查询成功，代理可用"
            print亮绿(result)
            return True
        return False
    except:
        result = f"代理配置 {proxies_https}, 代理所在地查询超时，代理可能无效"
        print亮红(result)
        return False

if __name__ == '__main__':
    import os
    os.environ['no_proxy'] = '*'  # 避免代理网络产生意外污染
    proxies, = get_conf('proxies')
    proxy_result = check_proxy(proxies)
    print("代理可用" if proxy_result else "代理不可用")