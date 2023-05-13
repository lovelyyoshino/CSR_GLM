if __name__ == '__main__':
    import os
    os.environ['no_proxy'] = '*'  # 避免代理网络产生意外污染
    proxies, = get_conf('proxies')
    proxy_result = check_proxy(proxies)
    print("代理可用" if proxy_result else "代理不可用")