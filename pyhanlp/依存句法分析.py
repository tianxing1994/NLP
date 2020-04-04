from pyhanlp import HanLP


def demo1():
    ret = HanLP.parseDependency("徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。")
    print(ret)

    return


if __name__ == '__main__':
    demo1()
