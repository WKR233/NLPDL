from pygoogletranslation import Translator
proxy={"http":"127.0.0.1:7890","https":"127.0.0.1:7890"}
translator=Translator(proxies=proxy)
dest=input("Please input the dest language(Chinese:zh-CN, English=en, Japanese=ja)\n")
query=input("Please input the query:\n")
t=translator.translate(query, src='auto', dest=dest)
print(t.text)
