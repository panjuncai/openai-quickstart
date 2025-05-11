import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import faiss
import gradio as gr
import openai
import json
import re
import glob
from langchain import SerpAPIWrapper, FAISS, InMemoryDocstore
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool, WriteFileTool, ReadFileTool
from langchain_experimental.autonomous_agents import AutoGPT

openai.api_key = os.getenv("OPENAI_API_KEY")

# 定义输出文件目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class WebCrawlerTool:
    """工具用于爬取网页内容"""
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def run(self, url):
        """爬取指定URL的网页内容"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本、样式等不需要的元素
            for script in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                script.extract()
                
            # 获取正文内容
            text = soup.get_text(separator=' ', strip=True)
            
            # 简单清理文本
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = ' '.join(lines)
            
            # 限制文本长度以减少token消耗 (约1000个字)
            if len(text) > 3000:
                text = text[:3000] + "..."
                
            return text
        except Exception as e:
            return f"爬取网页时出错: {str(e)}"


class NewsSearchTool:
    """工具用于搜索最近一周的中文新闻"""
    def __init__(self):
        self.search = SerpAPIWrapper()
        
    def run(self, query):
        """搜索最近一周相关的中文新闻"""
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        date_str = f"after:{week_ago.strftime('%Y-%m-%d')}"
        # 添加中文搜索参数
        search_query = f"{query} {date_str} 新闻 中文"
        
        results = self.search.run(search_query)
        return results


class NewsSummaryTool:
    """工具用于总结新闻内容"""
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.2)
        
    def run(self, content):
        """总结新闻内容，限制在300字以内"""
        prompt = f"""
        请对以下新闻内容进行总结，要求：
        1. 用中文回答
        2. 总结不超过300字
        3. 提取关键信息，包括主要事件、时间、地点和影响
        4. 保持客观，不添加个人观点
        
        新闻内容：
        {content}
        """
        
        response = self.llm.invoke(prompt)
        return response.content


class AutoGPTTool:
    def __init__(self):
        self.search = SerpAPIWrapper()
        self.webcrawler = WebCrawlerTool()
        self.news_search = NewsSearchTool()
        self.news_summary = NewsSummaryTool()
        
        # 设置输出文件目录
        self.output_dir = OUTPUT_DIR
        
        self.tools = [
            Tool(
                name="news_search",
                func=self.news_search.run,
                description="用于搜索最近一周的中文新闻。输入应该是你想搜索的新闻主题。",
            ),
            Tool(
                name="web_crawler",
                func=self.webcrawler.run,
                description="用于爬取网页内容。输入应该是一个完整的URL。",
            ),
            Tool(
                name="news_summary",
                func=self.news_summary.run,
                description="用于总结新闻内容，限制在300字以内。输入应该是要总结的文本内容。",
            ),
            WriteFileTool(root_dir=self.output_dir),  # 指定写入文件的根目录
            ReadFileTool(root_dir=self.output_dir),   # 指定读取文件的根目录
        ]

        self.embeddings_model = OpenAIEmbeddings()
        self.embedding_size = 1536
        self.index = faiss.IndexFlatL2(self.embedding_size)
        self.vectorstore = FAISS(
            self.embeddings_model.embed_query,
            self.index,
            InMemoryDocstore({}),
            {},
        )
        self.llm = ChatOpenAI(temperature=0)

        # self.agent = AutoGPT.from_llm_and_tools(
        #     ai_name="新闻君",
        #     ai_role="新闻总结助手",
        #     tools=self.tools,
        #     llm=ChatOpenAI(temperature=0),
        #     memory=self.vectorstore.as_retriever(),
        # )
        # 配置AI输出到指定文件路径
        # self.agent.chain.verbose = True

    def _create_agent(self): # 新增一个创建 agent 的方法
        # 每次调用都创建一个新的 FAISS 实例和 InMemoryDocstore，以确保记忆是干净的
        # 如果你希望跨调用保留一些记忆，这里的实现需要调整
        index = faiss.IndexFlatL2(self.embedding_size)
        docstore = InMemoryDocstore({}) # 每个 agent 有自己的 docstore
        vectorstore = FAISS(
            self.embeddings_model.embed_query,
            index,
            docstore,
            {},
            # relevant_docs=1 # 可以尝试限制检索的相关文档数量
        )

        agent = AutoGPT.from_llm_and_tools(
            ai_name="新闻君",
            ai_role="新闻总结助手",
            tools=self.tools,
            llm=self.llm, # 复用已创建的llm
            memory=vectorstore.as_retriever(search_kwargs={"k": 1}), # 每次都用新的 vectorstore
            # max_iterations=10, # 可以尝试限制最大迭代次数
            # chat_history_memory=None # 明确禁用或配置历史记忆，如果AutoGPT支持
        )
        agent.chain.verbose = True
        return agent

    def clean_output_files(self):
        """清理当前目录下的所有txt文件"""
        txt_files = glob.glob(os.path.join(self.output_dir, "*.txt"))
        for file in txt_files:
            try:
                os.remove(file)
                print(f"已删除文件: {file}")
            except Exception as e:
                print(f"删除文件 {file} 时出错: {str(e)}")
    
    def find_recent_output_files(self):
        """查找当前目录下最近创建的txt文件"""
        txt_files = glob.glob(os.path.join(self.output_dir, "*.txt"))
        recent_files = []
        
        # 查找30秒内创建的文件
        for file in txt_files:
            file_creation_time = os.path.getctime(file)
            if (datetime.now().timestamp() - file_creation_time) < 30:  # 30秒内创建的文件
                recent_files.append(file)
        
        return recent_files

    def process_news_summary(self, query):
        """处理新闻总结请求的主要方法 - 只处理一篇中文新闻以节省token"""
        try:
            # 清理之前的输出文件
            self.clean_output_files()
            
            # 搜索相关新闻
            search_results = self.news_search.run(query)
            
            # 从搜索结果中提取URL (假设搜索结果中包含URL信息)
            urls = self._extract_urls_from_search(search_results)
            
            if not urls:
                return f"未能找到相关中文新闻。搜索结果：{search_results}"
            
            # 只选择第一个URL (中文新闻网站优先)
            selected_url = self._select_best_chinese_news_url(urls)
            
            # 爬取内容
            content = self.webcrawler.run(selected_url)
            
            # 进行总结
            summary = self.news_summary.run(content)
            
            # 保存到文件
            filename = f"news_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"新闻总结：\n{summary}\n\n新闻来源：\n{selected_url}")
            
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            return file_content
        except Exception as e:
            return f"处理新闻总结时出错: {str(e)}"
    
    def _select_best_chinese_news_url(self, urls):
        """从URL列表中选择最合适的中文新闻URL"""
        # 中文新闻网站域名列表 (按优先级排序)
        chinese_news_domains = [
            'people.com.cn', 'xinhuanet.com', 'chinadaily.com.cn', 'thepaper.cn',
            'qq.com', 'sina.com.cn', '163.com', 'sohu.com', 'ifeng.com',
            'caixin.com', 'bjnews.com.cn', 'yicai.com', 'huanqiu.com'
        ]
        
        # 首先尝试匹配中文新闻网站
        for domain in chinese_news_domains:
            for url in urls:
                if domain in url.lower():
                    return url
        
        # 如果没有匹配到中文新闻网站，返回第一个URL
        return urls[0]
    
    def _extract_urls_from_search(self, search_results):
        """从搜索结果中提取URL"""
        try:
            # 尝试用简单的方式提取URL
            urls = []
            import re
            pattern = r'https?://[^\s)"]+'
            urls = re.findall(pattern, search_results)
            
            # 过滤掉不是新闻网站的URL
            news_domains = [
               'people.com.cn', 'xinhuanet.com', 'chinadaily.com.cn', 'thepaper.cn',
            'qq.com', 'sina.com.cn', '163.com', 'sohu.com', 'ifeng.com',
            'caixin.com', 'bjnews.com.cn', 'yicai.com', 'huanqiu.com' 
            ]
            
            filtered_urls = []
            for url in urls:
                if any(domain in url.lower() for domain in news_domains):
                    filtered_urls.append(url)
            
            return filtered_urls or urls  # 如果过滤后为空，则返回原始URL列表
        except:
            # 如果无法解析，返回空列表
            return []

    def process_question(self, question):
        """处理用户问题的方法"""
        # 清理之前的输出文件
        self.clean_output_files()
        
        # 对于新闻总结请求，使用特定的处理方法
        if "新闻" in question and ("总结" in question or "概括" in question):
            return self.process_news_summary(question)
        
        # 修改问题提示，指导AI将结果写入到文件中
        output_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        ai_instruction = f"{question}\n请将完整的回答写入到 {output_filename} 文件中。"
        # 每次处理问题时创建一个新的 agent 实例
        current_agent = self._create_agent()
        
        try:
            # 使用AutoGPT处理
            result = current_agent.run([ai_instruction])
        except Exception as e:
            # 处理可能的错误，例如finish命令导致的错误
            error_str = str(e)
            # 如果是finish命令导致的错误，从错误信息中提取响应
            if "finish" in error_str.lower():
                try:
                    # 尝试从错误信息中提取JSON响应
                    match = re.search(r'(\{.+\})', error_str, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        data = json.loads(json_str)
                        if "command" in data and data["command"]["name"] == "finish" and "args" in data["command"]:
                            result = data["command"]["args"].get("response", "任务已完成！")
                        else:
                            result = "AutoGPT完成了任务，但未能获取具体结果。"
                    else:
                        result = "AutoGPT完成了任务，但未能获取具体结果。"
                except:
                    # 如果解析失败，使用通用消息
                    result = "AutoGPT完成了任务，但未能提供详细结果。"
            else:
                # 其他类型的错误
                result = f"处理问题时出错: {error_str}"
            
            # 将结果写入文件
            try:
                filename = f"error_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result)
            except:
                pass
        
        # 检查是否有生成的文件
        recent_files = self.find_recent_output_files()
        if recent_files:
            # 读取所有生成的文件内容
            file_contents = []
            for file in recent_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    file_name = os.path.basename(file)
                    file_contents.append(f"{file_content}")
                except Exception as e:
                    file_contents.append(f"读取文件 {os.path.basename(file)} 时出错: {str(e)}")
            
            if file_contents:
                return "".join(file_contents)
        
        # 如果没有找到文件，则创建文件保存结果
        if isinstance(result, str) and result.strip():
            try:
                filename = f"autoresponse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                # 读取文件内容
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"创建结果文件时出错: {str(e)}")
        
        # 如果没有找到文件，返回原始结果
        return result

    def setup_gradio_interface(self):
        with gr.Blocks(theme="soft") as iface:
            gr.Markdown("# 中文新闻总结助手 - 新闻君")
            gr.Markdown("我是你的中文新闻总结助手：新闻君，输入你想了解的新闻主题，我会帮你搜索并总结最近一周的相关中文新闻～")
            
            with gr.Row():
                with gr.Column(scale=4):
                    query_input = gr.Textbox(
                        lines=5,
                        label="问题",
                        placeholder="请输入问题或新闻主题...",
                    )
                with gr.Column(scale=1):
                    submit_button = gr.Button("提交", variant="primary")
                    clear_button = gr.Button("清空")
            
            # 将示例放在这里，显示为一行
            with gr.Row():
                gr.Examples(
                    examples=[
                        "国内最新经济政策",
                        "中国航天最新进展",
                        "人工智能在医疗领域的应用",
                        "国内体育赛事最新消息",
                        "环保政策的最新动态",
                    ],
                    inputs=query_input,
                    label="示例点击",
                )
            
            # 结果显示
            result_output = gr.Textbox(
                lines=15,
                label="答案",
                placeholder="结果将显示在这里...",
            )
            
            # 定义处理函数
            def process_query(query):
                if not query.strip():
                    return "请输入有效的问题或新闻主题"
                
                try:
                    # 清理旧文件
                    self.clean_output_files()
                    
                    # 处理问题
                    result = self.process_question(query)
                    
                    # 返回结果
                    return result
                except Exception as e:
                    # 捕获所有异常，确保UI不会崩溃
                    error_msg = f"处理时发生错误: {str(e)}"
                    print(f"处理请求异常: {str(e)}")
                    
                    # 尝试将错误信息写入文件
                    try:
                        error_filename = f"ui_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        error_filepath = os.path.join(self.output_dir, error_filename)
                        with open(error_filepath, 'w', encoding='utf-8') as f:
                            f.write(f"处理问题 '{query}' 时出错:\n{str(e)}")
                        
                        # 读取文件
                        with open(error_filepath, 'r', encoding='utf-8') as f:
                            error_content = f.read()
                        
                        return error_content
                    except:
                        # 如果文件操作也失败，直接返回错误信息
                        return error_msg
            
            # 清空输入和输出
            def clear_inputs():
                return "", ""
            
            # 绑定事件
            submit_button.click(
                fn=process_query,
                inputs=query_input,
                outputs=result_output,
            )
            
            clear_button.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[query_input, result_output],
            )
            
            query_input.submit(
                fn=process_query,
                inputs=query_input,
                outputs=result_output,
            )
            
        return iface


if __name__ == "__main__":
    try:
        # 使用示例
        autogpt_tool = AutoGPTTool()
        
        # 启动时清理所有txt文件
        print("正在清理旧文件...")
        autogpt_tool.clean_output_files()
        
        print("启动Gradio界面...")
        gradio_interface = autogpt_tool.setup_gradio_interface()
        gradio_interface.launch(share=True, server_name="127.0.0.1")
    except Exception as e:
        print(f"启动时发生错误: {str(e)}")
        # 如果是关键错误，等待用户确认
        if "api_key" in str(e).lower() or "openai" in str(e).lower():
            print("可能是OpenAI API密钥问题，请检查环境变量OPENAI_API_KEY是否正确设置")
        input("按回车键退出...")