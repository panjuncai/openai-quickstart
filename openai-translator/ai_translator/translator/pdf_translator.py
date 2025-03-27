from typing import Optional, Callable
from ai_translator.model import Model
from .pdf_parser import PDFParser
from .writer import Writer
from ai_translator.utils import LOG
import os

class PDFTranslator:
    def __init__(self, model: Model):
        self.model = model
        self.pdf_parser = PDFParser()
        self.writer = Writer()

    def translate_pdf(self, pdf_file_path: str, file_format: str = 'PDF', target_language: str = '中文', output_file_path: str = None, pages: Optional[int] = None):
        """
        翻译PDF文件
        :param pdf_file_path: PDF文件路径
        :param file_format: 输出文件格式
        :param target_language: 目标语言，支持：中文、法语、日语
        :param output_file_path: 输出文件路径
        :param pages: 要翻译的页数
        :return: 翻译后的文本
        """
        self.book = self.pdf_parser.parse_pdf(pdf_file_path, pages)
        total_contents = sum(len(page.contents) for page in self.book.pages)
        processed_contents = 0
        translated_text = []

        # 根据目标语言设置输出文件名
        if output_file_path is None:
            base_name = os.path.splitext(pdf_file_path)[0]
            output_file_path = f"{base_name}_translated_{target_language}.{file_format.lower()}"

        for page_idx, page in enumerate(self.book.pages):
            for content_idx, content in enumerate(page.contents):
                prompt = self.model.translate_prompt(content, target_language)
                LOG.debug(prompt)
                translation, status = self.model.make_request(prompt)
                LOG.info(translation)
                
                # Update the content in self.book.pages directly
                self.book.pages[page_idx].contents[content_idx].set_translation(translation, status)
                translated_text.append(translation)
                
                processed_contents += 1
                LOG.info(f"翻译进度: {processed_contents}/{total_contents}")

        self.writer.save_translated_book(self.book, output_file_path, file_format)
        return "\n\n".join(translated_text)
