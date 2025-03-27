from typing import Optional, Callable
from ai_translator.model import Model
from .pdf_parser import PDFParser
from .writer import Writer
from ai_translator.utils import LOG

class PDFTranslator:
    def __init__(self, model: Model):
        self.model = model
        self.pdf_parser = PDFParser()
        self.writer = Writer()

    def translate_pdf(self, pdf_file_path: str, file_format: str = 'PDF', target_language: str = '中文', output_file_path: str = None, pages: Optional[int] = None):
        self.book = self.pdf_parser.parse_pdf(pdf_file_path, pages)
        total_contents = sum(len(page.contents) for page in self.book.pages)
        processed_contents = 0
        translated_text = []

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
