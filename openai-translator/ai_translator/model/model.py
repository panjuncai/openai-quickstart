from ai_translator.book import ContentType

class Model:
    def make_text_prompt(self, text: str, target_language: str) -> str:
        language_prompts = {
            "中文": "翻译为中文：",
            "法语": "Translate to French:",
            "日语": "翻訳を日本語に："
        }
        prompt = language_prompts.get(target_language, f"翻译为{target_language}：")
        return f"{prompt}{text}"

    def make_table_prompt(self, table: str, target_language: str) -> str:
        language_prompts = {
            "中文": "翻译为中文，保持表格格式：",
            "法语": "Translate to French, maintain table format:",
            "日语": "翻訳を日本語に、表の形式を維持："
        }
        prompt = language_prompts.get(target_language, f"翻译为{target_language}，保持表格格式：")
        return f"{prompt}\n{table}"

    def translate_prompt(self, content, target_language: str) -> str:
        if content.content_type == ContentType.TEXT:
            return self.make_text_prompt(content.original, target_language)
        elif content.content_type == ContentType.TABLE:
            return self.make_table_prompt(content.get_original_as_str(), target_language)

    def make_request(self, prompt):
        raise NotImplementedError("子类必须实现 make_request 方法")
