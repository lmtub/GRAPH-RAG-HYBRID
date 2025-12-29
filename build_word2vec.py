import os
import json
import logging
import re
import html # Thư viện để giải mã ký tự đặc biệt
from gensim.models import Word2Vec
from tqdm import tqdm

# Cấu hình log để theo dõi tiến độ
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class CPGCorpus:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def extract_pure_code(self, node):
        """Trích xuất mã nguồn từ định dạng <TYPE, ID<BR/>CODE>"""
        attrs = node.get('attrs', {})
        if not isinstance(attrs, dict): return ""
        
        label = attrs.get('label', '')
        if '<BR/>' in label:
            # 1. Lấy phần nội dung sau dấu <BR/>
            # Ví dụ: lấy được "static av_cold int...>"
            parts = label.split('<BR/>', 1)
            code_part = parts[1]
            
            # 2. Loại bỏ dấu '>' cuối cùng của label Joern
            if code_part.endswith('>'):
                code_part = code_part[:-1]
            
            # 3. Giải mã các ký tự HTML (ví dụ &lt; thành <)
            code_part = html.unescape(code_part)
            return code_part
        return ""

    def tokenize(self, code):
        if not code or code == "&lt;empty&gt;":
            return []
        
        # Loại bỏ dấu ba chấm "..." do Joern cắt ngắn code
        code = code.replace("...", "")
        
        # Tách từ: Chỉ giữ lại chữ cái, số và dấu gạch dưới
        tokens = re.findall(r'[a-zA-Z_]\w*', code)
        
        # CHỈ GIỮ LẠI:
        # 1. Từ có độ dài từ 2 ký tự trở lên (loại bỏ biến rác như i, j, x)
        # 2. Loại bỏ các từ quá dài hoặc bị cắt vụn (ví dụ > 30 ký tự thường là lỗi)
        clean_tokens = [t.lower() for t in tokens if 2 <= len(t) <= 30]
        
        return clean_tokens

    def __iter__(self):
        """Duyệt qua toàn bộ thư mục data/cpg để lấy dữ liệu"""
        # Đếm tổng số thư mục để hiển thị thanh tiến trình (progress bar)
        all_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for folder_name in tqdm(all_dirs, desc="Đang đọc dữ liệu CPG"):
            file_path = os.path.join(self.root_dir, folder_name, 'nodes.json')
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        nodes = json.load(f)
                        for node in nodes:
                            code_content = self.extract_pure_code(node)
                            tokenized = self.tokenize(code_content)
                            if tokenized:
                                yield tokenized
                except Exception:
                    continue

def main():
    # Đường dẫn thư mục chứa các folder .cpg14
    cpg_data_path = 'data/cpg'
    model_save_path = 'word2vec_cpg.model'

    print("--- Khởi tạo quá trình quét dữ liệu ---")
    corpus = CPGCorpus(cpg_data_path)

    # Huấn luyện mô hình Word2Vec
    # vector_size=100: mỗi từ biến thành dãy 100 con số
    # min_count=1: học cả những từ xuất hiện 1 lần (vì code C có nhiều biến đặc thù)
    print("--- Đang huấn luyện (Training)... ---")
    model = Word2Vec(
        sentences=corpus, 
        vector_size=128, 
        window=5, 
        min_count=5, 
        workers=4,
        epochs=20
    )

    # Lưu mô hình
    model.save(model_save_path)
    print(f"--- Hoàn thành! Đã lưu tại: {model_save_path} ---")

    # Thử nghiệm nhanh
    test_word = 'static'
    if test_word in model.wv:
        print(f"Kiểm tra: Từ '{test_word}' đã được học.")

if __name__ == "__main__":
    main()