import torch
import re
import numpy as np
import html
from gensim.models import Word2Vec

def extract_type(label: str):
    """
    Trích xuất loại nút từ nhãn (label). 
    Ví dụ: '<(METHOD, ...' -> 'METHOD'
    """
    if not label: 
        return "UNKNOWN"
    
    # Giải mã các ký tự HTML (&lt; -> <)
    s = html.unescape(str(label)).strip()
    
    # REGEX MỚI: Hỗ trợ cả định dạng <TYPE... và <(TYPE... của Joern
    m = re.match(r"^<\s*\(?\s*([A-Z0-9_]+)", s, re.IGNORECASE)
    if m:
        node_type = m.group(1).upper()
    else:
        return "UNKNOWN"

    # Danh sách các loại nút chuẩn để AI tập trung học
    standard_types = {
        'METHOD', 'METHOD_RETURN', 'PARAM', 'LOCAL', 'BLOCK', 
        'CALL', 'IDENTIFIER', 'LITERAL', 'RETURN', 'UNKNOWN',
        'CONTROL_STRUCTURE', 'JUMP_TARGET', 'FIELD_IDENTIFIER'
    }

    return node_type if node_type in standard_types else "UNKNOWN"

class CombinedW2VEncoder:
    def __init__(self, w2v_model_path):
        """Khởi tạo và nạp mô hình Word2Vec"""
        try:
            self.w2v_model = Word2Vec.load(w2v_model_path)
            self.w2v_dim = self.w2v_model.vector_size
        except Exception as e:
            print(f"❌ Không thể nạp Word2Vec: {e}")
            raise
            
        self.type_vocab = {}
        self.fitted = False

    def fit(self, all_nodes_lists):
        """Học danh sách các loại nút có trong dữ liệu"""
        types = set()
        for nodes in all_nodes_lists:
            for n in nodes:
                label = n.get("attrs", {}).get("label", "")
                types.add(extract_type(label))
        
        # Sắp xếp để đảm bảo thứ tự index luôn cố định (Rất quan trọng)
        self.type_vocab = {t: i for i, t in enumerate(sorted(list(types)))}
        self.fitted = True
        print(f"✅ [Encoder] Đã học {len(self.type_vocab)} loại nút: {list(self.type_vocab.keys())}")

    @property
    def feat_dim(self):
        """Tổng chiều dài vector = Chiều Word2Vec + Chiều One-hot loại nút"""
        return self.w2v_dim + len(self.type_vocab)

    def tokenize(self, code):
        """Tách code thành các từ để nạp vào Word2Vec"""
        if not code: return []
        code = html.unescape(str(code))
        # Chỉ lấy các ký tự chữ cái và dấu gạch dưới
        return re.findall(r'[a-zA-Z_]\w*', code.lower())

    def __call__(self, nodes):
        """Biến danh sách nút thành Tensor đặc trưng"""
        if not self.fitted: 
            raise RuntimeError("Encoder chưa fit! Hãy gọi hàm fit() trước.")
            
        features = []
        for n in nodes:
            label = n.get("attrs", {}).get("label", "")
            
            # --- TRÍCH XUẤT CODE TEXT (Phần nội dung code thực tế) ---
            # Ưu tiên lấy sau thẻ <BR/>, nếu không có thì lấy sau dấu phẩy
            if '<BR/>' in label:
                code_text = label.split('<BR/>', 1)[1].rstrip('>')
            elif ',' in label:
                code_text = label.split(',', 1)[1].rstrip('>')
            else:
                code_text = label.strip('<> ')

            # 1. Lấy vector ngữ nghĩa (Word2Vec)
            tokens = self.tokenize(code_text)
            vectors = [self.w2v_model.wv[t] for t in tokens if t in self.w2v_model.wv]
            
            if vectors:
                v_w2v = np.mean(vectors, axis=0)
            else:
                v_w2v = np.zeros(self.w2v_dim, dtype=np.float32)

            # 2. Lấy vector cấu trúc (One-hot Encoding cho loại nút)
            node_type = extract_type(label)
            v_type = np.zeros(len(self.type_vocab), dtype=np.float32)
            # Nếu loại nút lạ, mặc định về UNKNOWN (index thường là 0)
            v_type[self.type_vocab.get(node_type, 0)] = 1.0

            # 3. Ghép nối thành 1 vector duy nhất
            combined_vec = np.concatenate([v_w2v, v_type])
            features.append(combined_vec)

        return torch.tensor(np.array(features), dtype=torch.float32)