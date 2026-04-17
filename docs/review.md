Sau khi họp team để bàn về những gì đã xây dựng, tôi có được những thông tin sau.

1. Lka-suggest là folder chứ một repo mà chị tên lka trong team tìm được và đã chạy test thử. Trong đó thể hiện khá ấn tượng việc trích xuất mạch máu, động mạch và tĩnh mạch, thể hiện bằng xanh và đỏ, overlap thể hiện bằng xanh lá cây. Và hình như có một điểm nữa được thể hiện mà tôi thấy rất hợp lý trong repo này, Đó chính là việc phân loại ảnh dùng được và ảnh không dùng được. Thì đúng là bài toán này của chúng ta không phải ảnh nào cũng xử lý -> đây là vấn đề y khoa, tức là ảnh đầu vào phải chuẩn, khác với bài toán nhận diện biển số xe chả hặn. Không những thế, hình như repo này còn xử lý việc nhận diễn đĩa thị tốt thì phải.

-> vậy từ đây tôi có một số điều:
- Trong repo của chúng ta, đầu tiên, chúng ta có xử lý phân tĩnh mạch và động mạch để xử lý các bước đằng sau để đi đến kết luận cuối cùng có dấu hiệu đột quỵ hoặc không hay không.
- Tôi khá chắc thuật toán của chúng ta khác so với repo kia, nhưng vì chưa kiểm chứng cái nào hơn, và hai anh chị kia lại đang nghiêng về hướng repo đấy, nên bạn nghĩ sao khi thay vào trong project này của mình?
- Phần xử lý đĩa thị của chúng ta cũng đang chưa chuẩn lắm, vì nó chỉ đang lấy phần sáng nhất của khung ảnh thôi hay sao ý. nên nếu có thể cải thiện để áp dụng tốt vào pipeline, hãy làm.
- nhưng nếu như không phải do ảnh không tốt, ở đây chị đấy nhận định là ảnh không tốt tức là không nhìn thấy rõ các đường mạch máu. nhưng thực chất, việc không nhìn thấy rõ mạch máu cũng là một dấu hiệu đột quỵ mà! nó có thể rơi vào giảm mật độ mạch ấy.
-> Vậy nên xử lý như nào?
Tôi đưa ra hai ý kiến:
1. pipeline hiện giờ của chúng ta đang là ảnh nào cũng chơi, có thể giữ nguyên, rồi áp dụng chỉnh tay chả hặn.
Nhưng việc chỉnh tay sẽ tự sinh ra các vấn đề bên trong nó -> chỉnh tay liệu còn chuẩn y khoa không, liệu còn dùng được ảnh với ML không -> có thể không khả thi
2. Lọc ảnh và bắt đưa về mức độ phù hợp mới xử lý -> có thể bỏ sót các bệnh nhân có dấu hiệu đột quỵ
3. Một phương pháp tối ưu chuẩn nhất mà tôi chưa nghĩ ra.

Vậy Bạn thấy như nào?