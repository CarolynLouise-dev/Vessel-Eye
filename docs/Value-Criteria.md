Bài cuối kỳ cần có:
1. Slide báo cáo phần lý thuyết 
2. Code demo 
Yêu cầu:
1.  Chủ đề của bài tập lớn có tính thực tế, giải quyết một vấn đề trong thực tế liên quan đến hệ thống, sản phẩm phần mềm ứng dụng xử lý hình ảnh/video, có thể sử dụng các thuật toán của computer vision như nhận dạng đối tượng, ....
2.  Chủ đề cần đảm bảo:
a.     Có tính thực tiễn, ứng dụng rõ ràng trong thực tế (VD giám sát an ninh, y tế, giao thông, nông nghiệp, giáo dục, sản xuất, môi trường…).
b.    Bao gồm ít nhất một phần phân tích bài toán thực tế có liên quan đến xử lý ảnh/video hoặc computer vision.
c.     Có tiềm năng phát triển thành sản phẩm ứng dụng như website, ứng dụng di động, desktop, hệ thống nhúng hoặc API tích hợp.
3.  Các mô tả về giải pháp đã chọn, bao gồm:
a.     Lý do chọn chủ đề.
b.    Mục tiêu và phạm vi thực hiện về mặt chức năng (các tác vụ CV: detection, classification, segmentation, tracking, OCR, …), phi chức năng (độ chính xác, tốc độ thời gian thực, khả năng mở rộng, deploy…).
c.     Phân tích, thiết kế pipeline xử lý ảnh, computer vision, lựa chọn/thiết kế mô hình (OpenCV, CNN, YOLO, ViT, Segment Anything…), thu thập/huấn luyện/transfer learning dữ liệu, đánh giá mô hình; đồng thời thiết kế mô hình cơ sở dữ liệu (nếu có) và cài đặt trên một hệ quản trị CSDL cụ thể.
d.    Xây dựng được kịch bản, giao diện mô phỏng chức năng của sản phẩm (demo web/app hoặc chương trình chạy trực tiếp).
e.     Kết quả mong đợi sau khi hoàn thành bài tập lớn.
NỘI DUNG BÁO CÁO
1.     Giới thiệu chung
-     Lý do chọn đề tài
-     Mục tiêu và phạm vi thực hiện
-     Ý nghĩa và ứng dụng thực tế của sản phẩm
2.     Phân tích bài toán thực tế
-     Phân tích nghiệp vụ, mô tả yêu cầu chức năng và phi chức năng của sản phẩm
-     Đánh giá ưu nhược điểm của các công nghệ và mô hình xử lý ảnh, computer vision được lựa chọn cho dự án
3.     Thiết kế hệ thống và pipeline 
a.     Thiết kế pipeline xử lý 
-     Phân tích yêu cầu, xác định các module xử lý hình ảnh/video (pre-processing, feature extraction, model chính, post-processing)
-     Thiết kế kiến trúc tổng thể của hệ thống (data flow, model architecture) và vẽ
sơ đồ khối (Block Diagram) hoặc flowchart pipeline
-     Lựa chọn và thiết kế mô hình, chiến lược huấn luyện/transfer learning
b.    Thiết kế cơ sở dữ liệu và cài đặt trên hệ quản trị CSDL (nếu có)
-     Thiết kế cơ sở dữ liệu
-     Thiết kế và triển khai các bảng lưu trữ dữ liệu hình ảnh, metadata, kết quả nhận dạng, thông tin người dùng…
4.     Kết quả thực hiện
a.     Trình bày sản phẩm ứng dụng: giao diện người dùng, tính năng chính, kịch bản minh họa kết quả mô phỏng 
b.    Đánh giá kết quả đạt được
·       Đánh giá hiệu suất mô hình (accuracy, precision, recall, mAP, FPS, …)
·       Đánh giá ưu nhược điểm của pipeline và mô hình trong bối cảnh bài toán thực tế
·       Đánh giá ưu nhược điểm của cơ sở dữ liệu và hệ thống lưu trữ
c.     Những hạn chế của đề tài
d.    Hướng phát triển trong tương lai
Tài liệu tham khảo

---

| **Tiêu chí** | **Trọng số** | **Mô tả mức chất lượng (8.5 → 10 điểm)** |
| --- | --- | --- |
| **1. Mục tiêu và bài toán cần giải quyết** | 10 | Xuất hiện bài toán rõ ràng, đầy đủ, mạch lạc. Nêu chính xác mục tiêu, phạm vi, yêu cầu đầu vào/đầu ra, ý nghĩa thực tiễn. Ngôn ngữ chuyên nghiệp, dễ hiểu, có dẫn chứng hoặc ví dụ minh họa cụ thể. |
| **2. Các khó khăn của bài toán** | 10 | Nêu rõ ràng, đầy đủ và sâu sắc các khó khăn chính của bài toán (dữ liệu, tính phức tạp, nhiều, mất cân bằng, yêu cầu thời gian thực,…). Phân tích cụ thể, có dẫn chứng hoặc ví dụ thực tế, liên hệ chặt chẽ với bài toán đang giải quyết. |
| **3. Phương pháp giải quyết** | 20 | Trình bày phương pháp phù hợp, có cơ sở lý thuyết vững chắc, giải thích rõ ràng cách áp dụng vào bài toán. Thể hiện sự hiểu biết sâu về mô hình, thuật toán hoặc kỹ thuật sử dụng. Có minh họa hoặc ví dụ cụ thể. |
| **4. Kết quả đạt được** | 20 | Kết quả thể hiện rõ hiệu quả của phương pháp, có so sánh với baseline hoặc các phương pháp khác. Phân tích định lượng và định tính, có biểu đồ hoặc bảng minh họa. Giải thích hợp lý, logic, và thuyết phục. |
| **5. Đánh giá và thảo luận** | 20 | Đánh giá toàn diện, chỉ ra điểm mạnh và hạn chế của phương pháp. Có đề xuất cải tiến hoặc hướng phát triển tiếp theo. Thảo luận sâu sắc, thể hiện tư duy phản biện và khả năng tổng hợp. |
| **6. Trình bày và hình thức báo cáo** | 20 | Báo cáo trình bày rõ ràng, mạch lạc, bố cục hợp lý. Dùng ngôn ngữ chuyên nghiệp, chính tả và định dạng chuẩn. Có biểu đồ, hình ảnh minh họa phù hợp, dễ hiểu, thẩm mỹ. |
