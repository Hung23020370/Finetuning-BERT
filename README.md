# Finetuning-BERT

## Thành viên trong nhóm
- Đồng Mạnh Hùng.
- Đoàn Quang Huy.
- Nguyễn Đình Khải. 

### Phân công nhiệm vụ
- Slide tham khảo: [BERT Finetuning](https://www.figma.com/slides/HVqfuP4ZpmdZO6IBifh2uc/Data-Science-Project---Recipe-Site-Traffic?node-id=1-305&t=kwlcwnsjv97iZGLf-1).
- Báo cáo: [Báo cáo](docs/report/Báo%20cáo.pdf)
- Code về hai phần liên quan đến BERT: Đồng Mạnh Hùng, Nguyễn Đình Khải.
- Tìm hiểu, thu thập và phân tích dữ liệu: Đoàn Quang Huy. 

## Nội dung chính

### Tóm tắt bài toán

Nghiên cứu sử dụng mô hình BERT - một mô hình Transformer 2 chiều - để tính toán độ chính xác
trong công tác dự đoán cảm xúc của người tiêu dùng trên nền tảng Amazon thông qua những lời bình luận 
và trong công tác dự đoán chủ đề của một câu hỏi trên diễn đàn Yahoo!, 
đồng thời thực hiện tinh chỉnh (Fine-tuning) để giúp BERT học thêm về ngữ cảnh và đặc thù của bài toán, từ đó cải thiện độ hiệu quả.

### Dữ liệu

Trong nghiên cứu này, chúng tôi thực hiện hai nhiệm vụ chính. Đầu tiên, chúng tôi tiền huấn luyện mô hình ngôn ngữ trên tập dữ liệu Wikipedia tiếng Anh để học các biểu diễn ngữ nghĩa tổng quát. Sau đó, chúng tôi áp dụng phương pháp fine-tuning dựa trên mô hình đã tiền huấn luyện cho hai tác vụ cụ thể: phân tích cảm xúc và phân loại chủ đề văn bản.Tập dữ liệu dùng cho tác vụ phân loại chủ đề được xây dựng dựa trên tập dữ liệu Yahoo Answers do Xiang Zhang và cộng sự phát triển, từng được sử dụng trong nghiên cứu về Character-level Convolutional Networks for Text Classification, công bố tại Hội thảo về Hệ thống xử lý thông tin thần kinh (Neural Information Processing Systems - NIPS) năm 2015 .

#### Wikipedia
- Dữ liệu tiền huấn luyện được sử dụng trong nghiên cứu này là tập Wikipedia tiếng Anh phiên bản tháng 3 năm 2022, được cung cấp thông qua nền tảng Hugging Face tại địa chỉ \url{https://huggingface.co/datasets/wikipedia}. Đây là bản trích xuất đã được xử lý trước, bao gồm nội dung văn bản thuần từ các bài viết trên Wikipedia, loại bỏ các thẻ đánh dấu định dạng, hình ảnh, và siêu liên kết, giúp thuận tiện cho việc sử dụng trong huấn luyện mô hình ngôn ngữ. Tập dữ liệu này bao gồm hàng triệu bài viết với độ dài và chủ đề phong phú, phản ánh đa dạng các lĩnh vực từ khoa học, công nghệ đến văn hóa và xã hội.
#### Amazon Review Polarity

- Dữ liệu được xây dựng bằng việc lấy các đánh giá 1 sao và 2 sao là tiêu cực (label 1), 4 sao và 5 sao là tích cực (label 2). Các đánh giá 3 sao được lược bỏ. 
- Mỗi label sẽ có 1.8 triệu mẫu huấn luyện và 200,000 mẫu kiểm tra tương ứng. 
- Dữ liệu được lưu dưới dạng .csv và bao gồm 3 cột: Class Index (cột label), Review Title, Review Text.

![Data Sentiment](figures/data_sentiment.png)

#### Yahoo Answers

- Dữ liệu được xây dựng với việc sử dụng 10 chủ đề chính: Văn hóa và Xã hội; Khoa học và Toán học; Sức khỏe; Giáo dục và  Tham chiếu; Máy tính và Internet; Thể thao; Kinh doanh và Tài chính; Giải trí và Âm nhạc; Gia đình và các Mối quan hệ; Chính trị. 
- Mỗi chủ đề bao gồm 140,000 mẫu huấn luyện và 6,000 mẫu kiểm tra tương ứng. 
- Dữ liệu về câu trả lời: Chỉ lấy những câu trả lời hay nhất và bám sát với chủ đề. 
- Dữ liệu được lưu dưới dạng .csv và bao gồm 4 cột: Class Index (cột label), Question Title, Question Content, Best Answer. 

![Data Topic](figures/data_topic.png)
### Cách cài đặt và chạy ứng dụng demo

1. **Cài đặt thư viện cần thiết**  
- Mở terminal và chạy:
```bash
pip install torch streamlit transformers
```
2. **Chạy ứng dụng** 
- Mở terminal
- Di chuyển đến thư mục chứa mã nguồn, ví dụ nếu thư mục ở ổ C, trong folder Finetuning-BERT:
```bash
cd C:\Finetuning-BERT
```
- Chạy ứng dụng:
```bash
streamlit run demo.py
```
