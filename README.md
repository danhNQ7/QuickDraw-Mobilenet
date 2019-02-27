# Quick, Draw! Doodle Recognition Challenge

## Giới thiệu đề tài
Đây là 1 challenge của Google trên Kaggle, dataset lấy từ trò chơi “Quick, Draw !” của Google là các bản vẽ. Mục tiêu đề tài là build được 1 recognizer học từ dataset của trò chơi “Quick, Draw” giúp nhận dạng object từ các bản vẽ tay
**Input:** Đầu vào là các bản vẽ đc ghi lại dưới dạng vector thời gian
**Output:** Đầu ra là dự đoán đối tượng hay vật thể có trong bức vẽ
![alt](https://github.com/danhNQ7/QuickDraw-Mobilenet/blob/master/images/intro.jpg)
## Tập dữ liệu
### Dataset
Là bản vẽ của 340 objects ( apple, bee, calendar,...) bao gồm 2 loại data là data_raw và data_simplified. 
*	data_raw: bộ dataset nguyên gốc từ “Quick,Draw” thể hiện chính xác nét vẽ từ người dùng
*	data_simplified: bộ dataset chứa thông tin giống data_raw nhưng kích thước nhỏ nhờ giảm bớt các thông tin và các điểm vector không cần thiết
### Visualize data
Mỗi dòng trong file csv có dạng: 
*	“drawing” : thể hiện là nét vẽ với tọa độ x,y
*	“key_id” : id riêng biệt cho mỗi bức vẽ
*	“word” : tên của object
*	“countrycode” : mã quốc gia của người vẽ
*	“timestamp” : thời gian ghi bản vẽ
*	“recognized” : bản vẽ được nhận dạng đúng hay sai
## Mô hình đã huấn luyện
### Lựa chọn và xây dựng mô hình
*	Sử dụng mô hình MobileNet của Keras
*	Lý do: Vì MobileNet nhỏ gọn cho hiệu năng rất tốt, chạy rất nhanh, thời gian hạn chế
### Huấn luyện mô hình
Định nghĩa model hyperparameters:
*	step = 2000 
*	batchsize = 256
*	epochs = 15
*	n_classes= 340
*	LR = 0.0001
Sử dụng GPU :Tesla K20c (5GB)
Thời gian train: 1020s / epoch
### Đánh giá mô hình
Sử dụng độ đo Mean Average Precision @ 3 (MAP@3)

![alt](https://github.com/danhNQ7/QuickDraw-Mobilenet/blob/master/images/mAp.jpg)
![alt](https://github.com/danhNQ7/QuickDraw-Mobilenet/blob/master/images/result.jpg)
## Demo
Nhóm đã train được 1 model từ tập dataset, đồng thời đã viết được ứng dụng recognizer cho chữ viết tay. Sau đây là kết quả demo :

![alt](https://github.com/danhNQ7/QuickDraw-Mobilenet/blob/master/images/demo_1.jpg)
![alt](https://github.com/danhNQ7/QuickDraw-Mobilenet/blob/master/images/demo_2.jpg)
![alt](https://github.com/danhNQ7/QuickDraw-Mobilenet/blob/master/images/demo_3.jpg)
