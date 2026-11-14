# Báo cáo Day 7 — Agentic RAG

- Họ tên: Trần Đăng Khoa
- Mã học viên: 2A202603364
- Lớp: L3A
- Phạm vi: phần cơ bản, không thực hiện thử thách nâng cao.

## 1. Dữ liệu và benchmark

### Dữ liệu

Sử dụng lại dữ liệu và vector store Day 6:

- Tài liệu: World Bank, World Development Report 2016:
  Digital Dividends — Overview.
- PDF gốc có 58 trang; corpus giữ phần văn bản chính từ 33 trang
  nội dung, không phải toàn bộ nội dung của 33 trang đó.
- 132 chunk sentence-aware của strategy B; giữ ID, nguồn, trang
  PDF và metadata.
- Embedding: text-embedding-3-small, 1536 chiều.
- Vector store: PostgreSQL/pgvector, bảng day6_chunks,
  strategy B; tìm kiếm cosine chính xác.
- Driver: pg8000.
- Agent và verifier: gpt-4.1-mini-2025-04-14.

Nguồn PDF:
https://documents1.worldbank.org/curated/en/961621467994698644/pdf/102724-WDR-WDR2016Overview-ENGLISH-WebResBox-394840B-OUO-9.pdf

Attribution: World Bank. 2016. World Development Report 2016:
Digital Dividends. Overview booklet. Washington, DC.
Văn bản dẫn xuất theo CC BY 3.0 IGO; thành phần bên thứ ba có thể có
quyền riêng. Không đưa PDF gốc vào bài nộp.

Thông tin xuất bản là năm 2016; không coi số liệu là tình hình hiện tại.
Corpus còn từ bị ngắt do PDF, thiếu một số tiêu đề, bảng, hình và phần
ngoài phạm vi extraction Day 6. Không thực hiện OCR hoặc multimodal RAG.

### Benchmark

| Split | Có đáp án | Thiếu bằng chứng | Tổng |
|---|---:|---:|---:|
| Dev | 15 | 2 | 17 |
| Test mới Day 7 | 5 | 2 | 7 |
| Tổng | 20 | 4 | 24 |

- Dev: starter/data/day6/dev_draft.json.
- Test: benchmark/test_questions.json.
- Test mới hỏi các sự kiện khác với câu dev và test Day 6.
- Một số trang/chunk dùng chung giữa dev và test; test chủ yếu là
  câu hỏi sự kiện trong một đoạn, chưa đại diện cho tác vụ khó.
- Gold, đáp án, answerable và ID evidence chuẩn chỉ dùng đánh giá;
  không truyền vào agent hoặc retrieval.
- Chọn cấu hình trên dev; lưu artifacts/final_selection.json trước
  khi tạo và chạy test.
- benchmark/test_manifest.json lưu hash test, corpus và cấu hình.
- Không chỉnh hệ thống theo kết quả test.

### Tiêu chí chấm

- Hit@1 và Evidence Hit@3: evidence hợp lệ xuất hiện ở cutoff tương ứng;
  chỉ tính câu có đáp án.
- Context Evidence Rate: context đủ evidence theo quy tắc đã khai báo.
- q11 dev cần cả hai chunk để liên kết “this sector” với ngành ICT;
  báo riêng trường hợp nhiều chunk.
- Answer grade: 0 = sai/không được hỗ trợ; 1 = một phần;
  2 = đúng và đủ bằng chứng.
- Citation phải có trong context và hỗ trợ nội dung được dẫn.
- Abstention cần thực sự từ chối yêu cầu thiếu bằng chứng,
  không bịa hoặc dùng dữ kiện khác thay đáp án.
- Đọc câu trả lời và nguồn/context, ghi nhãn và lý do theo từng câu.
  LLM verifier hỗ trợ pipeline, không thay nhãn đánh giá.
- Script tổng hợp chất lượng loại câu error khỏi mẫu số và báo lỗi riêng.
  Vì vậy điểm trung bình có điều kiện không phải accuracy toàn bộ.
- Đối chiếu nguồn dựa trên văn bản trích từ PDF và context đã lưu;
  không khẳng định đã kiểm tra trực quan toàn bộ bố cục PDF.

## 2. Retrieval, agent và context

### Ba phương pháp

1. Dense: embedding query và tìm cosine trên pgvector Day 6.
2. TF-IDF: tách từ alphanumeric viết thường; TF log, IDF làm trơn,
   chuẩn hóa L2 và cosine; không bỏ stopword.
3. Hybrid RRF: lấy tối đa 10 candidate từ mỗi nhánh Dense và TF-IDF,
   hợp nhất theo 1/(60 + rank), trọng số hai nhánh bằng nhau.

Cả ba dùng cùng 132 chunk. Không cộng trực tiếp score Dense và TF-IDF.
Các lần so sánh không áp dụng metadata filter; phạm vi nêu trong câu hỏi
vẫn giữ nguyên.

### Agent

- Model gọi search để truy xuất, sau đó gọi finish để trả lời hoặc từ chối.
- Lượt tìm đầu thực thi nguyên văn câu hỏi; lưu riêng query model đề xuất.
- Các lượt sau có thể viết lại query.
- Tối đa 3 lượt search; tối đa 2 lượt sửa định dạng citation.
- Kiểm tra citation ID và citation inline; sai định dạng vẫn có thể
  dẫn đến error sau khi hết lượt sửa.
- Verifier kiểm tra câu trả lời có đáp ứng điều kiện năm, đối tượng,
  chỉ số và đơn vị hay không bằng context thực tế.
- Verifier trả quotation cần kiểm chứng; chuẩn hóa khoảng trắng và
  một số từ bị ngắt khi so khớp, giữ nguyên văn bản gốc trong trace.
- API hoặc validation lỗi được ghi error, không thay bằng mock.
- Nội dung tài liệu được xem là dữ liệu, không phải chỉ dẫn.

Context được đóng gói theo thứ tự kết quả và lượt tìm, loại ID trùng,
giữ thông tin nguồn/trang; bỏ block không vừa budget.
Budget tính bằng ký tự, gồm cả header nguồn.
Lịch sử tool cũ được thay thông báo để model dùng context tích lũy mới nhất.

Hạn chế: ưu tiên đoạn tìm trước có thể làm lượt tìm sau không thêm được
evidence khi budget đã đầy. Pipeline cho phép nhiều lượt, nhưng các
lần chạy dev cuối và test thực tế đều dùng một lượt search mỗi câu.

### Cấu hình thử nghiệm

| Cấu hình | top-k | Budget ký tự | Giả thuyết |
|---|---:|---:|---|
| Dense baseline | 3 | 2600 | Mốc so sánh |
| TF-IDF | 3 | 2600 | Từ khóa có thể tìm đúng thuật ngữ |
| Hybrid RRF | 3 | 2600 | Kết hợp ngữ nghĩa và từ khóa |
| Dense more context | 6 | 5200 | Giữ thêm evidence bổ trợ |

Hybrid và Dense more context là hai cấu hình thử cải thiện baseline.
Không giả định mọi thay đổi đều tăng chất lượng.
Thử nghiệm tăng context thay đổi đồng thời top-k và budget,
chưa tách riêng tác động từng yếu tố.

## 3. Kết quả dev

### Retrieval sơ bộ

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence Rate |
|---|---|---|---|
| Dense baseline | 12/15 — 80,0% | 13/15 — 86,7% | 13/15 — 86,7% |
| TF-IDF | 12/15 — 80,0% | 14/15 — 93,3% | 14/15 — 93,3% |
| Hybrid RRF | 11/15 — 73,3% | 13/15 — 86,7% | 13/15 — 86,7% |
| Dense more context | 12/15 — 80,0% | 13/15 — 86,7% | 14/15 — 93,3% |

Đây là phép chấm tự động theo bộ gold tại thời điểm thí nghiệm,
không phải điểm câu trả lời agent. Đánh giá nội dung về sau xác nhận
trang 30 là evidence phù hợp cho q02 Hybrid dù không khớp gold ban đầu.
Giữ nguyên kết quả sơ bộ, không sửa điểm cũ để che khác biệt.
Đối với q11, tiêu chí đủ hai chunk làm Hit@1 bằng 0 theo định nghĩa;
summary retrieval có nhóm một chunk và nhiều chunk riêng.

### Agent sau verifier và sửa luồng validation

| Cấu hình | Grade 0/1/2 | Trung bình /2 | Citation hợp lệ | Có citation / answered | Từ chối đúng | Từ chối nhầm | Lỗi |
|---|---|---:|---|---|---|---|---|
| Dense | 0/3/12; N=15 | 1,800 | 15/18 — 83,3% | 15/15 | 1/1 | 0/15 | 1/17 |
| TF-IDF | 0/2/10; N=12 | 1,833 | 11/13 — 84,6% | 12/12 | 2/2 | 0/12 | 3/17 |
| Hybrid | 0/2/12; N=14 | 1,857 | 12/14 — 85,7% | 14/14 | 2/2 | 0/14 | 1/17 |
| Dense more context | 0/1/12; N=13 | 1,923 | 12/14 — 85,7% | 13/13 | 2/2 | 0/13 | 2/17 |

Tool use trên các câu không lỗi lần lượt là 16/16, 14/14, 16/16,
15/15. Tất cả 17 câu mỗi cấu hình thực tế có một lượt search.

Dense có một câu thiếu bằng chứng bị error: không diễn giải 1/1 là
đã từ chối đúng cả hai câu thiếu bằng chứng.

Context thủ công trên câu có đáp án không lỗi:
- Dense: mức 0/1/2 = 0/2/13, N=15.
- TF-IDF: 0/1/11, N=12.
- Hybrid: 0/1/13, N=14.
- Dense more context: 0/0/13, N=13.

Các phân bố này có mẫu số khác nhau và không thay thế context rate
của toàn bộ benchmark.

## 4. Phân tích lỗi và cải thiện

### Thiếu đoạn bổ trợ — q11

Baseline tìm được 3–5% nhưng thiếu đoạn trước xác định ngành ICT.
Tăng top-k lên 6 và budget lên 5200 giữ được đoạn bổ trợ hạng 4:
context đủ evidence tăng từ 13/15 lên 14/15 ở thí nghiệm retrieval.

Trong agent Dense more context, context đủ nên answer grade đạt 2,
nhưng citation chỉ dẫn chunk chứa con số, chưa bao phủ đủ nhóm
evidence đã quy định. Citation được chấm riêng là chưa đầy đủ.

### Có citation ID nhưng thiếu citation inline

Các lần đầu model đặt ID trong citations nhưng không chèn vào answer.
Bổ sung mô tả schema và tối đa hai lượt sửa định dạng có trace.
Thay đổi giúp nhiều câu hoàn tất nhưng tăng token, độ trễ;
lỗi định dạng vẫn tồn tại, không được coi là hoàn toàn giải quyết.

### Sai điều kiện thời gian và abstention

Baseline trước verifier trả answered cho hai câu dev hỏi năm 2025,
dù chỉ có số liệu lịch sử từ báo cáo 2016.
Thêm kiểm tra đủ bằng chứng bằng model để từ chối khi thiếu điều kiện.

Verifier không bảo đảm đúng: có thể chấp nhận câu thiếu evidence
hoặc đưa quotation không hợp lệ. Giữ các lỗi và nhãn chấm riêng.

### Validation quotation và luồng từ chối

PDF có từ ngắt như “employ- ment”, model trả “employment”.
Chuẩn hóa có giới hạn khi so khớp, không đổi corpus hoặc embedding.

Một số lỗi còn lại do quotation lược đoạn bằng dấu ba chấm hoặc
thay đổi văn bản. Không bỏ kiểm tra để làm điểm tốt hơn.
Dense dev_u01 còn lỗi từ chối kèm citation.
Hybrid q14 còn lỗi quotation của verifier.

### Trả lời dài vượt quá evidence — q15

Câu trả lời đúng ý chính về lương thấp nhưng thêm giải thích chưa
được citation hỗ trợ đầy đủ; chấm answer grade 1 thay vì tự động 2.

## 5. Chọn cấu hình trước test

Chọn Hybrid RRF, top-k 3, budget 2600, tối đa 3 search,
candidate 10 mỗi nhánh, RRF k=60.

Lý do:
- Một lỗi dev, ít hơn TF-IDF và Dense more context.
- Hai câu thiếu bằng chứng được từ chối đúng.
- Dense more context có mean cao hơn sau khi loại lỗi nhưng có hai lỗi
  và dùng nhiều token hơn.
- Không chọn chỉ dựa trên mean có mẫu số khác nhau.

Lưu quyết định và hash trong artifacts/final_selection.json.
Giữ nguyên implementation đã chốt khi chạy test.

## 6. Kết quả test độc lập Day 7

| Chỉ số | Dense baseline | Hybrid RRF |
|---|---|---|
| Hit@1 | 5/5 — 100% | 5/5 — 100% |
| Evidence Hit@3 | 5/5 — 100% | 5/5 — 100% |
| Context đủ evidence | 5/5 — 100% | 5/5 — 100% |
| Context grade 0/1/2 | 0/0/5 | 0/0/5 |
| Answer grade 0/1/2 | 0/0/5 | 0/0/5 |
| Trung bình answer grade | 2,0/2 | 2,0/2 |
| Citation hợp lệ | 5/5 — 100% | 5/5 — 100% |
| Answered có citation | 5/5 | 5/5 |
| Tool use | 7/7 | 7/7 |
| Từ chối đúng | 2/2 | 2/2 |
| Từ chối nhầm | 0/5 | 0/5 |
| Lỗi | 0/7 | 0/7 |
| Tổng lượt search | 7, mỗi câu 1 | 7, mỗi câu 1 |
| Thời gian toàn bộ | 47,875 giây | 37,278 giây |

Hai cấu hình ngang nhau về chất lượng trên bộ test này.
Không kết luận Hybrid luôn nhanh hoặc tốt hơn: chỉ có một lần chạy,
test nhỏ, baseline chạy trước và cache embedding có lợi cho Hybrid.

Ví dụ có bằng chứng:
- t04: Instagram có 13 nhân viên khi Facebook mua năm 2012;
  citation tới chunk d207fb8f... tại trang PDF 26.

Ví dụ thiếu bằng chứng:
- test_u01 yêu cầu mã nguồn thuật toán UPS; agent nêu nguồn không
  cung cấp mã nguồn và không bịa code.
- test_u02 yêu cầu danh sách tên; agent nêu nguồn không có danh sách
  và không bịa tên.

Lỗi minh họa nằm trên dev, như Hybrid q14; không bịa lỗi test khi
lần chạy test thực tế không có lỗi.

## 7. Thời gian, token và chi phí ước tính

Token input đã bao gồm phần cached; khi tính tiền tách cached input.
Usage gồm agent, lượt sửa citation và verifier có phản hồi được lưu.

| Lần chạy | Input | Cached input | Output | Embedding token mới | Thời gian giây | USD ước tính |
|---|---:|---:|---:|---:|---:|---:|
| Dev Dense | 71682 | 33024 | 7641 | 0 | 114,035 | 0,0309912 |
| Dev TF-IDF | 75593 | 2816 | 7880 | 0 | 114,008 | 0,0420004 |
| Dev Hybrid | 75800 | 12288 | 7454 | 0 | 120,684 | 0,0385600 |
| Dev Dense more context | 108880 | 0 | 8276 | 0 | 119,391 | 0,0567936 |
| Test Dense | 25365 | 0 | 1969 | 165 | 47,875 | 0,0132997 |
| Test Hybrid | 24816 | 0 | 1965 | 0 | 37,278 | 0,0130704 |

Đơn giá tham chiếu USD/1 triệu token:
- GPT-4.1 mini: input 0,40; cached input 0,10; output 1,60.
- text-embedding-3-small: input 0,02.

Công thức:
(input - cached) × 0,40/1e6 + cached × 0,10/1e6
+ output × 1,60/1e6 + embedding token mới × 0,02/1e6.

Tổng sáu lần chạy trong bảng khoảng 0,1947153 USD.
Đây không phải tổng hóa đơn dự án: chưa gồm ingestion Day 6,
retrieval sơ bộ, các lần thử/sửa trước đó và request không có usage.
Embedding token bằng 0 ở một lần chạy có thể do dùng cache,
không có nghĩa model embedding miễn phí.

Nguồn đơn giá:
https://developers.openai.com/api/docs/models/gpt-4.1-mini
https://developers.openai.com/api/docs/models/text-embedding-3-small

## 8. Hồ sơ kết quả

Retrieval sơ bộ:
starter/results/retrieval_dev_20261114T085416733522Z/

Dev đã chấm:
- starter/results/agent_dev_dense_baseline_20261114T100948307118Z/
- starter/results/agent_dev_tfidf_20261114T101205688015Z/
- starter/results/agent_dev_hybrid_rrf_20261114T102003609244Z/
- starter/results/agent_dev_dense_more_context_20261114T102334968926Z/

Trong mỗi thư mục có protocol, cases, traces, nhãn reviews.json,
lý do chấm và evaluated_summary.json.

Test đã chấm:
- starter/results/final_test/dense_baseline/
- starter/results/final_test/hybrid_rrf/

Phiên bản chốt:
- artifacts/final_selection.json
- benchmark/test_manifest.json

Các kết quả lịch sử trước cải thiện được giữ riêng, không thay thế
kết quả dev/test cuối hoặc trộn thành một thí nghiệm.

## 9. Kết luận và giới hạn

Đã xây agent thật trên dữ liệu Day 6, so sánh ba phương pháp retrieval,
thử cải thiện trên dev và chốt Hybrid trước test.
Cả Dense và Hybrid đạt đủ bằng chứng trên năm câu có đáp án,
từ chối đúng hai câu thiếu bằng chứng của bộ test mới.

Kết quả không chứng minh khả năng tổng quát:
- Corpus chỉ giữ văn bản chính; không xử lý đầy đủ bảng/ảnh/biểu đồ.
- Test nhỏ, phần lớn câu đơn đoạn và có chung corpus với dev.
- Verifier có thể sai và còn gây lỗi quotation.
- Citation đúng ID chưa bảo đảm hỗ trợ đầy đủ phát biểu.
- Context ưu tiên kết quả cũ; nhiều lượt tìm chưa được thể hiện
  trong các lần chạy cuối.
- Latency chỉ đo một lượt chạy; điều kiện cache khác nhau.
