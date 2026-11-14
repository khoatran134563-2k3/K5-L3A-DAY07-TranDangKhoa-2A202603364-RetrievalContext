# Chạy bài nộp Day 7 — Trần Đăng Khoa — 2A202603364

## Đọc kết quả đã lưu — không cần API hoặc database

- Báo cáo: report/individual/2A202603364_TranDangKhoa.md.
- Quyết định chọn trên dev: artifacts/final_selection.json.
- Test và hash: benchmark/test_questions.json, benchmark/test_manifest.json.
- Kết quả test: starter/results/final_test/.
- Mỗi thư mục kết quả có traces.json, reviews.json,
  evaluated_summary.json và review_packet.md.
- Các lần chạy lỗi được giữ, không thay bằng kết quả giả.

Các kết quả dev cuối:
- agent_dev_dense_baseline_20261114T100948307118Z
- agent_dev_tfidf_20261114T101205688015Z
- agent_dev_hybrid_rrf_20261114T102003609244Z
- agent_dev_dense_more_context_20261114T102334968926Z

Các thư mục trên nằm trong starter/results/.

## Môi trường

Đã chạy trên Windows, Python 3.13.14.
Code yêu cầu Python 3.11 trở lên và Docker nếu cần database mới.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-day7.txt
Copy-Item .env.example .env
```

Điền key của người chạy vào .env, không commit file này.
DATABASE_URL mặc định dùng PostgreSQL tại 127.0.0.1:55433.

## Database mới — chỉ thực hiện khi cần tái tạo

Không cần làm nếu database Day 6 hiện tại đã chứa đúng 132 chunk B.
Nếu cổng 55433 đang được sử dụng, không khởi động thêm database trùng cổng;
dùng database tương ứng hoặc điều chỉnh mapping và DATABASE_URL.

```powershell
docker compose -p day7_repro -f runtime/docker-compose.yml up -d --wait
.\.venv\Scripts\python.exe -X utf8 restore_vectors.py --replace-b
```

Lệnh restore thay dữ liệu strategy B trong bảng day6_chunks thuộc
database được DATABASE_URL chỉ định; không thay strategy A.
runtime/B.jsonl chứa vector đã có từ Day 6, không gọi embedding lại.

## Launcher và adapter đi kèm

Các file starter được giữ nguyên phiên bản đã freeze để đối chiếu hash.
Chúng còn tham chiếu đường dẫn Day 6 của máy thực nghiệm.

run_portable.py nạp adapter Day 6 đi kèm tại runtime/src trước khi
thực thi starter. Launcher đọc .env ở gốc repo, nên đường chạy này
không cần thư mục D:\VIN Lab của tác giả.

Launcher và công cụ restore được thêm khi đóng gói sau đánh giá;
không sửa implementation đã dùng để chọn cấu hình và chạy test.

## Kiểm tra code — không gọi API

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 run_portable.py rag_agent.py --help
```

15 unit test của starter kiểm tra các contract và logic mẫu;
không thay thế benchmark thật hoặc bao phủ toàn bộ code bổ sung.

## Chạy một câu mới — có phí API

```powershell
.\.venv\Scripts\python.exe -X utf8 run_portable.py rag_agent.py --method hybrid --question "What institutional condition is included in the report's analog complements?" --top-k 3 --budget 2600 --max-searches 3
```

Model: gpt-4.1-mini-2025-04-14.
Query embedding: text-embedding-3-small, 1536 chiều.
Cần tài khoản có quyền model và quota.
Nếu model không còn khả dụng trong tương lai, phải ghi rõ model thay thế
và coi kết quả tái chạy là một thí nghiệm khác.

## Tái chạy thí nghiệm

Có thể chạy dev qua launcher, ví dụ:

```powershell
.\.venv\Scripts\python.exe -X utf8 run_portable.py run_agent_dev.py --config hybrid_rrf
```

Có phí API và tạo thư mục kết quả mới.

Runner test:
```powershell
.\.venv\Scripts\python.exe -X utf8 run_portable.py run_final_test.py
```

Runner kiểm tra hash và từ chối nếu starter/results/final_test đã tồn tại.
Bài nộp đã chứa lần test gốc; không ghi đè.
Nếu người chấm muốn tái chạy, dùng một bản sao repo riêng và giữ kết quả
gốc ngoài vị trí đầu ra trước khi chạy. Không chọn lần chạy tốt nhất hoặc
chỉnh hệ thống bằng test.

Các helper chuẩn bị dữ liệu từ Day 6/PDF giữ đường dẫn môi trường tác giả.
Không cần chạy các helper này để đọc kết quả hoặc chạy agent qua launcher.

## Nguồn và quyền sử dụng

World Bank. 2016. World Development Report 2016:
Digital Dividends — Overview. CC BY 3.0 IGO.

Nguồn, phạm vi trích xuất và SHA-256:
runtime/source_manifest.json.

PDF gốc:
https://documents1.worldbank.org/curated/en/961621467994698644/pdf/102724-WDR-WDR2016Overview-ENGLISH-WebResBox-394840B-OUO-9.pdf

Chỉ đưa văn bản dẫn xuất và vector cần tái lập; không đưa PDF gốc.
Thành phần bên thứ ba có thể có quyền riêng.