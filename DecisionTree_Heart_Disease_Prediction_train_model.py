import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

DUONG_DAN_DU_LIEU = "heart.csv"
DUONG_DAN_MODEL = "heart_dt_model.pkl"

def tai_du_lieu_thuan(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")
    du_lieu = pd.read_csv(csv_path)
    return du_lieu

def tien_xu_ly_cho_mo_hinh(du_lieu_goc: pd.DataFrame):
    du_lieu = du_lieu_goc.copy()

    cac_cot_bo_di = [ten for ten in ["id", "dataset"] if ten in du_lieu.columns]
    du_lieu.drop(columns=cac_cot_bo_di, inplace=True)

    if "num" not in du_lieu.columns:
        raise ValueError("Ko tìm thấy cột 'num' để tạo nhãn")
    du_lieu["target"] = (du_lieu["num"] > 0).astype(int)

    cac_cot_so = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    for cot in cac_cot_so:
        if cot in du_lieu.columns:
            du_lieu[cot] = du_lieu[cot].fillna(du_lieu[cot].median())

    cac_cot_phan_loai = ["sex", "cp", "restecg", "slope", "thal"]
    for cot in cac_cot_phan_loai:
        if cot in du_lieu.columns and du_lieu[cot].isna().any():
            du_lieu[cot] = du_lieu[cot].fillna(du_lieu[cot].mode()[0])

    cac_cot_nhi_phan = ["fbs", "exang"]
    for cot in cac_cot_nhi_phan:
        if cot in du_lieu.columns:
            du_lieu[cot] = pd.to_numeric(du_lieu[cot], errors="coerce")
            du_lieu[cot] = du_lieu[cot].fillna(0)
            du_lieu[cot] = du_lieu[cot].astype(int)

    nhan_muc_tieu = du_lieu["target"]
    du_lieu_thuoc_tinh = du_lieu.drop(columns=["target", "num"])

    danh_sach_cot_goc = du_lieu_thuoc_tinh.columns.tolist()

    du_lieu_da_ma_hoa = pd.get_dummies(du_lieu_thuoc_tinh, drop_first=True)

    return du_lieu_da_ma_hoa, nhan_muc_tieu, danh_sach_cot_goc

# in báo cáo phân loại
def in_bao_cao_phan_loai_tieng_viet(y_thuc, y_du_doan):
    report = classification_report(y_thuc, y_du_doan, output_dict=True)

    tong_mau = int(report["macro avg"]["support"])
    line = "+" + "-"*22 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*10 + "+"
    row_fmt = "| {:<20} | {:>12} | {:>12} | {:>12} | {:>8} |"

    print("Báo cáo phân loại:")
    print(line)
    print(row_fmt.format("Lớp / Thống kê", "Độ chính xác", "Độ bao phủ", "F1-score", "Số mẫu"))
    print(line)

    ten_lop = {
        "0": "0 - Không bệnh",
        "1": "1 - Có bệnh"
    }
    for key in ["0", "1"]:
        m = report[key]
        print(
            row_fmt.format(
                ten_lop.get(key, key), f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1-score']:.2f}", int(m['support']))
        )

    acc = report["accuracy"]
    print(line)
    print(
        row_fmt.format(
            "Độ chính xác (acc.)", "", "", f"{acc:.2f}", tong_mau)
    )

    # Trung bình macro
    macro = report["macro avg"]
    print(
        row_fmt.format(
            "Trung bình macro", f"{macro['precision']:.2f}", f"{macro['recall']:.2f}", f"{macro['f1-score']:.2f}", int(macro["support"])
        )
    )

    # Trung bình trọng số
    weighted = report["weighted avg"]
    print(
        row_fmt.format(
            "Trung bình trọng số",
            f"{weighted['precision']:.2f}",
            f"{weighted['recall']:.2f}",
            f"{weighted['f1-score']:.2f}",
            int(weighted["support"])
        )
    )

    print(line)

def huan_luyen_cay_quyet_dinh(du_lieu_da_ma_hoa, nhan_muc_tieu, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(du_lieu_da_ma_hoa, nhan_muc_tieu, test_size=test_size, random_state=random_state, stratify=nhan_muc_tieu)

    mo_hinh = DecisionTreeClassifier(max_depth=4, random_state=42)
    mo_hinh.fit(X_train, y_train)

    y_du_doan = mo_hinh.predict(X_test)

    do_chinh_xac = accuracy_score(y_test, y_du_doan)
    print("======ĐÁNH GIÁ MÔ HÌNH CÂY QUYẾT ĐỊNH (DECISION TREE)======")
    print(f"Độ chính xác: {do_chinh_xac:.4f}")
    print()

    in_bao_cao_phan_loai_tieng_viet(y_test, y_du_doan)
    print()
    print("Ma trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_du_doan))

    return mo_hinh, du_lieu_da_ma_hoa.columns.tolist()

def luu_mo_hinh(mo_hinh, danh_sach_cot_goc, danh_sach_cot_ma_hoa, duong_dan_model: str = DUONG_DAN_MODEL):
    goi_luu = {
        "model": mo_hinh,
        "feature_names_raw": danh_sach_cot_goc,
        "feature_names_encoded": danh_sach_cot_ma_hoa
    }
    joblib.dump(goi_luu, duong_dan_model)
    print()
    print(f"Đã lưu mô hình vào file: {duong_dan_model}")

def main():
    print("=========HUẤN LUYỆN MÔ HÌNH==========")

    du_lieu_goc = tai_du_lieu_thuan(DUONG_DAN_DU_LIEU)
    print(f"Đã đọc dữ liệu: {du_lieu_goc.shape[0]} dòng, {du_lieu_goc.shape[1]} cột")

    du_lieu_da_ma_hoa, nhan_muc_tieu, danh_sach_cot_goc = tien_xu_ly_cho_mo_hinh(du_lieu_goc)
    print(f"Sau tiền xử lý: {du_lieu_da_ma_hoa.shape[0]} mẫu, "
          f"{du_lieu_da_ma_hoa.shape[1]} thuộc tính")

    mo_hinh, danh_sach_cot_ma_hoa = huan_luyen_cay_quyet_dinh(
        du_lieu_da_ma_hoa, nhan_muc_tieu
    )

    luu_mo_hinh(mo_hinh, danh_sach_cot_goc, danh_sach_cot_ma_hoa, DUONG_DAN_MODEL)
    print()
    print("DONE")

if __name__ == "__main__":
    main()