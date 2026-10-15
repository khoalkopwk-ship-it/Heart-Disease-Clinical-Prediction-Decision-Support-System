import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

MODEL_PATH = "heart_dt_model.pkl"
DATA_PATH = "heart.csv"
FEATURE_NAME_VI = {
    "age": "Tuổi",
    "sex_Male": "Giới tính: Nam",
    "trestbps": "Huyết áp lúc nghỉ (mmHg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Đường huyết lúc đói > 120 mg/dl",
    "restecg_st-t abnormality": "Điện tâm đồ: bất thường ST-T",
    "restecg_lv hypertrophy": "Điện tâm đồ: phì đại thất trái",
    "thalch": "Nhịp tim tối đa đạt được",
    "exang": "Đau ngực khi gắng sức",
    "oldpeak": "Độ chênh ST so với lúc nghỉ",
    "slope_flat": "Độ dốc ST: phẳng",
    "slope_downsloping": "Độ dốc ST: giảm dần",
    "slope_upsloping": "Độ dốc ST: tăng dần",
    "ca": "Số mạch vành lớn bị hẹp",
    "thal_normal": "Thallium: bình thường",
    "thal_fixed defect": "Thallium: tổn thương cố định",
    "thal_reversable defect": "Thallium: tổn thương hồi phục",
    "cp_atypical angina": "Đau ngực: không điển hình",
    "cp_non-anginal": "Đau ngực: không do tim",
    "cp_asymptomatic": "Đau ngực: không triệu chứng",
}

# đọc mô hình đã huấn luyện từ file .pkl
def load_model(model_path=MODEL_PATH):
    package = joblib.load(model_path)
    model = package["model"]
    feature_names_raw = package["feature_names_raw"]
    feature_names_encoded = package["feature_names_encoded"]
    return model, feature_names_raw, feature_names_encoded

# đọc dữ liệu gốc từ file CSV
def load_raw_data(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    return df

# nhập dữ liệu 1 bệnh nhân
def input_patient_data(feature_names_raw):
    data = {}
    print("\n=== NHẬP THÔNG TIN BỆNH NHÂN ===")

    for feat in feature_names_raw:
        # Tuổi
        if feat == "age":
            while True:
                try:
                    age = float(input("Tuổi: "))
                    data[feat] = age
                    break
                except ValueError:
                    print("  -> Vui lòng nhập số.")

        # Giới tính
        elif feat == "sex":
            print("Giới tính:")
            print("  1. Nam")
            print("  2. Nữ")
            while True:
                choice = input("Chọn: ").strip()
                if choice == "1":
                    data[feat] = "Male"
                    break
                elif choice == "2":
                    data[feat] = "Female"
                    break
                else:
                    print("  -> Vui lòng chọn 1 hoặc 2.")

        # Đau ngực
        elif feat == "cp":
            print("Loại đau ngực:")
            print("  1. typical angina (đau thắt ngực điển hình)")
            print("  2. atypical angina (đau thắt ngực không điển hình)")
            print("  3. non-anginal (đau không do tim)")
            print("  4. asymptomatic (không triệu chứng)")
            mapping = {
                "1": "typical angina",
                "2": "atypical angina",
                "3": "non-anginal",
                "4": "asymptomatic",
            }
            while True:
                choice = input("Lựa chọn: ").strip()
                if choice in mapping:
                    data[feat] = mapping[choice]
                    break
                else:
                    print("  -> Vui lòng chọn 1-4.")

        # Huyết áp
        elif feat == "trestbps":
            while True:
                try:
                    x = float(input("Huyết áp lúc nghỉ (mmHg): "))
                    data[feat] = x
                    break
                except ValueError:
                    print("  -> Vui lòng nhập số.")

        # Cholesterol
        elif feat == "chol":
            while True:
                try:
                    x = float(input("Cholesterol (mg/dl): "))
                    data[feat] = x
                    break
                except ValueError:
                    print("  -> Vui lòng nhập số.")

        # Đường huyết đói
        elif feat == "fbs":
            print("Đường huyết lúc đói > 120 mg/dl:")
            print("  1. Có")
            print("  2. Không")
            while True:
                choice = input("Chọn: ").strip()
                if choice == "1":
                    data[feat] = True
                    break
                elif choice == "2":
                    data[feat] = False
                    break
                else:
                    print("  -> Vui lòng chọn 1-2.")

        # Điện tâm đồ lúc nghỉ
        elif feat == "restecg":
            print("Kết quả điện tâm đồ lúc nghỉ:")
            print("  1. normal (bình thường)")
            print("  2. st-t abnormality (bất thường ST-T)")
            print("  3. lv hypertrophy (phì đại thất trái)")
            mapping = {
                "1": "normal",
                "2": "st-t abnormality",
                "3": "lv hypertrophy",
            }
            while True:
                choice = input("Chọn: ").strip()
                if choice in mapping:
                    data[feat] = mapping[choice]
                    break
                else:
                    print("  -> Vui lòng chọn 1-3")

        # Nhịp tim tối đa đạt được
        elif feat == "thalch":
            while True:
                try:
                    x = float(input("Nhịp tim tối đa đạt được: "))
                    data[feat] = x
                    break
                except ValueError:
                    print("  -> Vui lòng nhập số.")

        # Đau ngực khi gắng sức (exang)
        elif feat == "exang":
            print("Đau ngực khi gắng sức:")
            print("  1. Có")
            print("  2. Không")
            while True:
                choice = input("Chọn: ").strip()
                if choice == "1":
                    data[feat] = True
                    break
                elif choice == "2":
                    data[feat] = False
                    break
                else:
                    print("  -> Vui lòng chọn 1-2")

        # oldpeak
        elif feat == "oldpeak":
            while True:
                try:
                    x = float(input("Độ chênh ST so với lúc nghỉ: "))
                    data[feat] = x
                    break
                except ValueError:
                    print("  -> Vui lòng nhập số.")

        # slope
        elif feat == "slope":
            print("Độ dốc đoạn ST khi gắng sức:")
            print("  1. upsloping (tăng dần)")
            print("  2. flat (phẳng)")
            print("  3. downsloping (giảm dần)")
            mapping = {
                "1": "upsloping",
                "2": "flat",
                "3": "downsloping",
            }
            while True:
                choice = input("Chọn: ").strip()
                if choice in mapping:
                    data[feat] = mapping[choice]
                    break
                else:
                    print("  -> Vui lòng chọn 1-3.")

        # ca
        elif feat == "ca":
            while True:
                try:
                    x = float(input("Số mạch vành lớn bị hẹp: "))
                    data[feat] = x
                    break
                except ValueError:
                    print("  -> Vui lòng nhập số.")

        # thal
        elif feat == "thal":
            print("Thallium (tưới máu cơ tim):")
            print("  1. normal (bình thường)")
            print("  2. fixed defect (tổn thương cố định)")
            print("  3. reversable defect (tổn thương hồi phục)")
            mapping = {
                "1": "normal",
                "2": "fixed defect",
                "3": "reversable defect",
            }
            while True:
                choice = input("Chọn: ").strip()
                if choice in mapping:
                    data[feat] = mapping[choice]
                    break
                else:
                    print("  -> Vui lòng chọn 1-3.")
        else:
            value = input(f"Nhập giá trị cho {feat}: ")
            data[feat] = value

    df_row = pd.DataFrame([data])
    return df_row

# Hàm mã hóa 1 dòng dữ liệu (one-hot)
def encode_input_row(row_raw, feature_names_encoded):
    row_encoded = pd.get_dummies(row_raw, drop_first=True)
    row_encoded = row_encoded.reindex(columns=feature_names_encoded, fill_value=0)
    return row_encoded

# diễn giải nhãn dự đoán
def interpret_prediction(pred):
    if pred == 1:
        return "Có nguy cơ bệnh tim"
    else:
        return "Không có nguy cơ bệnh tim"

# vẽ biểu đồ cột phân bố bệnh tim
def plot_disease_distribution(df_raw):
    df = df_raw.copy()
    if "num" not in df.columns:
        print("Không có cột num -> không vẽ được phân bố bệnh.")
        return

    df["target"] = (df["num"] > 0).astype(int)
    counts = df["target"].value_counts().sort_index()

    labels = ["Không bệnh", "Có bệnh"]

    plt.figure()
    counts.plot(kind="bar")
    plt.xticks([0, 1], labels, rotation=0)
    plt.ylabel("Số lượng bệnh nhân")
    plt.title("Phân bố bệnh tim trong tập dữ liệu")
    plt.tight_layout()
    plt.show()

# vẽ các đặc trưng quan trọng của cây quyết định
def plot_feature_importance(model, feature_names_encoded, top_k=10):
    if not hasattr(model, "feature_importances_"):
        print("Mô hình không có thuộc tính feature_importances_.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_k = min(top_k, len(feature_names_encoded))
    top_idx = indices[:top_k]

    names = [feature_names_encoded[i] for i in top_idx]
    values = importances[top_idx]

    vi_labels = []
    for name in names:
        vi_labels.append(FEATURE_NAME_VI.get(name, name))

    plt.figure()
    plt.barh(range(top_k), values[::-1])
    plt.yticks(range(top_k), vi_labels[::-1])
    plt.xlabel("Mức độ quan trọng")
    plt.title("Những đặc trưng quan trọng của mô hình")
    plt.tight_layout()
    plt.show()

def plot_decision_tree_model(model, feature_names_encoded, max_depth_show=3):
    # Đổi tên feature sang tiếng Việt nếu có trong FEATURE_NAME_VI
    vi_feature_names = [FEATURE_NAME_VI.get(f, f) for f in feature_names_encoded]

    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=vi_feature_names,
        class_names=["Không bệnh", "Có bệnh"],
        filled=True,
        rounded=True,
        max_depth=max_depth_show
    )
    plt.title("Minh họa cây quyết định (Decision Tree)")
    plt.tight_layout()
    plt.show()

# Hàm thống kê số ca mắc bệnh theo nhóm tuổi
def plot_age_hist_positive(df_raw):
    df = df_raw.copy()
    if "age" not in df.columns or "num" not in df.columns:
        print("Không có cột age hoặc num trong dữ liệu, không vẽ được.")
        return

    df["target"] = (df["num"] > 0).astype(int)
    df_pos = df[df["target"] == 1].copy()

    if df_pos.empty:
        print("Trong dữ liệu không có bệnh nhân nào bị bệnh.")
        return

    df_pos["age_group_start"] = (df_pos["age"] // 5) * 5

    group_counts = df_pos["age_group_start"].value_counts().sort_index()

    labels = []
    values = []
    for start_age, count in group_counts.items():
        start_age = int(start_age)
        labels.append(f"{start_age}-{start_age + 4}")
        values.append(count)

    # Vẽ biểu đồ cột
    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.xlabel("Nhóm tuổi")
    plt.ylabel("Số ca mắc bệnh tim")
    plt.title("Thống kê số ca mắc bệnh tim theo nhóm tuổi")
    plt.tight_layout()
    plt.show()

# Hàm main: menu chính của chương trình
def main():
    print("=========DỰ ĐOÁN BỆNH BẰNG CÂY QUYẾT ĐỊNH============")

    try:
        model, feature_names_raw, feature_names_encoded = load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"Không tìm thấy file mô hình: {MODEL_PATH}")
        print("Hãy chạy DecisionTree_Heart_Disease_Prediction_App.py trước.")
        return

    df_raw = None

    while True:
        print("\n---------MENU---------")
        print("1. Nhập thông tin 1 bệnh nhân để dự đoán")
        print("2. Vẽ biểu đồ phân bố bệnh tim từ file .csv")
        print("3. Vẽ biểu đồ feature importance của Decision Tree")
        print("4. Thống kê số ca mắc bệnh tim theo nhóm tuổi")
        print("5. Minh họa cây quyết định (plot_tree)")
        print("0. Thoát")
        choice = input("Chọn chức năng: ").strip()

        if choice == "1":
            row_raw = input_patient_data(feature_names_raw)
            row_encoded = encode_input_row(row_raw, feature_names_encoded)

            pred = model.predict(row_encoded)[0]
            if hasattr(model, "predict_proba"):
                proba_full = model.predict_proba(row_encoded)[0]
                confidence = proba_full[pred]
            else:
                confidence = None

            print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
            explain = interpret_prediction(pred)
            print("Kết quả dự đoán:", explain)
            if confidence is not None:
                print("Độ tin cậy: {:.2f}%".format(confidence * 100.0))

        elif choice == "2":
            if df_raw is None:
                try:
                    df_raw = load_raw_data(DATA_PATH)
                except FileNotFoundError:
                    print(f"Không tìm thấy file dữ liệu: {DATA_PATH}")
                    continue
            plot_disease_distribution(df_raw)

        elif choice == "3":
            plot_feature_importance(model, feature_names_encoded, top_k=10)

        elif choice == "4":
            if df_raw is None:
                try:
                    df_raw = load_raw_data(DATA_PATH)
                except FileNotFoundError:
                    print(f"Không tìm thấy file dữ liệu: {DATA_PATH}")
                    continue
            plot_age_hist_positive(df_raw)

        elif choice == "5":
            plot_decision_tree_model(model, feature_names_encoded, max_depth_show=3)

        elif choice == "0":
            print("Dừng chương trình")
            break

        else:
            print("Lỗi, chọn lại")

if __name__ == "__main__":
    main()
