import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "Импорт": RAW / "import_tovarov_vazh-2021-1.xlsx",
    "Экспорт": RAW / "export_tovarov_vazh-2021-1.xlsx",
}

def to_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).replace("\xa0", " ").strip()
    if s in {"", "-", "—", "–", "…", "...", "nan"}:
        return np.nan
    s = re.sub(r"\s+", "", s).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan

def split_name_unit(raw):
    name = re.sub(r"\s+", " ", str(raw)).strip().rstrip(",")
    if "," in name:
        head, tail = name.rsplit(",", 1)
        if re.search(r"(тыс|тонн|штук|млн|долл|м3|шт)", tail, re.I):
            return head.strip(), tail.strip()
    return name, "ед."

CATEGORY = {
    # импорт
    "Растительные масла": "Продовольствие и сырьё для АПК",
    "Отходы, полученные при извлечении соевого масла": "Продовольствие и сырьё для АПК",
    "Лекарственные средства, расфасованные для розничной продажи": "Фармацевтика",
    "Каучук синтетический": "Химия и полимеры",
    "Черные металлы": "Металлы и металлоизделия",
    "Трубы из черных металлов": "Металлы и металлоизделия",
    "Металлоконструкции из черных металлов": "Металлы и металлоизделия",
    "Двигатели внутреннего сгорания": "Машины и оборудование",
    "Вычислительные машины для автоматической обработки информации": "Электроника и ИКТ",
    "Машины и механические устройства специального назначения": "Машины и оборудование",
    "Аппаратура связи и части к ней": "Электроника и ИКТ",
    "Легковые автомобили": "Транспорт",
    "Части и принадлежности для автомобилей и тракторов": "Транспорт",
    # экспорт
    "Тара пластмассовая": "Химия и полимеры",
    "Лесоматериалы продольно-распиленные": "Деревообработка и мебель",
    "Плиты древесно-стружечные": "Деревообработка и мебель",
    "Мебель, включая медицинскую": "Деревообработка и мебель",
    "Химические волокна и нити": "Химия и полимеры",
    "Холодильники, морозильники и холодильное оборудование": "Машины и оборудование",
    "Провода изолированные, кабели": "Машины и оборудование",
    "Сельскохозяйственная техника": "Машины и оборудование",
    "Говядина": "Продовольствие и сырьё для АПК",
    "Мясо и пищевые субпродукты домашней птицы": "Продовольствие и сырьё для АПК",
    "Молоко и сливки несгущенные": "Продовольствие и сырьё для АПК",
    "Молоко и сливки сгущенные и сухие": "Продовольствие и сырьё для АПК",
    "Масло сливочное": "Продовольствие и сырьё для АПК",
    "Сыры и творог": "Продовольствие и сырьё для АПК",
    "Сахар": "Продовольствие и сырьё для АПК",
}

def load_one(flow, path):
    raw = pd.read_excel(path, header=None)
    hdr = None
    for i in range(min(10, len(raw))):
        years = [to_num(v) for v in raw.iloc[i]]
        if sum(1 for y in years if y and 1995 <= y <= 2030) >= 10:
            hdr = i; break
    if hdr is None:
        raise ValueError(f"Не найдена строка с годами в {path}")

    years = {j: int(to_num(v)) for j, v in enumerate(raw.iloc[hdr])
             if to_num(v) and 1995 <= to_num(v) <= 2030}

    rows = []
    for i in range(hdr + 1, len(raw)):
        label = raw.iat[i, 0]
        if pd.isna(label) or not str(label).strip():
            continue
        product, unit = split_name_unit(label)
        for j, year in years.items():
            val = to_num(raw.iat[i, j])
            if np.isnan(val):
                continue
            rows.append({
                "Поток": flow,
                "Категория": CATEGORY.get(product, "Прочее"),
                "Товар": product,
                "Единица": unit,
                "Год": year,
                "Значение": val,
            })
    return pd.DataFrame(rows)

df = pd.concat([load_one(f, p) for f, p in FILES.items()], ignore_index=True)
df = df.sort_values(["Поток", "Товар", "Год"]).reset_index(drop=True)

g = df.groupby(["Поток", "Товар"], sort=False)
df["Значение_пред"] = g["Значение"].shift(1)
df["Прирост_гг_%"] = ((df["Значение"] / df["Значение_пред"] - 1) * 100).round(1)
base = (df[df["Год"] == 2000].set_index(["Поток", "Товар"])["Значение"])
df["База2000"] = df.set_index(["Поток", "Товар"]).index.map(base)
df["Индекс_2000"] = (df["Значение"] / df["База2000"] * 100).round(1)
df["Товар_полн"] = df["Поток"] + ": " + df["Товар"]

df.drop(columns=["База2000"]).to_csv(OUT / "trade_long.csv", index=False, encoding="utf-8-sig")
df.drop(columns=["База2000"]).to_excel(OUT / "trade_long.xlsx", index=False)

summary = (df.pivot_table(index=["Поток", "Категория", "Товар", "Единица"],
                          columns="Год", values="Значение")
             .assign(Рост_2021_к_2000=lambda t: (t[2021] / t[2000]).round(2))
             .sort_values("Рост_2021_к_2000", ascending=False))
summary.to_csv(OUT / "summary_growth.csv", encoding="utf-8-sig")
print(df.shape, "строк готово")