import re
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

d1 = pd.read_excel("fmd_report_2025-04-28_12.45.xlsx", header=1).drop(columns=["Unnamed: 0","Tags"], errors="ignore")
d2 = pd.read_excel("fmd_report_2025-05-05_13.25.xlsx",  header=1).drop(columns=["Unnamed: 0","Tags"], errors="ignore")
d3 = pd.read_excel("fmd_report_2025-05-05_13.30.xlsx",  header=1).drop(columns=["Unnamed: 0","Tags"], errors="ignore")
data = pd.concat([d1, d2, d3], ignore_index=True)

USE_LLM = True
LLM_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_LLM_PRED = 8000

print(data["ID"].duplicated().sum())

data = data.dropna(subset=["Sentiment"]).copy()
def text_label(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return str(x)

data["Sentiment"] = data["Sentiment"].apply(text_label)

def make_text(df):
    if ("Title" in df.columns) and ("Content" in df.columns):
        return (df["Title"].fillna("").astype(str) + " " +
                df["Content"].fillna("").astype(str)).str.strip()
    elif "Content" in df.columns:
        return df["Content"].fillna("").astype(str)
    elif "Title" in df.columns:
        return df["Title"].fillna("").astype(str)
    else:
        return pd.Series([""] * len(df))

data["text"] = make_text(data)

def label_country(domain):
    d = str(domain).lower()
    tld = d.split(".")[-1] if d else ""
    if tld == "de": return "Germany"
    if tld == "hu": return "Hungary"
    if tld == "sk": return "Slovakia"
    return "Other"

if "Domain" in data.columns:
    data["CountryGuess"] = data["Domain"].apply(label_country)
else:
    data["CountryGuess"] = "Other"

seed_phrases = [
    "lab leak","made in a lab","bioweapon","cover up","false flag",
    "aus dem labor","laborursprung","biowaffe","vertuschung",
    "labor szivárgás","labor eredet","biológiai fegyver","eltussolás",
    "únik z laboratória","laboratórny pôvod","biologická zbraň","utajovanie",
]

keyword_list = []
for p in seed_phrases:
    keyword_list.append(p)
    keyword_list.append(p.replace(" ", "-"))

pattern = r"\b(" + "|".join(map(re.escape, keyword_list)) + r")\b"
misinfo_rx = re.compile(pattern, flags=re.IGNORECASE)

def has_misinfo(text):
    return bool(misinfo_rx.search("" if text is None else str(text)))

data["misinfo_candidate"] = data["text"].apply(has_misinfo)

data["Pred_LLM"] = np.nan

if USE_LLM:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

        tok = AutoTokenizer.from_pretrained(LLM_MODEL, cache_dir=r"C:\hf_cache")
        mdl = AutoModelForSequenceClassification.from_pretrained(LLM_MODEL, cache_dir=r"C:\hf_cache")
        pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=False, truncation=True)

        idx = data.sample(min(MAX_LLM_PRED, len(data)), random_state=42).index
        texts = data.loc[idx, "text"].fillna("").tolist()

        preds = pipe(texts, batch_size=32)

        def map_label(lbl):
            n = "".join(ch for ch in (lbl or "") if ch.isdigit())
            n = int(n) if n else 3
            return "-1.0" if n <= 2 else ("0.0" if n == 3 else "1.0")


        mapped = [map_label(p["label"]) for p in preds]
        data.loc[idx, "Pred_LLM"] = mapped

        mask = data["Pred_LLM"].notna()
        if mask.any():
            y_true = data.loc[mask, "Sentiment"]
            y_pred = data.loc[mask, "Pred_LLM"]
            print("\n Classification report")
            print(classification_report(y_true, y_pred, zero_division=0))
            print("\nConfusion matrix (order):", sorted(y_true.unique()))
            print(confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique())))
    except Exception as e:
        USE_LLM = False
        print("\n[INFO] transformers/LLM cannot be used, because:", e)

country_sent = (data.groupby("CountryGuess")["Sentiment"].value_counts(normalize=True).rename("share").reset_index())
country_sent.to_csv("summary_country_sentiment.csv", index=False)

misinfo_overall = pd.DataFrame({"metric":["misinfo_share_overall"],"value":[data["misinfo_candidate"].mean()]})
misinfo_by_country = (data.groupby("CountryGuess")["misinfo_candidate"].mean().rename("misinfo_share").reset_index())

misinfo_overall.to_csv("summary_misinfo_overall.csv", index=False)
misinfo_by_country.to_csv("summary_misinfo_by_country.csv", index=False)

misinfo_sent = (data.groupby("misinfo_candidate")["Sentiment"].value_counts(normalize=True).rename("share").reset_index())
misinfo_sent.to_csv("summary_misinfo_vs_sentiment.csv", index=False)

print("\nSUMMARY")
print("Top countries (guess):")
print(data["CountryGuess"].value_counts().head(10))
print("\nOverall misinfo share:", round(float(data["misinfo_candidate"].mean()), 4))
print("\nCountry misinfo shares:")
print(misinfo_by_country.sort_values("misinfo_share", ascending=False))
