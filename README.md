
# Akbank Hackathon: DisasterTech - Our Contribution

--- 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62bdd8065f304e8ea762287f/raHCZDUuHPckwwrKDRz-A.png)

--- 

## 🎯 Introduction

**Akbank LAB** and **imece** teamed up to launch the **Akbank Hackathon: DisasterTech**, a beacon for innovators passionate about harnessing technology to revolutionize disaster management and relief. The event, which began on the 14th of October online and culminated at the Sabancı Center on the 22nd, saw a plethora of teams brainstorming and developing visionary solutions aimed at disaster alerts, preparedness, and post-calamity assistance.

In response to this call-to-action, our team stepped up, and this repository stands testament to the innovation we brought to the table during this monumental event.

For an in-depth look at the hackathon, feel free to visit [Akbank Hackathon: DisasterTech](https://www.akbanklab.com/tr/akbank-hackathon-disastertech#section-4).

---

### 🌪️ **Disaster Management Classification Overview** 🚨


📊 Our model, boasting a commendable accuracy of **89.09%**, is adept at swiftly classifying textual data into pivotal categories, proving invaluable during crisis management and relief efforts. 

- 🏠 **Shelter Needs (Barınma İhtiyacı)**
  
- 🔌 **Electricity Source (Elektrik Kaynağı)**
  
- 💧 **Water Needs (Su İhtiyacı)**
  
- 🍲 **Food Needs (Yemek İhtiyacı)**

- 🚧 **Debris Removal Alerts (Enkaz Kaldırma İhbarı)**
  
- 🚑 **Emergency Health Assistance Requests (Acil Sağlık Yardımı Talebi)**

Our vigilant model doesn't stop there:

- ❌ It discerns non-relevant alerts, categorizing them as **Unrelated Reports (Alakasız İhbar)**.
  
- ⚠️ It stays alert to potential threats, recognizing **Looting Incident Reports (Yağma Olay Bildirimi)**.

Whether it's about ensuring 🚚 logistical support, 👕 clothing provisions, or 🔥 heating essentials, our model stands as a holistic solution for discerning and categorizing diverse requirements amidst disaster scenarios. 

---


## 📊 Model Performance & Usage

In this document, you can find detailed insights regarding our classification model's performance.

- 🤗 [View Model on Hugging Face](https://huggingface.co/tarikkaankoc7/zeltech-akbank-hackathon)

#### 🎯 Overall Accuracy

- **Accuracy Metric**: 📈 89.09%

## 📝 Classification Report

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Alakasız İhbar           | 0.90      | 0.92   | 0.91     | 327     |
| Barınma İhtiyacı            | 0.90      | 0.90   | 0.90     | 124     |
| Elektrik Kaynağı   | 0.82      | 0.93   | 0.87     | 58      |
| Enkaz Kaldırma İhbarı     | 0.88      | 0.85   | 0.86     | 202     |
| Giysi İhtiyacı              | 0.88      | 0.80   | 0.84     | 45      |
| Isınma İhtiyacı             | 0.94      | 0.90   | 0.92     | 171     |
| Lojistik Destek Talebi           | 0.90      | 0.86   | 0.88     | 63      |
| Acil Sağlık Yardımı Talebi             | 0.88      | 0.82   | 0.85     | 34      |
| Su İhtiyacı                 | 0.86      | 0.91   | 0.89     | 220     |
| Yağma Olay Bildirimi              | 1.00      | 1.00   | 1.00     | 15      |
| Yemek İhtiyacı              | 0.90      | 0.88   | 0.89     | 226     |
| **Total/Avg**      | **0.89**  | **0.89**| **0.89** | **1485**|


## 🖥️ How to use the model

Here is a Python example demonstrating how to use the model for predicting class of a given text:

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

model_name = "tarikkaankoc7/zeltech-akbank-hackathon"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probs, dim=-1).item()
    predicted_class_name = model.config.id2label[predicted_class_id]

    return predicted_class_name

text = "Hatay/Antakya odabaşı atatürk bulvarı ahmet gürses apartmanı arkadasım ilayda kürkçü enkaz altında paylaşır mısınız"
predicted_class_name = predict(text)
print(f"Predicted Class: {predicted_class_name}")
```

#### Expected Output:

```bash
Predicted Class: Enkaz Kaldırma İhbarı
```

## 🖋️ Authors

- **Şeyma SARIGIL** - [📧 Email](mailto:seymasargil@gmail.com)
- **Tarık Kaan KOÇ** - [📧 Email](mailto:tarikkaan1koc@gmail.com)
- **Alaaddin Erdinç DAL** - [📧 Email](mailto:aerdincdal@icloud.com)
- **Anıl YAĞIZ** - [📧 Email](mailto:anill.yagiz@gmail.com)
