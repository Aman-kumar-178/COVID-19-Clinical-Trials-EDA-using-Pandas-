# 🦠 COVID-19 Clinical Trials EDA using Pandas

## 📖 Overview  
The **COVID-19 Clinical Trials EDA** project provides a comprehensive exploratory data analysis of global COVID-19 clinical trials.  
Using **Pandas**, **NumPy**, and **Seaborn**, this notebook examines patterns in trial phases, locations, sponsors, and intervention types — offering valuable insights into how medical research evolved during the pandemic.  

The project also includes a **pre-processed data/model file (`.pkl`)** hosted on Google Drive for quick access and reuse in further analysis or ML pipelines.  

---

## 🎯 Objectives  
- Perform detailed **Exploratory Data Analysis (EDA)** on COVID-19 clinical trials dataset.  
- Understand trends across **study phases**, **intervention types**, and **sponsors**.  
- Visualize the **global research response** and activity during the pandemic.  
- Provide a **ready-to-load `.pkl` file** containing processed results or ML-ready data.  

---

## 📊 Dataset  
- **Source:** [ClinicalTrials.gov COVID-19 Dataset](https://clinicaltrials.gov)  
  or [Kaggle – COVID-19 Clinical Trials Dataset](https://www.kaggle.com/)  
- **File Format:** CSV (`covid19_trials.csv`)  
- **Sample Columns:**  
  - `Study Title`  
  - `Status`  
  - `Study Type`  
  - `Intervention Type`  
  - `Phase`  
  - `Start Date`  
  - `Completion Date`  
  - `Location`  
  - `Sponsor`  

---

## 🧰 Technologies Used  

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python 3.8+ |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook / Google Colab |
| **Model Storage** | Pickle (`.pkl` format) |

---

## 🧾 Project Workflow  

### 1️⃣ Data Import & Cleaning  
- Import dataset using Pandas  
- Handle missing or inconsistent values  
- Convert date columns and categorical variables  
- Filter relevant COVID-19 trial records  

### 2️⃣ Exploratory Data Analysis (EDA)  
- Analyze **study phase distribution**  
- Examine **country-wise** and **sponsor-wise** trial trends  
- Compare **active vs completed** trial status  
- Observe **time-series trends** in study registrations  

### 3️⃣ Visualization  
- Create visual insights using:  
  - Bar plots  
  - Pie charts  
  - Count plots  
  - Heatmaps  
  - Interactive summary charts  

### 4️⃣ Model Saving  
- Extract and store key analytical features  
- Save processed DataFrame and EDA summary results as `.pkl`  

---

## 📦 Model File (PKL Download Section)  

To ensure reproducibility and easy loading, the processed `.pkl` model/data file is available on Google Drive.  

### 📁 Download Model File  
> 👉 [**Click here to download `covid19_clinical_trials.pkl`**](https://drive.google.com/your_model_link_here)

> 💡 **Note:** Replace the above link with your actual Google Drive shareable link to the `.pkl` file.  

---
