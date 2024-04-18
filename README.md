# SklearnRAGNet
RAG System on Scikit-Learn Documentation


# Automatic Tagging of Consumer Complaints

![Python version](https://img.shields.io/badge/python-3.9+-blue.svg) ![transformers](https://img.shields.io/badge/transformers-4.39.3+-yellow.svg) ![Streamlit version](https://img.shields.io/badge/streamlit-1.31.1-red.svg) 

## Description

- Consumers can face challenges with financial products and services, leading to complaints that may not always be resolved directly with financial institutions. The **Consumer Financial Protection Bureau (CFPB)** acts as a mediator in these scenarios. However, consumers often struggle to categorize their complaints accurately, leading to inefficiencies in the resolution process.
  
- Our project aims to facilitate **faster complaint submission and resolution** by automatically categorizing complaints directly based on narrative descriptions, enhancing the efficiency of complaint management and smoothly routing it to the appropriate teams.
  
- We developed a **Hybrid** approach by leveraging language models (**BERT**) and traditional machine learning techniques to find a trade-off between computational complexity and the need for model re-training. We've fine-tuned [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) using the Consumer Complaints data on the main [product](https://huggingface.co/Mahesh9/distil-bert-fintuned-product-cfpb-complaints) and [issue](https://huggingface.co/Mahesh9/distil-bert-fintuned-issues-cfpb-complaints) categories, and used supervised classifiers for categorizing sub-products and sub-issues.


## Impact

The implementation of our project has two primary impacts:

- **Ease for Consumers:** Automates the tagging of complaints into appropriate categories, reducing the need for consumers to understand complex financial product categories.

- **Industry Adoption:** Offers a streamlined approach to complaint handling that can be adopted by financial institutions beyond the CFPB, promoting consistency across the industry.


## App Demo

![Demo](https://github.com/mahesh973/TagMyComplaint/assets/59694546/6bb06562-3d97-40f5-afb1-12add20c9812)
&nbsp;

## Model Workflow

![Model Workflow](https://github.com/mahesh973/TagMyComplaint/assets/59694546/256b0003-3807-4eb7-9e19-ab94cd09686a)

&nbsp;

## Results
![Results](https://github.com/mahesh973/TagMyComplaint/assets/59694546/df93d763-7d32-4bf7-be5d-092e60eb48cf)

&nbsp;


## User Installation

**1. Clone the Repository**
```python
git clone "https://github.com/mahesh973/TagMyComplaint.git"
```
**2. Navigate to the Directory**
```python
cd "TagMyComplaint"
```
**3. Install the necessary dependencies**
```python
pip install -r requirements.txt
```

**4. Navigate to the app Directory**
```python
cd src
```

**5. Launch the application**
```python
streamlit run main.py
```

After completing these steps, the application should be running on your local server. Open your web browser and navigate to http://localhost:8501 to start exploring the Consumer Complaint Insights 2023.


## References

- This application is built using [Streamlit](https://streamlit.io).
- For more detailed information to explore the raw data, visit the official Consumer Complaints Database: [CFPB Complaints Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Link to the data file we've used for training](https://drive.google.com/file/d/1-0KAszo-DlmnlXKhk2V677kMnHsUrD7O/view?usp=drive_link)


&nbsp;
&nbsp;

Feel free to contribute to the project by submitting issues or pull requests on GitHub. Your feedback and contributions are highly appreciated!




