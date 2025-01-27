# Machine Translation: English-Dzongkha

## Background
The development of bilingual neural machine translation (NMT) systems for Dzongkha and English was initiated with aims to enable effective communication for government, educational, and cultural exchanges between Dzongkha-speaking communities and English-speaking populations, supporting the preservation and dissemination of Dzongkha literature and cultural content in a global context
and making information more accessible to non-Dzongkha speakers while providing Dzongkha speakers with resources in English.

To achieve this, we began by training a transformer model on a parallel corpus of 10,000 sentence pairs acquired from the Department of Culture and Dzongkha Development (DoCDD) through a collaboration with Omdena. While this provided a valuable starting point, the dataset was insufficient for robust model training. To address this, we expanded our dataset by scraping additional data from online resources, including websites, English-Dzongkha and Dzongkha-English dictionaries, and various documents. This effort resulted in a collection of approximately 165,000 sentence pairs. However, a significant portion of these sentence pairs could not be fully verified due to the substantial resources and manpower required for validation. To improve the accuracy and consistency of the training data, we collaborated with DoCDD, which provided an additional 43,000
verified sentence pairs. This partnership significantly enhanced the quality and reliability of our dataset, enabling further development of our model. The total of just over 200,000 sentence pairs is still significantly less than the requirement of at least 1 million sentence pairs as per the global standard for low-resource language translation model.

## Setup
clone the project:
```bash
git clone https://gitlab.bhutansfl.com/software_development/neural-machine-translation.git
```

Open the project directory (cloned project)
```bash
cd neural-machine-translation
```
Create a new python environment for the project:
```bash
python -m venv nmt
```
Activate the newly created environment:
```bash
source nmt/bin/activate  # On Windows: .\venv\Scripts\activate
```
Install all the dependencies:
```bash
pip install -r requirements.txt
```