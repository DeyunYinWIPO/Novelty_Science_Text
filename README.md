# Novelty_Element

Code for Measuring Novelty/Originality of all scientific publications in Web of Science Database with Word Embeddings 

## Basic info

- **Authors**: Deyun Yin,Zhao Wu,Kazuki Yokota,Kuniko Matsumoto,Sotaro Shibayama
- **Abstract**: As novelty is a core value in science, a reliable approach to measuring the novelty of scientific documents is critical. Previous novelty measures however had a few limitations. First, the majority of previous measures are based on recombinant novelty concept, attempting to identify a novel combination of knowledge elements, but insufficient effort has been made to identify a novel element itself (element novelty). Second, most previous measures are not validated, and it is unclear what aspect of newness is measured. Third, some of the previous measures can be computed only in certain scientific fields for technical constraints. This study thus aims to provide a validated and field-universal approach to computing element novelty. We drew on machine learning to develop a word embedding model, which allows us to extract semantic information from text data. Our validation analyses suggest that our word embedding model does convey semantic information. Based on the trained word embedding, we quantified the element novelty of a document by measuring its distance from the rest of the document universe. We then carried out a questionnaire survey to obtain self-reported novelty scores from 800 scientists. We found that our element novelty measure is significantly correlated with self-reported novelty in terms of discovering and identifying new phenomena, substances, molecules, etc. and that this correlation is observed across different scientific fields.
- **Keywords**: Novelty, Originality, Scientific Publications, Web of Science, Word Embeddings, Word2Vec
- **Citation**: Yin D, Wu Z, Yokota K, Matsumoto K, Shibayama S (2023) [Identify novel elements of knowledge with word embedding](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0284567). PLoS ONE 18(6): e0284567. https://doi.org/10.1371/journal.pone.0284567

## Explanation of algorithms

Our algorithm measures novelty/originality of scientific publications based on word embeddings. Users can change to their own preferred document embeddings. 

 ### Features

- **Word Embedding Training**: Train a Word2Vec model on scientific publications to capture semantic relationships between words.
- **Document Vectorization**: Convert word embeddings into document-level vectors for comprehensive analysis.
- **Novelty Calculation**: Compute novelty metrics based on vector similarities.

### Input data 

- **Input data for Training word embedding model**: 
  - text: title and abstract of the 23 million scientific publications

- **Input data for Computing document vector**: 
  - id: ID of scientific publications 
  - sci_text: title and abstract 
  - model: trained word embedding model 

- **Input data for Computing element novelty**: 
  - id: ID of scientific publications 
  - vec: document vector of scientific publications

## Algorithm

<img src="https://github.com/DeyunYinWIPO/Novelty_Science_Text/tree/main/imgs/Novelty Science02.png" style="zoom:50%;" />

## **Contact**

Regarding paper and algorithm:

- Shibayama Sotaro: https://sotaroshibayama.weebly.com/contact.html 
- Yin Deyun: yindeyunut@gmail.com 

Regarding code: Dr. Wu Zhao (HITSZ): wuzhao@stu.hit.edu.cn
