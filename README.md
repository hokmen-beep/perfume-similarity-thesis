# perfume-similarity-thesis
Bachelor Thesis Coding Component

Project Overview

This project investigates whether computational similarity measures derived from perfume descriptions and fragrance notes can predict human olfactory similarity judgments collected in a smell-based experiment.

Several computational similarity approaches were compared, including:

* MiniLM description embedding similarity
* TF-IDF description similarity
* fragrance-note overlap
* a combined embedding and note-overlap model

The goal of the project was to examine how well language-based perfume representations align with human olfactory similarity perception, and how they compare to structured fragrance-note information.


Dataset

The project uses a perfume dataset containing:

* perfume descriptions
* fragrance notes
* perfume metadata

The original dataset included approximately 84,000 perfumes.

A smaller subset of perfume pairs was selected for the human experiment after several filtering stages. These filtering steps aimed to:

* remove perfumes with missing descriptions
* reduce superficial lexical overlap
* reduce the effect of generic perfume terminology
* select perfumes that were physically available for the smell-based experiment

Human similarity judgments were collected from 28 participants through an in-person smell-based similarity-rating experiment conducted in Qualtrics.

Participants rated perfume-pair similarity on a 7-point scale.


Dataset Availability

The original dataset was based on a publicly available Kaggle dataset containing perfume descriptions and fragrance-note information scraped from Fragrantica.com.

At the time this project was finalized, the original online dataset was no longer publicly accessible. A local copy used during the analyses was preserved for reproducibility purposes.

Raw perfume descriptions are not redistributed in this repository due to data ownership considerations. The repository instead contains processed similarity measures, embeddings, selected perfume pairs, and analysis outputs required to reproduce the analyses.


Folder Structure

fragrantica_dataset/

Contains the preserved local copy of the original Fragrantica-based dataset used during preprocessing and dataset construction.

Main file used in the project:

* perfumes_table.csv

Additional files in the folder were kept only for reference and intermediate preprocessing purposes.

Example development path:

```python
base_path = "/Users/helinokmen/Desktop/thesis_coding_component/fragrantica_dataset"
