# :anchor: AnchorBERT: Phenotyping with Positive Unlabelled Learning for Genome-Wide Association Studies

This repository contains the code used in Phenotyping with Positive Unlabelled Learning for Genome-Wide Association
Studies.

# Abstract

Identifying phenotypes plays an important role in furthering our understanding of disease biology through practical
applications within healthcare and the life sciences. The challenge of dealing with the complexities and noise within
electronic health records (EHRs) has motivated applications of machine learning in phenotypic discovery. While recent
research has focused on finding predictive subtypes for clinical decision support, here we instead focus on the noise
that results in phenotypic misclassification, which can reduce a phenotypes ability to detect associations in
genome-wide association studies
(GWAS). We show that by combining anchor learning and transformer architectures into our proposed model, AnchorBERT, we
are able to detect genomic associations only previously found in large consortium studies with 5x more cases. When
reducing the number of controls available by 50%, we find our model is able to maintain 40% more significant genomic
associations from the GWAS catalog compared to standard phenotype definitions.

# Requirements and Data preprocessing

Python package requirement at: `src/requirements.sh`

Setup mongodb

This project requires access to the UK Biobank Primary and Secondary care data.

Run the following 
```{zsh}
python src/data/preprocess_raw.py
```

# Train Anchor Classifier Models

```{zsh}

```

# Extracting results

```{zsh}
python task/analysis/full_data_anchor_metrics.py
python task/analysis/data_ablation_gwas_results.py

```

# Jobs

The experiments for this work were done through the UCL CS HPC with
SGE: http://gridscheduler.sourceforge.net/htmlman/manuals.html.
