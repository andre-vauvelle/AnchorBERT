# :anchor: AnchorBERT: Phenotyping with Positive Unlabelled Learning for Genome-Wide Association Studies

This repository contains the code used in Phenotyping with Positive Unlabelled Learning for Genome-Wide Association
Studies.

## Abstract

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

## Requirements and Data preprocessing

Without access to UK Biobank data it is difficult to fully reproduce this work. We provide some examples simulated data
to run phenotyping experiments but results will be meaningless without the real data.

Python package requirement at: `src/requirements.sh`

```{zsh}
conda install --file requirements.txt
```

### Setup Sacred

We use sacred to log all of our experiments. This requires setting up a mongo db, install from
here https://docs.mongodb.com/manual/installation/. Alternatively, you can use a local
FileStorageObserver https://sacred.readthedocs.io/en/stable/observers.html#adding-a-filestorageobserver.

If running FileStorageObserver leave `MONGO_DB=` in `src/.env` and add `--file_storage=BASEDIR` to all python commands.

### Setup .env

Create/edit the required paths for the `src/.env` file. An example is shown at `src/.env_example`.
Create `<DATA_DIR>/raw`, `<DATA_DIR>/interim`, `<DATA_DIR>/processed` dirs.

```{zsh}
mkdir $DATA_DIR/raw  $DATA_DIR/interim $DATA_DIR/processed
```

### Source environment

Import your environmental variables, setup cuda, conda etc...

```
source jobs/import.sh
```

### Preprocess data

This project requires access to the UK Biobank Primary and Secondary care data. Place files in `<DATA_DIR>raw/` and run
the following:

```{zsh}
python src/data/preprocess_raw.py with  \
    patient_base_raw_path = /your/path/to/patient_base.csv \
    event_data_path = /your/path/to/hesin.tsv \
    diag_event_data_path=/your/path/to/hesin_diag.tsv \
    opcs_event_data_path=/your/path/to/hesin_oper.tsv \
    gp_event_data_path=/your/path/to/gp_clinical.tsv 
```

This will create the required files at `$DATA_DIR/raw`.

Alternatively, you can use your own data with minimal modification if you can create than same list of lists of
strings format, `List[List[String]]`. Saved in .parquet format.

## Train Anchor Classifier Models

To run anchor variable models and baselines for a specific `<token>`:

```{zsh}
python src/task/pheno.py with target_token=<token> 
```

## Run GWAS

If you have access to QC's UK Biobank GWAS data you can run plink scrips.

To run GWAS

```{zsh}
bash jobs/pheno_plink.sh <Pfile> <phenotype_file.csv>
```

## Extracting results

To extract results from sacred mongodb:

```{zsh}
python task/analysis/full_data_anchor_metrics_results.py
python task/analysis/data_ablation_gwas_results.py
python task/analysis/snp_assoc_catalog_results.py
python task/analysis/snp_assoc_ablation_results.py
```

## Jobs

The experiments for this work were done through the UCL CS HPC with
SGE: http://gridscheduler.sourceforge.net/htmlman/manuals.html.
