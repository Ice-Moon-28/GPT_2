---
annotations_creators:
- no-annotation
language_creators:
- found
language:
- en
license:
- cc0-1.0
multilinguality:
- monolingual
pretty_name: OpenWebText
size_categories:
- 1M<n<10M
source_datasets:
- original
task_categories:
- text-generation
- fill-mask
task_ids:
- language-modeling
- masked-language-modeling
paperswithcode_id: openwebtext
dataset_info:
  features:
  - name: text
    dtype: string
  config_name: plain_text
  splits:
  - name: train
    num_bytes: 39769491688
    num_examples: 8013769
  download_size: 12880189440
  dataset_size: 39769491688
---

# Dataset Card for "openwebtext"

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [https://skylion007.github.io/OpenWebTextCorpus/](https://skylion007.github.io/OpenWebTextCorpus/)
- **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Paper:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Size of downloaded dataset files:** 13.51 GB
- **Size of the generated dataset:** 41.70 GB
- **Total amount of disk used:** 55.21 GB

### Dataset Summary

An open-source replication of the WebText dataset from OpenAI, that was used to train GPT-2.

This distribution was created by Aaron Gokaslan and Vanya Cohen of Brown University.

### Supported Tasks and Leaderboards

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Dataset Structure

### Data Instances

#### plain_text

- **Size of downloaded dataset files:** 13.51 GB
- **Size of the generated dataset:** 41.70 GB
- **Total amount of disk used:** 55.21 GB

An example of 'train' looks as follows.
```
This example was too long and was cropped:

{
    "text": "\"A magazine supplement with an image of Adolf Hitler and the title 'The Unreadable Book' is pictured in Berlin. No law bans “Mei..."
}
```

### Data Fields

The data fields are the same among all splits.

#### plain_text
- `text`: a `string` feature.

### Data Splits

| name       |   train |
|------------|--------:|
| plain_text | 8013769 |

## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

The authors started by extracting all Reddit post urls from the Reddit submissions dataset. These links were deduplicated, filtered to exclude non-html content, and then shuffled randomly. The links were then distributed to several machines in parallel for download, and all web pages were extracted using the newspaper python package. Using Facebook FastText, non-English web pages were filtered out.

Subsequently, near-duplicate documents were identified using local-sensitivity hashing (LSH). Documents were hashed into sets of 5-grams and all documents that had a similarity threshold of greater than 0.5 were removed. The the remaining documents were tokenized, and documents with fewer than 128 tokens were removed. This left 38GB of text data (40GB using SI units) from 8,013,769 documents.

#### Who are the source language producers?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Annotations

The dataset doesn't contain annotations.

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

These data are released under this licensing scheme from the original authors ([source](https://skylion007.github.io/OpenWebTextCorpus/)):

```
We do not own any of the text from which these data has been extracted.

We license the actual packaging of these parallel data under the [Creative Commons CC0 license (“no rights reserved”)](https://creativecommons.org/share-your-work/public-domain/cc0/)
```

#### Notice policy

Should you consider that our data contains material that is owned by you and should therefore not be reproduced here, please:

Clearly identify yourself, with detailed contact data such as an address, telephone number or email address at which you can be contacted.

Clearly identify the copyrighted work claimed to be infringed.

Clearly identify the material that is claimed to be infringing and information reasonably sufficient to allow us to locate the material.

And contact us at the following email address: openwebtext at gmail.com and datasets at huggingface.co

#### Take down policy

The original authors will comply to legitimate requests by removing the affected sources from the next release of the corpus.
Hugging Face will also update this repository accordingly.

### Citation Information

```
@misc{Gokaslan2019OpenWeb,
    title={OpenWebText Corpus},
    author={Gokaslan, Aaron and Cohen, Vanya and Pavlick, Ellie and Tellex, Stefanie},
    howpublished={\url{http://Skylion007.github.io/OpenWebTextCorpus}},
    year={2019}
}
```

### Contributions

Thanks to [@richarddwang](https://github.com/richarddwang) for adding this dataset.
