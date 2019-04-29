
# Fast Concept Mention Grouping for Concept Map-based Multi-Document Summarization

Accompanying code for our [NAACL 2019 paper](https://tubiblio.ulb.tu-darmstadt.de/111692/). Please use the following citation:

```
@inproceedings{naacl19-lshcw,
	title = {Fast Concept Mention Grouping for Concept Map-based Multi-Document Summarization},
	author = {Falke, Tobias and Gurevych, Iryna},
	booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
	year = {2019},
	location = {Minneapolis, USA}
}
```

> **Abstract:** Concept map–based multi-document summarization has recently been proposed as a variant of the traditional summarization task with graph-structured summaries. As shown by previous work, the grouping of coreferent concept mentions across documents is a crucial subtask of it. However, while the current state-of-the-art method suggested a new grouping method that was shown to improve the summary quality, its use of pairwise comparisons leads to polynomial runtime complexity that prohibits the application to large document collections. In this paper, we propose two alternative grouping techniques based on locality sensitive hashing, approximate nearest neighbor search and a fast clustering algorithm. They exhibit linear and log-linear runtime complexity, making them much more scalable. We report experimental results that confirm the improved runtime behavior while also showing that the quality of the summary concept maps remains comparable.

**Contacts** 
  * https://www.ukp.tu-darmstadt.de
  * https://www.aiphes.tu-darmstadt.de

Don't hesitate to send us an e-mail or report an issue if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Requirements

The code provided here is based on this implementation: https://github.com/UKPLab/ijcnlp2017-cmaps

Make sure you have that codebase available, all dependencies documented there installed and you can completely run that pipeline.

## Usage

To reproduce the experiments reported in this paper, follow the instructions A step 1-4 of the baseline pipeline linked above. Running that pipeline as provided yields the results referred to as *Reference* in the paper. Results for *lemma-only* can also be obtained using the pure baseline implementation.

To reproduce the results of *w2v-only*, *LSH-only* and *LSH-CW*, add the models and java files provided in this repository to the baseline.

Before running step 3, change the grouping strategy by replacing `ConceptGrouperSimLog` in `PipelineGraph.java`

```
String[] pipeline = { "extraction.PropositionExtractor",
                      "grouping.ConceptGrouperSimLog",
                      "grouping.ExtractionResultsSerializer" };
```

with the corresponding class provided in this repository (`ConceptGrouperSimLogW2V`, `ConceptGrouperLSH` or `ConceptGrouperLSHCW`).

Before running steps 4.3 and 4.4, change the paths (see instructions of the baseline) to the corresponding model files provided in this repository.

All other steps can be run without changes.

