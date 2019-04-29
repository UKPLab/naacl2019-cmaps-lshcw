package grouping;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import grouping.clf.sim.WordEmbeddingDistance;
import grouping.clf.sim.WordEmbeddingDistance.EmbeddingType;
import model.Concept;
import model.ConceptGroup;
import model.PToken;
import model.Proposition;
import pipeline.Extractor;
import preprocessing.NonUIMAPreprocessor;

/**
 * Concept grouping with LSH and CW clustering
 * 
 * @author falke
 *
 */
public class ConceptGrouperLSHCW extends ConceptGrouperBase implements Extractor {

	public static final int seed = 42;
	public static final Random random = new Random(seed);

	public static int d = 200; // functions used for hashing
	public static int q = 20; // permutations of hashes
	public static int b = 200; // beam for neighbor search in sorted list
	public static double minSim = 0.89; // threshold for similarity of neighbors

	@Override
	public void processCollection() {

		// get extracted concepts and propositions
		Extractor ex = this.parent.getPrevExtractor(this);
		this.concepts = ex.getConcepts();
		this.propositions = ex.getPropositions();
		for (Concept c : this.concepts)
			this.fixLemmas(c);

		// 0) group by same label
		Map<Concept, ConceptGroup> groups = LemmaGrouper.group(this.concepts);
		List<Concept> repConcepts = new ArrayList<Concept>(groups.keySet());
		int n = repConcepts.size();
		this.parent.log(this, "unique concepts: " + groups.size());

		// 1) get word embeddings
		WordEmbeddingDistance embLookup = new WordEmbeddingDistance(EmbeddingType.WORD2VEC, 300, false);
		List<INDArray> embList = new ArrayList<INDArray>();
		INDArray unkEmb = embLookup.getConceptVector(new Concept("<UNK>"));
		double unks = 0;
		for (Concept c : repConcepts) {
			INDArray e = embLookup.getConceptVector(c);
			if (e != null) {
				embList.add(e);
			} else {
				embList.add(unkEmb);
				unks += 1;
			}
		}
		INDArray embs = Nd4j.vstack(embList);
		this.parent.log(this, "unknowns: " + unks + " " + (unks / n));

		// 2) create random vectors
		Nd4j.getRandom().setSeed(seed);
		INDArray dVectors = Nd4j.randn(embLookup.getDim(), d);

		// 3) compute hashes
		INDArray dotProducts = embs.mmul(dVectors);
		boolean[][] hashes = new boolean[n][d];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				if (dotProducts.getFloat(i, j) >= 0) {
					hashes[i][j] = true;
				} else {
					hashes[i][j] = false;
				}
			}
		}

		// 4) create permutation functions for fast hamming search
		List<List<Integer>> permutations = new ArrayList<List<Integer>>();
		for (int i = 0; i < q; i++) {
			List<Integer> indices = IntStream.rangeClosed(0, d - 1).boxed().collect(Collectors.toList());
			Collections.shuffle(indices, random);
			permutations.add(indices);
		}

		// 5) fast hamming search: find close vectors
		Map<Concept, Set<Concept>> pairsAdjList = new HashMap<Concept, Set<Concept>>();
		for (Concept c : repConcepts)
			pairsAdjList.put(c, new HashSet<Concept>());
		int nbPairs = 0;

		List<Integer> conceptIndices = IntStream.rangeClosed(0, n - 1).boxed().collect(Collectors.toList());
		int qi = -1;
		for (final List<Integer> permIndices : permutations) {
			qi++;

			// sort concepts by permuted hashes
			Collections.sort(conceptIndices, new Comparator<Integer>() {
				@Override
				public int compare(Integer c1, Integer c2) {
					// compare hashes in the order given by the permutation
					for (int i : permIndices) {
						int c = Boolean.compare(hashes[c1][i], hashes[c2][i]);
						if (c != 0)
							return c;
					}
					return 0;
				}
			});

			// search with beam of b in ordered list
			for (int i = 0; i < n; i++) {
				int ci = conceptIndices.get(i);
				for (int j = i + 1; j < i + b + 1; j++) {
					if (i == j || j < 0 || j >= n)
						continue;
					int cj = conceptIndices.get(j);
					double d = hammingDist(hashes[ci], hashes[cj]);
					double sim = Math.cos(d * Math.PI);
					if (sim >= minSim) {
						pairsAdjList.get(repConcepts.get(ci)).add(repConcepts.get(cj));
						pairsAdjList.get(repConcepts.get(cj)).add(repConcepts.get(ci));
					}
				}
			}

			int nbPairsNew = (pairsAdjList.values().stream().mapToInt(x -> x.size()).sum() / 2);
			System.out.println("fast hamming search " + qi + ": added " + (nbPairsNew - nbPairs) + " new pairs");
			nbPairs = nbPairsNew;
		}
		System.out.println("total pairs: " + nbPairs);

		// 6) cluster with Chinese Whispers
		Set<List<Concept>> clusters = this.runCW(repConcepts, pairsAdjList);

		// create final cluster and update relations
		// (also sorts mentions to get representative mention)
		this.updateDataStructures(clusters, groups);
		this.clusters = clusters;

		this.parent.log(this, "grouped concepts: " + concepts.size());
		this.parent.log(this, "relations: " + propositions.size());

	}

	private double hammingDist(boolean[] h1, boolean[] h2) {
		double d = 0;
		for (int i = 0; i < h1.length; i++) {
			if (h1[i] != h2[i])
				d += 1;
		}
		return d / h1.length;
	}

	private Set<List<Concept>> runCW(List<Concept> concepts, Map<Concept, Set<Concept>> pairsAdjList) {

		// intialize labels
		Map<Concept, Integer> labels = new HashMap<Concept, Integer>();
		for (int i = 0; i < concepts.size(); i++)
			labels.put(concepts.get(i), i);

		// run CW iterations
		int changed = -1;
		int it;
		for (it = 0; it < 10000 && changed != 0; it++)
			changed = this.runCWIteration(labels, pairsAdjList);
		System.out.println("cw: iterations " + it);
		if (changed != 0)
			System.out.println("cw: did not converge " + changed);

		// build and return clusters
		Map<Integer, List<Concept>> clusters = new HashMap<Integer, List<Concept>>();
		for (Entry<Concept, Integer> e : labels.entrySet()) {
			if (!clusters.containsKey(e.getValue()))
				clusters.put(e.getValue(), new ArrayList<Concept>());
			clusters.get(e.getValue()).add(e.getKey());
		}

		return new HashSet<List<Concept>>(clusters.values());
	}

	private int runCWIteration(Map<Concept, Integer> labels, Map<Concept, Set<Concept>> pairsAdjList) {

		List<Concept> concepts = new ArrayList<Concept>(labels.keySet());
		Collections.shuffle(concepts, random);

		int changed = 0;
		for (Concept c : concepts) {

			// check all neighbors
			Map<Integer, Double> labelCounts = new HashMap<Integer, Double>();
			for (Concept nb : pairsAdjList.get(c)) {
				int label = labels.get(nb);
				labelCounts.put(label, 1 + labelCounts.getOrDefault(label, 0.0));
			}

			// find label with max weight
			int newLabel = -1;
			if (labelCounts.size() == 0)
				newLabel = labels.get(c);
			else {
				double maxWeight = 0;
				for (int l : labelCounts.keySet()) {
					if (labelCounts.get(l) > maxWeight) {
						maxWeight = labelCounts.get(l);
						newLabel = l;
					}
				}
			}
			// update
			if (newLabel != labels.get(c)) {
				labels.put(c, newLabel);
				changed++;
			}
		}

		return changed;
	}

	private void fixLemmas(Concept c) {
		for (int i = 0; i < c.tokenList.size(); i++) {
			PToken t = c.tokenList.get(i);
			t = NonUIMAPreprocessor.getInstance().lemmatize(t);
		}
	}

	private void updateDataStructures(Set<List<Concept>> clusters, Map<Concept, ConceptGroup> groups) {

		// merge pre-clustering and classifier clustering
		for (List<Concept> cluster : clusters) {
			List<Concept> extra = new ArrayList<Concept>();
			for (Concept c : cluster)
				extra.addAll(groups.get(c).getAll());
			cluster.clear();
			cluster.addAll(extra);
			Collections.sort(cluster);
		}

		// build mapping
		Map<Concept, Concept> conceptMapping = new HashMap<Concept, Concept>();
		for (List<Concept> cluster : clusters) {
			Concept labelConcept = cluster.get(0);
			conceptMapping.put(labelConcept, labelConcept);
			for (Concept otherConcept : cluster.subList(0, cluster.size())) {
				conceptMapping.put(otherConcept, labelConcept);
			}
		}

		// update concepts
		this.concepts.clear();
		for (List<Concept> cluster : clusters)
			this.concepts.add(cluster.get(0));

		// adapt propositions
		List<Proposition> updated = new ArrayList<Proposition>();
		for (Proposition p : this.propositions) {
			p.sourceConcept = conceptMapping.get(p.sourceConcept);
			p.targetConcept = conceptMapping.get(p.targetConcept);
			if (p.sourceConcept != p.targetConcept)
				updated.add(p);
		}
		this.propositions = updated;
	}

}
