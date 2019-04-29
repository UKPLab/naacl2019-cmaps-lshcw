package grouping;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
 * Concept grouping with LSH only
 * 
 * @author falke
 *
 */
public class ConceptGrouperLSH extends ConceptGrouperBase implements Extractor {

	public static int seed = 42;
	public static int d = 17;

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
		System.out.println("unknowns: " + unks + " " + (unks / repConcepts.size()));

		// 2) create random vectors
		Nd4j.getRandom().setSeed(seed);
		INDArray dVectors = Nd4j.randn(embLookup.getDim(), d);

		// 3) compute hashes
		INDArray dotProducts = embs.mmul(dVectors);

		// 4) binning based on hashes
		Map<String, List<Concept>> binnedConcepts = new HashMap<String, List<Concept>>();
		for (int i = 0; i < repConcepts.size(); i++) {
			String signature = this.getSignature(dotProducts.getRow(i));
			if (!binnedConcepts.containsKey(signature))
				binnedConcepts.put(signature, new LinkedList<Concept>());
			binnedConcepts.get(signature).add(repConcepts.get(i));
		}
		System.out.println("unique signature: " + binnedConcepts.size());

		// create final cluster and update relations
		// (also sorts mentions to get representative mention)
		Set<List<Concept>> clusters = new HashSet<List<Concept>>(binnedConcepts.values());
		this.updateDataStructures(clusters, groups);
		this.clusters = clusters;

		this.parent.log(this, "grouped concepts: " + concepts.size());
		this.parent.log(this, "relations: " + propositions.size());
	}

	private String getSignature(INDArray dotProductsRow) {
		StringBuffer sig = new StringBuffer();
		for (int i = 0; i < dotProductsRow.columns(); i++) {
			if (dotProductsRow.getDouble(i) < 0)
				sig.append("0");
			else
				sig.append("1");
		}
		return sig.toString();
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
