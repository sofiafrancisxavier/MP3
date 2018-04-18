/**
 * 
 */
package analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.LinkedHashMap;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import org.tartarus.snowball.ext.porterStemmer;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.LanguageModel;
import structures.Post;
import structures.PostHashDoubleComparator;
import structures.StringHashDoubleComparator;
import structures.Token;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 * NOTE: the code here is only for demonstration purpose, 
 * please revise it accordingly to maximize your implementation's efficiency!
 */
public class DocAnalyzer {
	//MP3
	HashMap<String, Integer> m_tf;
	int m_NP; //number of positive documents
	int m_NN; //number of negative documents
	HashMap<String, Integer> m_posUni; //unigrams in positive docs
	HashMap<String, Integer> m_negUni; //unigrams in negative docs
	HashMap<String, Double> m_infGain; //information gain
	HashMap<String, Double> m_chiSq; //chi square
	HashMap<String, Token> m_uniPosCorpusModel = new HashMap<String, Token>();
	HashMap<String, Token> m_uniNegCorpusModel = new HashMap<String, Token>();
	int m_uniPosTot;
	int m_uniNegTot;
	HashMap<String, Double> m_logRatio;
	double m_delta;
	ArrayList<Post> m_corpus;
	HashMap<Post, Double> fx;
	
	
	
	
	//N-gram to be created
	int m_N;
	int m_UN;
	int m_BN;
	int m_unitot;
	int m_bitot;
	
	//a list of stopwords
	HashSet<String> m_stopwords;
	
	//you can store the loaded reviews in this arraylist for further processing
	ArrayList<Post> m_reviews;
	
	//you might need something like this to store the counting statistics for validating Zipf's and computing IDF
	HashMap<String, Token> m_stats;	
	HashMap<String, Token> m_unistats;
	HashMap<String, Token> m_bistats;
	
	HashMap<String, Integer> m_utf;
	HashMap<String, Integer> m_bitf;
	HashMap<String, Integer> m_df;
	HashMap<String, Double> m_idf;
	HashMap<String, Integer> m_ndf;
	HashMap<String, Double> linearInter;
	List<Double> PerpLinear;
	List<Double> PerpAbs;
	List<Double> PerpUni;
	Double meanUni;
	Double meanBiL;
	Double meanBiA;
	int testDoc_N;
	
	HashMap<String, HashMap<String, Integer>> m_testReview;
	HashSet<String> m_vocabulary;
	
	//we have also provided a sample implementation of language model in src.structures.LanguageModel
	Tokenizer m_tokenizer;
	
	//this structure is for language modeling
	LanguageModel m_langModel;
	
	public DocAnalyzer(String tokenModel, int N) throws InvalidFormatException, FileNotFoundException, IOException {
		//MP3
		m_tf = new HashMap<String, Integer>();
		m_NP = 0;
		m_NN =0;
		m_posUni = new HashMap<String, Integer>();
		m_negUni = new HashMap<String, Integer>();
		m_infGain = new HashMap<String, Double>();
		m_chiSq = new HashMap<String, Double>();
		m_uniPosCorpusModel = new HashMap<String, Token>();
		m_uniNegCorpusModel = new HashMap<String, Token>();
		m_uniPosTot =0;
		m_uniNegTot =0;
		m_logRatio = new HashMap<String, Double>();
		m_delta =0.1;
		m_corpus = new ArrayList<Post>();
		
		
		
		m_N = 0;
		testDoc_N=0;
		meanUni=(double) 0;
		meanBiL=(double) 0;
		meanBiA=(double) 0;
		m_UN = 0;
		m_BN = 0;
		m_unitot = 0;
		m_bitot = 0;
		m_reviews = new ArrayList<Post>();
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stopwords = new HashSet<String>();
		
		m_bitf = new HashMap<String, Integer>();
		m_utf = new HashMap<String, Integer>();
		m_df = new HashMap<String, Integer>();
		m_ndf = new HashMap<String, Integer>();
		m_idf = new HashMap<String, Double>();
		m_testReview = new HashMap<String, HashMap<String, Integer>>();
		m_vocabulary = new HashSet<String>();
		m_unistats = new HashMap<String, Token>();
		m_bistats = new HashMap<String, Token>();
		linearInter = new HashMap<String, Double>();
		PerpLinear = new ArrayList<>();
		PerpAbs = new ArrayList<>();
		PerpUni = new ArrayList<>();
		
	}
	//sample code for loading a list of stopwords from file
	//you can manually modify the stopword file to include your newly selected words
	public void LoadStopwords(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				//it is very important that you perform the same processing operation to the loaded stopwords
				//otherwise it won't be matched in the text content
				line = SnowballStemming(Normalization(line));
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s\n", m_stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	public void writeToCSV() {
		
		PrintWriter pw = null;
		try {
		    pw = new PrintWriter(new File("./Data/ttfData.csv"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		String ColumnNamesList = "vocabulary,ttf,rank";
		// No need give the headers Like: id, Name on builder.append
		builder.append(ColumnNamesList +"\n");
		m_tf = sortByValues(m_tf);
		int x=1;
		for(String S : m_tf.keySet())
		{
			builder.append(S+",");
			builder.append(m_tf.get(S)+",");
			builder.append(x);
			builder.append('\n');
			x=x+1;
		}
		pw.write(builder.toString());
		pw.close();
		
		PrintWriter pw1 = null;
		try {
		    pw1 = new PrintWriter(new File("./Data/tfidf.csv"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		builder = new StringBuilder();
		// No need give the headers Like: id, Name on builder.append
		ColumnNamesList = "vocabulary,ttf,df,idf";
		builder.append(ColumnNamesList +"\n");
		for(String S : m_vocabulary)
		{
			builder.append(S+",");
			builder.append(m_bitf.get(S));
			builder.append(","+m_df.get(S));
			builder.append(","+m_idf.get(S));
			
			builder.append('\n');
		}
		pw1.write(builder.toString());
		pw1.close();
		
		System.out.println("done!");
	}
	
	public void computeIdf()
	{
		for(String S : m_vocabulary)
		{
			Double value = 1 + Math.log10(m_N/m_df.get(S));
			m_idf.put(S, value);
		}
	}
	
	public void cosineSimilarity(JSONObject json) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				String[] tokens = Tokenize(review.getContent());
				HashMap<String,Double> docTokens = new HashMap<String,Double>();
				String prevWord = "";
				
				for (int j =0; j < tokens.length; j++)
				{
					String word = tokens[j];
					String biword = "";
					word = SnowballStemming(Normalization(word));
					
					if (!word.isEmpty())
					{
						if (m_vocabulary.contains(word))
						{
							if (docTokens.containsKey(word))
							{
								docTokens.put(word,docTokens.get(word)+1);
							}
							else
							{
								docTokens.put(word,1.0);
							}
							//System.out.println(word+m_tf.get(word));
						}
						if (!prevWord.isEmpty())
						{
							biword = prevWord + "-"+word;
							prevWord = word;
							if (m_vocabulary.contains(biword))
							{
								if (docTokens.containsKey(biword))
								{
									docTokens.put(biword,docTokens.get(biword)+1);
								}
								else
								{
									docTokens.put(biword,1.0);
								}
 							}
							
						}
						else
						{
							prevWord = word;
						}						 
						
					}
					
				}
				double mod = 0;
				for(String S : docTokens.keySet()) {
					docTokens.put(S, m_idf.get(S)*(1+Math.log10(docTokens.get(S))));
					mod = mod + Math.pow(docTokens.get(S),2);
					
				}
				mod = Math.sqrt(mod);
				review.setVct(docTokens);
				review.setTokens(tokens);
				review.setMod(mod);
				HashMap<Post, Double> sim = new HashMap<Post, Double>();
				for (Post p: m_reviews) {
					double simi = review.similiarity(review,p);
					if (!Double.isNaN(simi))
					{
						sim.put(p, simi);
					}
						
					//System.out.println(p.getMod()+" "+ review.getMod());
					//System.out.println(review.similiarity(p));
				}
				sim = sortByValues(sim);
				//System.out.println(sim);
				System.out.println("Query: \nAuthor: "+review.getAuthor() +"\n"+"Content: "+review.getContent()+"\n"+"Date: "+review.getDate()+"\n");
				int f=0;
				for (Post P: sim.keySet()) {
					if (f<3) {
						System.out.println("Cosine similarity: "+sim.get(P) +"\nAuthor: "+P.getAuthor() +"\n"+"Content: "+P.getContent()+"\n"+"Date: "+P.getDate()+"\n");
					}
					f=f+1;
				}
				
				//return top 3
			}
	} catch (JSONException e) {
		e.printStackTrace();
	}
	}
	
	public HashMap sortByValues(HashMap map) { 
	       List list = new LinkedList(map.entrySet());
	       // Defined Custom Comparator here
	       Collections.sort(list, new Comparator() {
	            public int compare(Object o1, Object o2) {
	               return ((Comparable) ((Map.Entry) (o2)).getValue())
	                  .compareTo(((Map.Entry) (o1)).getValue());
	            }
	       });

	       // Here I am copying the sorted list in HashMap
	       // using LinkedHashMap to preserve the insertion order
	       HashMap sortedHashMap = new LinkedHashMap();
	       for (Iterator it = list.iterator(); it.hasNext();) {
	              Map.Entry entry = (Map.Entry) it.next();
	              sortedHashMap.put(entry.getKey(), entry.getValue());
	       } 
	       return sortedHashMap;
	  }
	
	public void encodeDocument(JSONObject json) {
			try {
				//System.out.println("idf: "+m_idf);
				JSONArray jarray = json.getJSONArray("Reviews");
				for(int i=0; i<jarray.length(); i++) {
					Post review = new Post(jarray.getJSONObject(i));
					String[] tokens = Tokenize(review.getContent());
					HashMap<String,Double> docTokens = new HashMap<String,Double>();
					String prevWord = "";
					
					for (int j =0; j < tokens.length; j++)
					{
						String word = tokens[j];
						String biword = "";
						word = SnowballStemming(Normalization(word));
						
						if (!word.isEmpty())
						{
							if (m_vocabulary.contains(word))
							{
								if (docTokens.containsKey(word))
								{
									docTokens.put(word,docTokens.get(word)+1);
								}
								else
								{
									docTokens.put(word,1.0);
								}
								//System.out.println(word+m_tf.get(word));
							}
							if (!prevWord.isEmpty())
							{
								biword = prevWord + "-"+word;
								prevWord = word;
								if (m_vocabulary.contains(biword))
								{
									if (docTokens.containsKey(biword))
									{
										docTokens.put(biword,docTokens.get(biword)+1);
									}
									else
									{
										docTokens.put(biword,1.0);
									}
								}
								
							}
							else
							{
								prevWord = word;
							}						 
							
						}
						
					}
					double mod = 0;
					for(String S : docTokens.keySet()) {
						//System.out.println("dc:"+m_idf.get(S));
						double d = m_idf.get(S)*(1+Math.log10(docTokens.get(S)));
						//System.out.println("d"+d);
						docTokens.put(S, d);
						mod = mod + Math.pow(docTokens.get(S),2);
						//System.out.println("temp mod:"+mod);
					}
					//System.out.println("mod sum"+mod);
					mod = Math.sqrt(mod);
					//System.out.println("mod"+mod);
					review.setVct(docTokens);
					review.setTokens(tokens);
					review.setMod(mod);
					m_reviews.add(review);
				}
				
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	public void analyzeDocument(JSONObject json) {		
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			m_N = m_N+jarray.length();
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				String[] tokens = Tokenize(review.getContent());
				HashSet<String> docTokens = new HashSet<String>();
				//String prevWord = "";
				
				for (int j =0; j < tokens.length; j++)
				{
					String word = tokens[j];
					//String biWord = "";
					word = SnowballStemming(Normalization(word));
					
					if (!word.isEmpty())
					{
						if (!m_stopwords.contains(word))
						{
							if (!docTokens.contains(word))
							{
								docTokens.add(word);
							}
							if (m_tf.containsKey(word))
							{
								int x= m_tf.get(word)+1;
								m_tf.put(word, x);
								//m_bitf.put(word, x);
							}
							else
							{
								m_tf.put(word,1);
								//m_bitf.put(word,1);
							}
							//System.out.println(word+m_tf.get(word));
						}				 
						
					}
					
				}
				review.setTokens(docTokens.toArray(new String[docTokens.size()]));
				//System.out.println(m_tf.keySet());
				//System.out.println(review.getTokens().length);
				m_reviews.add(review);
				if (review.getRating()<4)
				{ 
					m_NN +=1;
					for (String S: docTokens){
						if (m_negUni.containsKey(S))
						{
							m_negUni.put(S, m_negUni.get(S)+1);
						}
						else
						{
							m_negUni.put(S,1);
						}
					}
				}
				else {
					m_NP +=1;
					for (String S: docTokens) {
						if (m_posUni.containsKey(S))
						{
							m_posUni.put(S, m_posUni.get(S)+1);
						}
						else
						{
							m_posUni.put(S,1);
						}
					}
				}
				for (String S: docTokens)
				{ 
					if (m_df.containsKey(S))
					{
						m_df.put(S, m_df.get(S)+1);
					}
					else
					{
						m_df.put(S,1);
					}
				}
			}
			//System.out.println(m_reviews.get(0).getContent()); 
			
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	public HashSet<String> computeInformationGain() {
		System.out.println(m_NP +" "+ m_NN);
		double ppos = (double)m_NP/(double)(m_NN+m_NP);
		double pneg = (double)m_NN/(double)(m_NN+m_NP);
		double FT1 = 0.0;
		if (ppos > 0) {
			FT1 = (ppos * Math.log(ppos));
		}
		double FT2 = 0.0;
		if (pneg > 0) {
			FT2 = (pneg * Math.log(pneg));
		}
		double firstTerm = -1 * (FT1 + FT2);
		for (String S : m_df.keySet()) {
			if(m_df.get(S)>=10) {
				double PT = (double)m_df.get(S)/(double)m_reviews.size(); //probability of T
				double PPT = 0.0; //probability of a positive document where we observe the term t
				if (m_posUni.containsKey(S)) {
					 PPT = (double)m_posUni.get(S)/(double)m_NP; 
				}
				double PNT = 0.0; //probability of a negative document where we observe the term t
				if (m_negUni.containsKey(S)) {
					 PNT = (double)m_negUni.get(S)/(double)m_NN; 
				}
				double ST1 = 0.0;
				if (PPT > 0) {
					ST1 = (PPT * Math.log(PPT));
				}
				double ST2 = 0.0;
				if (PNT > 0) {
					ST2 = (PNT * Math.log(PNT));
				}
				double secondTerm = PT * ( ST1 + ST2); // p(t) ∑y=0,1 p(y|t) log p(y|t)
				double PTN = 1.0 - PT;
				double PPTN =  1.0 - PPT;//probability of a positive document where we not observe the term t
				double PNTN = 1.0 - PNT;//probability of a negative document where we not observe the term t
				double TT1 = 0.0;
				if (PPTN > 0) {
					TT1 = (PPTN * Math.log(PPTN));
				}
				double TT2 = 0.0;
				if (PNTN > 0) {
					TT2 = (PNTN * Math.log(PNTN));
				}
				double thirdTerm = PTN * (TT1 + TT2);
				double IG = firstTerm + secondTerm + thirdTerm;
				m_infGain.put(S, IG);
				//System.out.println(firstTerm +" "+secondTerm +" "+ thirdTerm + " "+IG);
			}
		}
		List<Entry<String, Double>> entryList = new ArrayList<Entry<String, Double>>(m_infGain.entrySet());
		Collections.sort(entryList,new StringHashDoubleComparator());
		System.out.println("\nSize of IG: "+ m_infGain.size());
		System.out.println("\nTop 20 words from Information Gain: ");
		for (int i=0; i<20;i++) {
			System.out.println(entryList.get(i));
		}
		HashSet<String> returnSet = new HashSet<String>();
		for (int i=0; i<5000 && i<entryList.size();i++) {
			returnSet.add(entryList.get(i).getKey());
		}
		System.out.println("\nIG return set: "+returnSet.size());
		return returnSet;
	}
	
	public HashSet<String> computeChiSquare() {
		for (String S : m_df.keySet()) {
			if(m_df.get(S)>=10) {
				double A = 0.0;
				if (m_posUni.containsKey(S)) {
					 A = (double) m_posUni.get(S); 
				}
				double B = (double) (m_NP - A);
				double C = 0.0;
				if (m_negUni.containsKey(S)) {
					 C = (double) m_negUni.get(S); 
				}
				double D = (double) (m_NN - C);
				double numerator = (A + B + C + D) * Math.pow(((A * D)-(B * C)),2);
				double denominator = (A + C) * (B + D) * (A + B) * (C + D);
				double cs = numerator/denominator; 
				if (cs >= 3.841) {
					m_chiSq.put(S, cs);
				}
			}
		}
		List<Entry<String, Double>> entryList = new ArrayList<Entry<String, Double>>(m_chiSq.entrySet());
		Collections.sort(entryList,new StringHashDoubleComparator());
		System.out.println("\nSize of chi Square: "+m_chiSq.size());
		System.out.println("\nTop 20 words from chi Square: ");
		for (int i=0; i<20;i++) {
			System.out.println(entryList.get(i));
		}
		HashSet<String> returnSet = new HashSet<String>();
		for (int i=0; i<5000 && i<entryList.size();i++) {
			returnSet.add(entryList.get(i).getKey());
		}
		System.out.println("\nIG return set: "+returnSet.size());
		return returnSet;
	}

	public void computeControlledVocabularyAndCorpus() {
		m_vocabulary = new HashSet<String>();
		System.out.println("\nbefore: "+m_vocabulary.size());
		m_vocabulary.addAll(computeInformationGain());
		System.out.println("\nmid: "+m_vocabulary.size());
		m_vocabulary.addAll(computeChiSquare());
		System.out.println("\nafter: "+m_vocabulary.size());
		System.out.println("\nSize of vacab: "+m_vocabulary.size());
		m_NP = 0;
		m_NN = 0;
		for (Post review: m_reviews) {
			HashSet<String> docTokens = new HashSet<String>(Arrays.asList(review.getTokens()));
			HashSet<String> toks = new HashSet<String>();
			docTokens.retainAll(m_vocabulary);
			if (docTokens.size() > 5) {
				String[] tokens = Tokenize(review.getContent());
				if (review.getRating()<4) {
					m_NN += 1;
					for (int j =0; j < tokens.length; j++)
					{
						String word = tokens[j];
						word = SnowballStemming(Normalization(word));
						if (!word.isEmpty() && m_vocabulary.contains(word)) {
							if (!toks.contains(word))
							{
								toks.add(word);
							}
							m_uniNegTot +=1;
							m_unitot +=1;
							if (m_uniNegCorpusModel.containsKey(word)) {
								m_uniNegCorpusModel.get(word).setID(m_uniNegCorpusModel.get(word).getID()+1);
								//m_unistats.put(word, m_utf.get(word)+1);
							}
							else {
								Token tok = new Token(word);
								tok.setID(1);
								m_uniNegCorpusModel.put(word, tok);
								//m_unistats.put(word,1);
							}	
						}
					
					}
				}
				else {
					m_NP += 1;
					for (int j =0; j < tokens.length; j++)
					{
						String word = tokens[j];
						word = SnowballStemming(Normalization(word));
						if (!word.isEmpty() && m_vocabulary.contains(word)) {
							if (!toks.contains(word))
							{
								toks.add(word);
							}
							m_uniPosTot +=1;
							m_unitot +=1;
							if (m_uniPosCorpusModel.containsKey(word)) {
								m_uniPosCorpusModel.get(word).setID(m_uniPosCorpusModel.get(word).getID()+1);
								//m_unistats.put(word, m_utf.get(word)+1);
							}
							else {
								Token tok = new Token(word);
								tok.setID(1);
								m_uniPosCorpusModel.put(word, tok);
								//m_unistats.put(word,1);
							}	
						}
					
					}
				}
				review.setTokens(toks.toArray(new String[toks.size()]));
				m_corpus.add(review);
			}
			
		}
		for(String S: m_vocabulary) {
			double pwp = 0.0;
			double pw = 0.0;
			if (m_uniPosCorpusModel.containsKey(S)) {
				pw = (double)(m_uniPosCorpusModel.get(S).getID());
				pwp = (pw + m_delta)/(m_uniPosTot+(m_vocabulary.size() * m_delta));
				m_uniPosCorpusModel.get(S).setValue(pwp);
			}
			else {
				pwp = (pw + m_delta)/(m_uniPosTot+(m_vocabulary.size() * m_delta));
				Token tok = new Token(S);
				tok.setValue(pwp);
				m_uniPosCorpusModel.put(S, tok);
			}
			
			double nwp = 0.0;
			double nw =0.0;
			if (m_uniNegCorpusModel.containsKey(S)) {
				nw = (double)(m_uniNegCorpusModel.get(S).getID());
				nwp = (nw + m_delta)/(m_uniNegTot+(m_vocabulary.size() * m_delta));
				m_uniNegCorpusModel.get(S).setValue(nwp);
			}		
			else {
				nwp = (nw + m_delta)/(m_uniNegTot+(m_vocabulary.size() * m_delta));
				Token tok = new Token(S);
				tok.setValue(nwp);
				m_uniNegCorpusModel.put(S, tok);
			}
			
			double lr = Math.log(pwp/nwp);
			m_logRatio.put(S, lr);
		}
		List<Entry<String, Double>> entryList = new ArrayList<Entry<String, Double>>(m_logRatio.entrySet());
		Collections.sort(entryList,new StringHashDoubleComparator());
		System.out.println("\nSize of log ratio: "+m_logRatio.size());
		System.out.println("\nTop 20 words from log ratio: ");
		for (int i=0; i<20;i++) {
			System.out.println(entryList.get(i));
		}
		System.out.println("\nBottom 20 words from log ratio: ");
		for (int i=m_logRatio.size()-1; i>(m_logRatio.size()-21);i--) {
			System.out.println(entryList.get(i));
		}
	}
	
	public double NaiveBayesLinearClassification(Post review) {
		//System.out.println(m_NP +" "+ m_NN);
		double ppos = (double)m_NP/(double)(m_NN+m_NP);
		double pneg = (double)m_NN/(double)(m_NN+m_NP);
		double summation = Math.log(ppos/pneg);
		for (String S: review.getTokens()) {
			summation += Math.log(m_uniPosCorpusModel.get(S).getValue()) - Math.log(m_uniNegCorpusModel.get(S).getValue());
		}
		return summation;
	}
	public void computePrecisionRecallCurve() {
		PrintWriter pw = null;
		try {
		    pw = new PrintWriter(new File("./Data/percisionRecall.csv"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		String ColumnNamesList = "fx,recall,precision";
		// No need give the headers Like: id, Name on builder.append
		builder.append(ColumnNamesList +"\n");
		fx = new HashMap<Post, Double>();
		for (Post review: m_corpus) {
			fx.put(review, NaiveBayesLinearClassification(review)) ;
		}
		List<Entry<Post, Double>> entryList = new ArrayList<Entry<Post, Double>>(fx.entrySet());
		Collections.sort(entryList,new PostHashDoubleComparator());
		//Arrays.sort(fx,comparator.reversed());
		for (int j=0;j<fx.size();j++) {
			double threshold = entryList.get(j).getValue();
			int TP=0;
			int TN=0;
			int FP=0;
			int FN=0;
			for (int k=0;k<fx.size();k++) {
				double curfx = entryList.get(k).getValue();
				double rating = entryList.get(k).getKey().getRating();
				int groundTruth = 1;
				int pred = 1;
				if (rating <4) groundTruth = 0;
				if (curfx<threshold) pred = 0;
				if (groundTruth == pred) {
					if (groundTruth == 0) TN += 1;
					else TP += 1;
				}
				else {
					if (groundTruth == 0) FP += 1;
					else FN += 1;
				}
			}
			double precision = (double)(TP)/(double)(TP+FP);
			double recall = (double)(TP)/(double)(TP+FN);
			builder.append(threshold+","+recall+","+precision+"\n");
		}
		pw.write(builder.toString());
		pw.close();
	}
	public void computeControlledVocabulary() {
		m_ndf=sortByValues(m_df);
		PrintWriter pw = null;
		try {
		    pw = new PrintWriter(new File("./Data/newStopwords.txt"));
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		}
		StringBuilder builder = new StringBuilder();
		int i = 1; 
		for (String S: m_ndf.keySet()) {
			if (i>100) {
				if(m_df.get(S)>50) {
					m_vocabulary.add(S);
				}
			}
			else {
				m_stopwords.add(S);
				builder.append(S);
				builder.append('\n');
			}
			i=i+1;
		}
		pw.write(builder.toString());
		pw.close();
		System.out.println("New vocabulary size: "+m_vocabulary.size());
	}
	

	public void LoadTrainDirectoryForLanguageModel(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				analyzeDocumentPart2(LoadJson(f.getAbsolutePath()));
			else if (f.isDirectory())
				LoadTrainDirectoryForLanguageModel(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Loading " + size + " review train documents from " + folder);
	}
	public void LoadTestDirectoryForLanguageModel(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				analyzeTestDocumentPart2(LoadJson(f.getAbsolutePath()));
			else if (f.isDirectory())
				LoadTrainDirectoryForLanguageModel(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Loading " + size + " review train documents from " + folder);
		double meanU = meanUni/testDoc_N;
		double meanbl = meanBiL/testDoc_N;
		double meanba = meanBiA/testDoc_N;
		double sdU = 0.0;
		double sdbl = 0.0;
		double sdba = 0.0;
		for(int i=0; i<PerpUni.size();i++) {
			sdU = sdU + Math.pow(PerpUni.get(i)-meanU, 2);
		}
		for(int i=0; i<PerpLinear.size();i++) {
			sdbl = sdbl + Math.pow(PerpLinear.get(i)-meanbl, 2);
			sdba = sdba + Math.pow(PerpAbs.get(i)-meanba, 2);
		}
		System.out.println("Perplexity");
		System.out.println("Unigram Mean:"+meanU+ " standard deviation" + Math.sqrt(sdU/testDoc_N));
		System.out.println("Bigram linear Mean:"+meanbl+ " standard deviation" + Math.sqrt(sdbl/testDoc_N));
		System.out.println("Bigram absolute Mean:"+meanba+ " standard deviation" + Math.sqrt(sdba/testDoc_N));
	}
	
	public void analyzeTestDocumentPart2(JSONObject json) {		
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				HashMap<String, Token> m_unistatstest = new HashMap<String, Token>();
				HashMap<String, Token> m_bistatstest = new HashMap<String, Token>();
				String[] tokens = Tokenize(review.getContent());
				String prevWord = "";
				
				for (int j =0; j < tokens.length; j++) {
					String word = tokens[j];
					String biWord = "";
					word = SnowballStemming(Normalization(word));
					if (!word.isEmpty()) {
						if (m_unistatstest.containsKey(word)) {
							m_unistatstest.get(word).setID(m_unistatstest.get(word).getID()+1);
							//m_unistats.put(word, m_utf.get(word)+1);
						}
						else {
							Token tok = new Token(word);
							tok.setID(1);
							m_unistatstest.put(word, tok);
							//m_unistats.put(word,1);
						}	
						//m_UN = m_UN+1;
						if (!prevWord.isEmpty()) {
							biWord = prevWord + "_"+word;
							prevWord = word;
							if (m_bistatstest.containsKey(biWord)) {
								m_bistatstest.get(biWord).setID(m_bistatstest.get(biWord).getID()+1);
								//m_unistats.put(word, m_utf.get(word)+1);
							}
							else {
								Token tok = new Token(biWord);
								tok.setID(1);
								m_bistatstest.put(biWord, tok);
								//m_unistats.put(word,1);
							}	
							//m_BN = m_BN+1;
						}
						else {
							prevWord = word;
						}						 
					}
				}
				//review.setTokens(tokens);
				//m_reviews.add(review);
				if (m_unistatstest.keySet().size()>0) {
					double m = m_langModel.getRefModel().logLikelihood(m_unistatstest, false);
					PerpUni.add(m);
					meanUni = meanUni + m;
				}
				if (m_bistatstest.keySet().size()>0) {
					double ml = m_langModel.logLikelihood(m_bistatstest, true);
					double ma = m_langModel.logLikelihood(m_bistatstest, false);
					PerpLinear.add(ml);
					PerpAbs.add(ma);
					meanBiL = meanBiL + ml;
					meanBiA = meanBiA + ma;
				}
				testDoc_N = testDoc_N+1;
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public void analyzeDocumentPart2(JSONObject json) {		
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				
				String[] tokens = Tokenize(review.getContent());
				String prevWord = "";
				
				for (int j =0; j < tokens.length; j++) {
					String word = tokens[j];
					String biWord = "";
					word = SnowballStemming(Normalization(word));
					if (!word.isEmpty()) {
						m_unitot +=1;
						if (m_unistats.containsKey(word)) {
							m_unistats.get(word).setID(m_unistats.get(word).getID()+1);
							//m_unistats.put(word, m_utf.get(word)+1);
						}
						else {
							Token tok = new Token(word);
							tok.setID(1);
							m_unistats.put(word, tok);
							//m_unistats.put(word,1);
						}	
						//m_UN = m_UN+1;
						if (!prevWord.isEmpty()) {
							biWord = prevWord + "_"+word;
							prevWord = word;
							m_bitot = m_bitot +1;
							if (m_bistats.containsKey(biWord)) {
								m_bistats.get(biWord).setID(m_bistats.get(biWord).getID()+1);
								//m_unistats.put(word, m_utf.get(word)+1);
							}
							else {
								Token tok = new Token(biWord);
								tok.setID(1);
								m_bistats.put(biWord, tok);
								//m_unistats.put(word,1);
							}	
							//m_BN = m_BN+1;
						}
						else {
							prevWord = word;
						}						 
					}
				}
				//review.setTokens(tokens);
				m_reviews.add(review);
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	public String[] findTopNwordslinear(String S, int N) {
		String[] linearout = new String[N];
		for(String T: m_unistats.keySet()) {
			linearInter.put(T, m_langModel.calcLinearSmoothedProb(S+"_"+T));
		}
		linearInter = sortByValues(linearInter);
		int i =0;
		System.out.println("Linear interpolation");
		for (String T:linearInter.keySet()) {
			if (i<N) {
				System.out.println(T+" "+linearInter.get(T));
				linearout[i] = T;
				i= i + 1;
			}
			else {
				break;
			}
		}
		return linearout;
	}
	public String[] findTopNwordsabsolute(String S, int N) {
		HashMap<String, Double> absDisc = new HashMap<String, Double>();
		String[] absout = new String[N];
		for(String T: m_unistats.keySet()) {
			absDisc.put(T, m_langModel.calcAbsDiscountSmoothedProb(S+"_"+T));
		}
		absDisc = sortByValues(absDisc);
		int i =0;

		System.out.println("Absolute Discount");
		for (String T:absDisc.keySet()) {
			if (i<N) {
				System.out.println(T+" "+absDisc.get(T));
				absout[i]=T;
				i= i + 1;
			}
			else {
				break;
			}
		}
		return absout;
	}

	public void createLanguageModel() {
		LanguageModel m_UniModel = new LanguageModel(1, m_unistats.size());
		m_UniModel.setModel(m_unistats);
		m_UniModel.setTotal(m_unitot);
		//m_biModel = new LanguageModel(2, m_bistats.size());
		m_langModel = new LanguageModel(2,m_bistats.size());
		m_langModel.setModel(m_bistats);
		m_langModel.setRefModel(m_UniModel);
		m_langModel.setTotal(m_bitot);
		System.out.println("Model created");
		//for(Post review:m_reviews) {
		//	String[] tokens = Tokenize(review.getContent());
			/**
			 * HINT: essentially you will perform very similar operations as what you have done in analyzeDocument() 
			 * Now you should properly update the counts in LanguageModel structure such that we can perform maximum likelihood estimation on it
			 */
			
		//}
	}
	
	//sample code for loading a json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();
			
			return new JSONObject(buffer.toString());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		} catch (JSONException e) {
			System.err.format("[Error]Failed to parse json file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}
	
	
	public void LoadDirectory(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				analyzeDocument(LoadJson(f.getAbsolutePath()));
			else if (f.isDirectory())
				LoadTrainDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Loading " + m_N + " review train documents from " + folder);
	}
	
	// sample code for demonstrating how to recursively load files in a directory 
	public void LoadTrainDirectory(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				analyzeDocument(LoadJson(f.getAbsolutePath()));
			else if (f.isDirectory())
				LoadTrainDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Loading " + m_N + " review train documents from " + folder);
		//System.out.println("token size: "+ m_tf.size());
		//System.out.println("total review : "+m_reviews );
		computeControlledVocabulary();
		computeIdf();
	}
	public void LoadTestDirectory(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				encodeDocument(LoadJson(f.getAbsolutePath()));
			else if (f.isDirectory())
				LoadTestDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Loading " + size + " review test documents from " + folder);
		//System.out.println("token size: "+ m_tf.size());
		//System.out.println("total review : "+m_reviews.size() );
	}
	
	public void Sampling() {
		for(int i =0; i< 10;i++) {
			String S="";
			String SL="";
			String SA = "";
			String seed = "";
			String seedL = "";
			String seedA = "";
			for (int j=0; j<15;j++) {
				String str = m_langModel.getRefModel().sampling("",true);
				if (j==0) {
					seed = str;
					seedL = seed;
					seedA = seed;
					S = S+ str;
					SL = SL+ str;
					SA = SA +str;
				}
				else {
					S = S + " "+ str;
					seedL = m_langModel.sampling(seedL,true);
					seedA = m_langModel.sampling(seedA,false);
					SL = SL +" "+ seedL;
					SA = SA +" "+ seedA;
				}
			}
			System.out.println("S"+i+" : "+S);
			System.out.println("SL"+i+" : "+SL);
			System.out.println("SA"+i+" : "+SA);
		}
		//m_langModel.getRefModel().sampling("good");
	}
	
	public void LoadQueryFile(String filepath) {
		//File file = new File(filepath);
		cosineSimilarity(LoadJson(filepath));
		System.out.print("query");
	}
	//sample code for demonstrating how to use Snowball stemmer
	public String SnowballStemming(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to use Porter stemmer
	public String PorterStemming(String token) {
		porterStemmer stemmer = new porterStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to perform text normalization
	//you should implement your own normalization procedure here
	public String Normalization(String token) {
		// remove all non-word characters
		// please change this to removing all English punctuation
		// convert to lower case
		token = token.toLowerCase(); 
		token = token.replaceAll(" ", "");
		// add a line to recognize integers and doubles via regular expression
		// and convert the recognized integers and doubles to a special symbol "NUM"
		token = token.replaceAll("\\d+[.]\\d+|\\d+", "NUM");
		
		//token = token.replaceAll("[.!,;:?-]|[\"'«»]|\\[|\\]|\\{|\\}|\\(|\\)|\\⟨|\\⟩|/", ""); 
		token = token.replaceAll("[.!,;:?-@*&•^+$#]|[\"'«]|-|\\[|\\]|\\{|\\}|\\(|\\)|\\⟨|\\⟩|/|\\#|№|÷|×|º|ª|%|‰|\\\\+|−|=|‱|¶|′|″|‴|§|\\\\~|_|\\\\||‖|¦|©|℗|®|℠|™|¤|؋|₳|฿|\\u20BF|₵|¢|₡|₢|\\\\$|₫|₯|֏|₠|€|ƒ|₣|₲|₴|₭|₺|\\u20BE|\\u20BC|ℳ|₥|₦|₧|₱|₰|£|元|圆|圓|﷼|៛|\\u20BD|₹|₨|₪|৳|₸|₮|₩|¥|円|⁂|,|❧|☞|‽|⸮|◊|⁀|\\\\\\\\|•|\\\\^|†|‡|°|”|¡|¿|,|,\\t|※", "");
		//token = token.replaceAll("\\W+","");
		return token;
	}
	
	String[] Tokenize(String text) {
		return m_tokenizer.tokenize(text);
	}
	
	public void TokenizerDemon(String text) {
		System.out.format("Token\tNormalization\tSnonball Stemmer\tPorter Stemmer\n");
		String[] temp = m_tokenizer.tokenize(text);
		System.out.println("length: "+temp[0]);
		for(String token:m_tokenizer.tokenize(text)){
			System.out.format("%s\t%s\t%s\t%s\n", token, Normalization(token), SnowballStemming(token), PorterStemming(token));
		}	
	}
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {		
		DocAnalyzer analyzer = new DocAnalyzer("./data/Model/en-token.bin", 2);
		analyzer.LoadStopwords("./Data/stopwords.txt");
		analyzer.LoadDirectory("./Data/yelp/train1", ".json");
		analyzer.computeControlledVocabularyAndCorpus();
		analyzer.computePrecisionRecallCurve();
		//analyzer.computeInformationGain();
		//analyzer.computeChiSquare();
		//code for demonstrating tokenization and stemming
		//analyzer.TokenizerDemon("hi i am sofia I am a good person.");
		//String res = ("the number is 22.5 22").replaceAll("\\d+[.]\\d+|\\d+", "NUM");
		//String res = ("the number is/ 2.5⟨, «but[fad] he said {sad}'hello(mad) it is 2!' that is it.").replaceAll("[.!,;:?-@*&•^+$#]|[\"'«]|\\[|\\]|\\{|\\}|\\(|\\)|\\⟨|\\⟩|/|\\#|№|÷|×|º|ª|%|‰|\\\\+|−|=|‱|¶|′|″|‴|§|\\\\~|_|\\\\||‖|¦|©|℗|®|℠|™|¤|؋|₳|฿|\\u20BF|₵|¢|₡|₢|\\\\$|₫|₯|֏|₠|€|ƒ|₣|₲|₴|₭|₺|\\u20BE|\\u20BC|ℳ|₥|₦|₧|₱|₰|£|元|圆|圓|﷼|៛|\\u20BD|₹|₨|₪|৳|₸|₮|₩|¥|円|⁂|,|❧|☞|‽|⸮|◊|⁀|\\\\\\\\|•|\\\\^|†|‡|°|”|¡|¿|,|,\\t|※", "");
		//System.out.println(res);
		//entry point to deal with a collection of documents
		//analyzer.LoadStopwords("./Data/stopwords.txt");
		//analyzer.LoadTrainDirectory("./Data/yelp/train", ".json");
		//analyzer.writeToCSV();
		//analyzer.LoadTestDirectory("./Data/yelp/test", ".json");
		//System.out.print(Math.log10(1474/2908));
		//analyzer.LoadQueryFile("./Data/samples/query.json");
		//analyzer.LoadTrainDirectoryForLanguageModel("./Data/yelp/train1", ".json");
		//analyzer.analyzeDocumentPart2();
		//String word = analyzer.SnowballStemming(analyzer.Normalization("good"));
		//analyzer.createLanguageModel();
		//analyzer.findTopNwordslinear(word,10);
		//analyzer.findTopNwordsabsolute(word,10);
		//analyzer.Sampling();
		//analyzer.LoadTestDirectoryForLanguageModel("./Data/yelp/test", ".json");
	}
}
