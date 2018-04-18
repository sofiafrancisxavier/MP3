/**
 * 
 */
package structures;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * @author hongning
 * Suggested structure for constructing N-gram language model
 */
public class LanguageModel {

	int m_N; // N-gram
	int m_V; // the vocabular size
	int m_total;
	
	public void setTotal(int T) {
		this.m_total = T;
	}
	
	public HashMap<String, Token> m_model; // sparse structure for storing the maximum likelihood estimation of LM with the seen N-grams
	//public HashMap<String, Double> m_model;
	public void setModel(HashMap<String, Token> model) {
		this.m_model =model;
	}
	
	LanguageModel m_reference; // pointer to the reference language model for smoothing purpose
	public void setRefModel(LanguageModel refmodel) {
		this.m_reference =refmodel;
	}
	public LanguageModel getRefModel() {
		return m_reference ;
	}
	
	double m_lambda; // parameter for linear interpolation smoothing
	double m_delta; // parameter for absolute discount smoothing
	
	public LanguageModel(int N, int V) {
		m_N = N;
		m_V = V;
		m_lambda = 0.9;
		m_delta = 0.1;
		//m_model = new HashMap<String, Token>();
		//m_reference = new LanguageModel(1,V);
		//m_model = new HashMap<String, Double>();
	}
	
	public double calcMLProb(String token) {
		if (m_model.containsKey(token))
			return m_model.get(token).getID()/m_total; // should be something like this
		else 
			return 0;
	}

	public double calcLinearSmoothedProb(String token) {
		if (m_N>1) {
			String[] txt = token.split("_");
			return (1.0-m_lambda) * calcMLProb(token) + m_lambda * m_reference.calcLinearSmoothedProb(txt[1]);
		}	
		else
			//return calcAdditiveSmoothedProb(token); // please use additive smoothing to smooth a unigram language model
			//return (m_model.get(token).getID()+m_delta)/(m_total+(m_delta*m_V));
			//System.out.println("cnt: "+m_model.get(token).getID()+" tot : " +m_total);
			//System.out.println(m_model.get(token).getID()/m_total);
			if (m_model.containsKey(token)){
				return (double)(m_model.get(token).getID()+0.1)/(m_total+0.1*m_V);
				//return (double)m_model.get(token).getID()/m_total;
			}
			else {
				return (double)(0.1)/(m_total+0.1*m_V);
				}
	}
	
	public double calcAbsDiscountSmoothedProb(String token) {
		int count = 0;
		if (m_model.containsKey(token)) {
			count = m_model.get(token).m_id;
		}
		
		if (m_N>1) {
			String[] txt = token.split("_");
			//System.out.println(token);
			//System.out.println(txt[0]);
			//System.out.println(m_reference.calcAbsDiscountSmoothedProb(txt[1]));
			if (!m_reference.m_model.containsKey(txt[0])) {
				return m_lambda*m_reference.calcAbsDiscountSmoothedProb(txt[1]);
			}
			return (Math.max(count-m_delta, 0)/m_reference.m_model.get(txt[0]).m_id)+m_lambda*m_reference.calcAbsDiscountSmoothedProb(txt[1]);
			//return 0;
		}
		else {
			//return m_lambda*calcAdditiveSmoothedProb(token);

				//return (m_model.get(token).getID()+m_delta)/(m_total+(m_delta*m_V));

			//return calcAdditiveSmoothedProb(token);
			if (m_model.containsKey(token)){
				return (double)(m_model.get(token).getID()+0.1)/(m_total+0.1*m_V);
				//return (double)m_model.get(token).getID()/m_total;
			}
			else {
				return (double)(0.1)/(m_total+0.1*m_V);
				}
			
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
	
	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public String sampling(String S, boolean b) {
		if (m_N == 1) {
				//System.out.println("hi");
				double prob =1.0;
				while (prob>0){
					prob = Math.random(); // prepare to perform uniform sampling
					//System.out.println("random" + prob);
					for(String token:m_model.keySet()) {
						prob -= calcLinearSmoothedProb(token);
						//System.out.println("new prob :" + prob+ " "+token);
						if (prob<=0) {
							return token;
						}
				}
				}
				
			
		}
		else 
			{
			if ( b == true){
			HashMap<String, Double> linear = new HashMap<String, Double>();
			for(String T: m_reference.m_model.keySet()) {
				linear.put(T, calcLinearSmoothedProb(S+"_"+T));
			}
			linear = sortByValues(linear);
			int i =0;

			for (String T:linear.keySet()) {
				if (i<1 && !T.equals(S)) {
					i= i + 1;
					return T; 
				}
			}
		}
		else {
			HashMap<String, Double> absDisc = new HashMap<String, Double>();
			for(String T: m_reference.m_model.keySet()) {
				absDisc.put(T, calcAbsDiscountSmoothedProb(S+"_"+T));
			}
			absDisc = sortByValues(absDisc);
			int i =0;

			//System.out.println("Absolute Discount");
			for (String T:absDisc.keySet()) {
				if (i<1 && !T.equals(S)) {
					i= i + 1;
					return T;
				}
			}
		}
			}
		//System.out.println(m_N);
		return null; //How to deal with this special case?
	}
	
	
	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public double logLikelihood(HashMap<String, Token> stats, boolean isLinear) {
		double likelihood = 0;
		for(String token:stats.keySet()) {
			if (isLinear) {
				likelihood += Math.log(calcLinearSmoothedProb(token));
			}
			else {
				
				likelihood += Math.log(calcAbsDiscountSmoothedProb(token));
			}
		}
		double perplexity = 0.0;
		perplexity=Math.exp((-1*likelihood)/stats.keySet().size());		
		return perplexity;
	}
}
