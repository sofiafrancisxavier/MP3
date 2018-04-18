package structures;

import java.util.Comparator;
import java.util.Map.Entry;

public class StringHashDoubleComparator implements Comparator<Entry<String, Double>>{

	@Override
	public int compare(Entry<String, Double> arg0, Entry<String, Double> arg1) {
		double dif = arg1.getValue() - arg0.getValue();;
		if (dif > 0) {
			return 1;
		}
		else if (dif == 0) {
			return 0;
		}
		else {
			return -1;
		}
	}

}
