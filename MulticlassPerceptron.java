
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * University of Central Florida
 * CAP4630 Artifical Intelligence
 * Multi-Class Vector Perceptron Classifier Class
 * Author: Barath Tirumala
 */
public class MulticlassPerceptron implements weka.classifiers.Classifier {
    private Instances data;
    private String inputFileName;
    private int numTrainingEpochs;
    private int numWeightUpdates = 0;
    private int bias = 1;
    private double[][] weights;
    private boolean debug = false;

    public MulticlassPerceptron(String[] options) {
        inputFileName = options[0] ;
        numTrainingEpochs = Integer.parseInt(options[1]);
    }

    public void buildClassifier(Instances instances) throws Exception {
        printHeader();

        data = new Instances(instances);
        double correction;
        int predictedClass, correctClass, numAttributes = data.numAttributes();
        weights = new double[data.numClasses()][numAttributes];


        for(int i = 0; i < numTrainingEpochs; i++) {
            System.out.print("Epoch\t" + (i+1) + ": ");

            for(Instance inst : instances) {
                predictedClass = predict(inst);
                correctClass = (int) inst.classValue();

                if(debug) {
                    System.out.println("\n\tDEBUG\t predictedClass = " + predictedClass + ", correctClass = " + inst.classValue());
                }

                if (predictedClass != correctClass) {
                    System.out.print("0");

                    if(debug) {
                        System.out.println("\n\tDEBUG\t Wrong Prediction. Old Weights:");
                        System.out.print(appendWeights(new StringBuilder()).toString());
                    }

                    numWeightUpdates++;
                    for(int j = 0; j < numAttributes; j++) {
                        correction = (j == numAttributes - 1) ? bias : inst.value(j);
                        weights[predictedClass][j] -= correction;
                        weights[correctClass][j] += correction;
                    }

                    if(debug) {
                        System.out.println("\n\tDEBUG\t New Weights");
                        System.out.print(appendWeights(new StringBuilder()).toString());
                    }
                } else {
                    System.out.print("1");
                }
            }

            System.out.println();
        }
    }

    private int computeActivation(double[] w, Instance inst) {
        double sum = 0;

        if(debug) {
            System.out.print("W * F = ");
        }

        for ( int i = 0; i < w.length-1; i++) {
            sum += w[i] * inst.value(i);

            if(debug) {
                System.out.print(w[i] + " * " + inst.value(i));
                if(i < w.length-1) {
                    System.out.print(" + ");
                }
            }
        }

        sum += w[w.length-1] * bias;

        if(debug) {
            System.out.print(" = " + sum);
        }

        return sum < 0 ? -1 : 1;
    }

    private int predict(Instance inst) {
        int maxActivation = Integer.MIN_VALUE;
        int predictedClass = 0;
        int currentActivation;

        for(int i = 0; i < inst.numClasses(); i++) {
            if(debug)
                System.out.print("\n\tDEBUG\t Class " + i + ": ");

            currentActivation = computeActivation(weights[i], inst);

            if(debug) {
                System.out.print(" => Activation: " + currentActivation + ", maxActivation: " + maxActivation);
            }

            if (currentActivation > maxActivation) {
                maxActivation = currentActivation;
                predictedClass = i;

                if(debug)
                    System.out.print("\n\tDEBUG\t Updating Predicted Class to " + i);
            }
        }

        return predictedClass;
    }

    private StringBuilder appendWeights(StringBuilder sb) {
        for(int i = 0; i < weights.length; i++) {
            sb.append("Class ").append(i).append(" weights:\t");

            for(double weight: weights[i]) {
                sb.append(String.format(java.util.Locale.US,"%.3f", weight)).append(" ");
            }

            if (i < weights.length-1) {
                sb.append("\n");
            }
        }

        return sb;
    }

    private void printHeader() {
        System.out.println("University of Central Florida");
        System.out.println("CAP4630 Artifical Intelligence - Fall 2018");
        System.out.println("Multi-Class Perceptron by Barath Tirumala\n");
    }

    @Override
    public String toString() {
        String str = "Source File: " + inputFileName + "\nTraining epochs: " + numTrainingEpochs + "\nTotal # weight updates = " + numWeightUpdates + "\n\nFinal weights:\n\n";
        StringBuilder sb = new StringBuilder(str);
        sb = appendWeights(sb);
        return sb.toString();
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] result = new double[ data.numClasses() ];
        result[ predict(instance) ] = 1;
        return result;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }
}
