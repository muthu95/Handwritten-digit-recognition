import java.util.*;
import java.io.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
        ArrayList<Double> scores = computeNNout(instance);
        int i, maxIdx = -1;
        for(i=0; i<scores.size(); i++) {
            if(maxIdx == -1 || scores.get(i) > scores.get(maxIdx))
                maxIdx = i;
        }
        return maxIdx;
    }


    private ArrayList<Double> computeNNout(Instance instance) {
        int i;
        for(i=0; i<instance.attributes.size(); i++) {
            inputNodes.get(i).setInput(instance.attributes.get(i));
        }

        for(i=0; i<hiddenNodes.size(); i++) {
            hiddenNodes.get(i).calculateOutput();
        }

        double expSum = 0;
        ArrayList<Double> scores = new ArrayList<Double>();
        for(i=0; i<outputNodes.size(); i++) {
            outputNodes.get(i).calculateOutput();
            expSum += outputNodes.get(i).getOutput();
        }
        for(i=0; i<outputNodes.size(); i++) {
            outputNodes.get(i).normalizeOutput(expSum);
            scores.add(outputNodes.get(i).getOutput());
        }

        return scores;
    }

    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
        //System.out.println("Training");
        int i, j;
        for(i=0; i<maxEpoch; i++) {
            //System.out.println("Epoch: " + i);
            Collections.shuffle(trainingSet, random);

            for(Instance instance : trainingSet) {
                //System.out.println("New Instance");
                ArrayList<Double> scores = computeNNout(instance);
                ArrayList<Integer> target = instance.classValues;
                double [] hiddenNodeHelpers = new double[hiddenNodes.size()];

                //Calculating deltas
                for(j=0; j<scores.size(); j++) {
                    outputNodes.get(j).calculateDelta(target.get(j) - scores.get(j));
                    int k;
                    for(k=0; k<outputNodes.get(j).parents.size(); k++) {
                        NodeWeightPair parent = outputNodes.get(j).parents.get(k);
                        double d = outputNodes.get(j).getDelta();
                        hiddenNodeHelpers[k] += (parent.weight * d);
                    }
                }
                for(j=0; j<hiddenNodes.size(); j++) {
                    hiddenNodes.get(j).calculateDelta(hiddenNodeHelpers[j]);
                }

                //Updating weights
                for(j=0; j<hiddenNodes.size(); j++) {
                    hiddenNodes.get(j).updateWeight(learningRate);
                }
                for(j=0; j<outputNodes.size(); j++) {
                    outputNodes.get(j).updateWeight(learningRate);
                }
            }

            double totalLoss = 0;
            for(Instance instance: trainingSet) {
                totalLoss += loss(instance);
            }
            System.out.printf("Epoch: %d, Loss: %.3e\n", i, totalLoss/trainingSet.size());
            //printWeights(i);
        }
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        ArrayList<Double> scores = computeNNout(instance);
        ArrayList<Integer> target = instance.classValues;
        int i;
        double loss = 0;
        for(i=0; i<scores.size(); i++) {
            double actual = target.get(i);
            double computed = scores.get(i);
            loss += (actual * Math.log(computed));
        }
        return -loss;
    }

    public void printWeights(int epoch) {
    	PrintWriter out = null;
    	try {
    		out = new PrintWriter(new FileOutputStream("myWeights.txt", true));
    		out.print("EPOCH: " + epoch + "\n");
    		out.print("**************************\n");
    		out.print("Updated Weights between the Hidden and Input Layers after Epoch " + epoch + "\n");
    		out.print("-------------------------------------------------------------------------------------------\n");
    		for (Node hidden : hiddenNodes) {
    			if (hidden.parents != null) {
    				for (NodeWeightPair nwp : hidden.parents) {
    					out.print(nwp.weight + "\n");
    				}
    			}
    		}
    		out.print("\nUpdated Weights between the Output and Hidden Layers after Epoch " + epoch	+ "\n");
    		out.print("-------------------------------------------------------------------------------------------\n");
    		for (Node output : outputNodes) {
    			for (NodeWeightPair nwp : output.parents) {
    				out.print(nwp.weight + "\n");
    			}
    		}
    		out.println();
    	} catch (FileNotFoundException e) {
    		System.out.println("no file found for output");
    	}
    	if (out != null)
    		out.close();
    }
}
