package ocrreader;

	import java.util.Arrays;

	public class app {

		public static void main(String[] args) throws Exception {
			
			CharacterReader characterReader=new CharacterReader();
			characterReader.readImage();
		
		
	float[][] trainingData = new float[][] {
				 new float[] {0,0,1,1,1,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0},
				 new float[] {1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
				 new float[] {0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
				 new float[] {0,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1},
				 new float[] {0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,1,1,1,1,0},
				 new float[] {0,0,0,0,1,1,0,0,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0},
				 new float[] {1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0},
				 new float[] {0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,0,1,1,0,0,0,1,1,1,1,0,0,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,0,0,1,0,0,1,1,0},
				 new float[] {1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,1,1,0,1,0,0,0},
				 new float[] {0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0},
				 new float[] {0,1,1,1,0,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,1,1,1,1}
			};
			
			float[][] trainingResults = new float[][] {
				 new float[] {1,0,0,0,0,0,0,0,0,0}, // "0"
				 new float[] {1,0,0,0,0,0,0,0,0,0}, // "0"
				 new float[] {0,1,0,0,0,0,0,0,0,0},// "1"
				 new float[] {0,0,1,0,0,0,0,0,0,0},// "2"
				 new float[] {0,0,0,1,0,0,0,0,0,0},// "3"
				 new float[] {0,0,0,0,1,0,0,0,0,0},// "4"
				 new float[] {0,0,0,0,0,1,0,0,0,0},// "5"
				 new float[] {0,0,0,0,0,0,1,0,0,0},// "6"
				 new float[] {0,0,0,0,0,0,0,1,0,0},// "7"
				 new float[] {0,0,0,0,0,0,0,0,1,0},// "8"
				 new float[] {0,0,0,0,0,0,0,0,0,1}// "9"
			};
			
			BackpropNeuralNetwork backpropagationNeuralNetworks = new BackpropNeuralNetwork(64, 15, 10);
		
			for (int iterations = 0; iterations < NeuralNetConstants.ITERATIONS; iterations++) {
		
				for (int i = 0; i < trainingResults.length; i++) {
					backpropagationNeuralNetworks.train(trainingData[i], trainingResults[i], NeuralNetConstants.LEARNING_RATE, NeuralNetConstants.MOMENTUM);
				}
		
				if ((iterations + 1) % 100 == 0) {
					System.out.println();
					for (int i = 0; i < trainingResults.length; i++) {
						float[] data = trainingData[i];
						float[] calculatedOutput = backpropagationNeuralNetworks.run(data);
						System.out.println(calculatedOutput[0]+" "+calculatedOutput[1]+" "+calculatedOutput[2]+" "+calculatedOutput[3]+" "+calculatedOutput[4]+" "+calculatedOutput[5]+" "+calculatedOutput[6]+" "+calculatedOutput[7]+" "+calculatedOutput[8]+" "+calculatedOutput[9]);
					}
				}
			}		
			System.out.println("after training lets make predictions");
			System.out.println("---------------------------");
			
			System.out.println("---------------------------");
			
			calculatedOutput = backpropagationNeuralNetworks.run(new float[] {0,0,1,1,1,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0});
			System.out.println(Math.round(calculatedOutput[0])+" "+Math.round(calculatedOutput[1])+" "+Math.round(calculatedOutput[2])+" "+Math.round(calculatedOutput[3])+" "+Math.round(calculatedOutput[4])+" "+Math.round(calculatedOutput[5])+" "+Math.round(calculatedOutput[6])+" "+Math.round(calculatedOutput[7])+" "+Math.round(calculatedOutput[8])+" "+Math.round(calculatedOutput[9]));
			
			System.out.println("---------------------------");
			
			calculatedOutput = backpropagationNeuralNetworks.run(new float[] {0,1,1,1,0,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,1,1,1,1});
			System.out.println(Math.round(calculatedOutput[0])+" "+Math.round(calculatedOutput[1])+" "+Math.round(calculatedOutput[2])+" "+Math.round(calculatedOutput[3])+" "+Math.round(calculatedOutput[4])+" "+Math.round(calculatedOutput[5])+" "+Math.round(calculatedOutput[6])+" "+Math.round(calculatedOutput[7])+" "+Math.round(calculatedOutput[8])+" "+Math.round(calculatedOutput[9]));
			*/
		}
	}