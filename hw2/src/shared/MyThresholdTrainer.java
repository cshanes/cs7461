package shared;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;

/**
 * A threshold trainer trains a network
 * until the error goes below a threshold, using another trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class MyThresholdTrainer /*implements Trainer*/ {
    /** The default threshold */
    private static final double THRESHOLD = 1E-6;
    /** The maxium number of iterations */
    private static final int MAX_ITERATIONS = 500;

    /**
     * The trainer
     */
    private OptimizationAlgorithm trainer;

    /**
     * The threshold
     */
    private double threshold;

    /**
     * The number of iterations trained
     */
    private int iterations;

    /**
     * The maximum number of iterations to use
     */
    private int maxIterations;

    /**
     * Create a new convergence trainer
     * @param trainer the thrainer to use
     * @param threshold the error threshold
     * @param maxIterations the maximum iterations
     */
    public MyThresholdTrainer(OptimizationAlgorithm trainer,
                            double threshold, int maxIterations) {
        this.trainer = trainer;
        this.threshold = threshold;
        this.maxIterations = maxIterations;
    }


    /**
     * Create a new convergence trainer
     * @param trainer the trainer to use
     */
    public MyThresholdTrainer(OptimizationAlgorithm trainer) {
        this(trainer, THRESHOLD, MAX_ITERATIONS);
    }

    /**
     * @see Trainer#train()
     */
    public double train(double globalOptima, EvaluationFunction ef) {
        double error = Double.MAX_VALUE;
        do {
            iterations++;
            trainer.train();
            error = ef.value(trainer.getOptimal()) - globalOptima;
//            System.out.println("error: " + error);
        } while (Math.abs(error) > threshold
                && iterations < maxIterations);
        return error;
    }

    /**
     * Get the number of iterations used
     * @return the number of iterations
     */
    public int getIterations() {
        return iterations;
    }


}
