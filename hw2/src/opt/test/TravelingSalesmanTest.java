package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        for (int n = 30; n < 100; n = n + 5) {

            Random random = new Random();
            // create the random points
            double[][] points = new double[N][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextInt();
                points[i][1] = random.nextInt();
            }
            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            GATrainer gaTrainer = new GATrainer(200, 100, 10, gap, ef, n);
//            gaTrainer.train();

            SATrainer saTrainer = new SATrainer(1E11, 0.6, hcp, ef, n);
            saTrainer.train();

//            MIMICTrainer mTrainer = new MIMICTrainer(200, 5, pop, ef, t);
//            mTrainer.train();

//        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
//        fit = new FixedIterationTrainer(sa, 20);
//        fit.train();
//        System.out.println(ef.value(sa.getOptimal()));
//
//        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
//        fit = new FixedIterationTrainer(ga, 20);
//        fit.train();
//        System.out.println(ef.value(ga.getOptimal()));

            // for mimic we use a sort encoding
//            ef = new TravelingSalesmanSortEvaluationFunction(points);
//            int[] ranges = new int[N];
//            Arrays.fill(ranges, N);
//            odd = new  DiscreteUniformDistribution(ranges);
//            Distribution df = new DiscreteDependencyTree(.1, ranges);
//            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
//
//            MIMIC mimic = new MIMIC(200, 100, pop);
//            fit = new FixedIterationTrainer(mimic, 20);
//            fit.train();
//            System.out.println(ef.value(mimic.getOptimal()));
        }

    }
}
