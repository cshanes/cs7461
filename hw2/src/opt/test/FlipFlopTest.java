package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
    /** The n value */
    private static final int N = 80;
    
    public static void main(String[] args) {

        for(int n = 50; n < 200; n = n + 10) {
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FlipFlopEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            GATrainer gaTrainer = new GATrainer(1000, 500, 200, gap, ef, n);
//            gaTrainer.train();

            SATrainer saTrainer = new SATrainer(100, 0.6, hcp, ef, n);
//            saTrainer.train();

            MIMICTrainer mTrainer = new MIMICTrainer(200, 5, pop, ef, n);
            mTrainer.train();

//            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(1000, 500, 200, gap);
//            fit = new FixedIterationTrainer(ga, 1000);
//            fit.train();
//            System.out.println(ef.value(ga.getOptimal()));
//
//
//            MIMIC mimic = new MIMIC(200, 5, pop);
//            fit = new FixedIterationTrainer(mimic, 1000);
//            fit.train();
//            System.out.println(ef.value(mimic.getOptimal()));
        }

    }
}
