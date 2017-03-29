package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.ProbabilisticOptimizationProblem;

import java.util.Arrays;

/**
 * Copied from ContinuousPeaksTest
 *
 * @version 1.0
 */
public class FourPeaksTest {
    /**
     * The n value
     */
    private static final int N = 100;

    public static void main(String[] args) {


        for (int t = 10; t <= 26; t = t + 2) {
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(t);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);

            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            GATrainer gaTrainer = new GATrainer(200, 100, 10, gap, ef, t);
            gaTrainer.train();

            SATrainer saTrainer = new SATrainer(1E11, 0.6, hcp, ef, t);
            saTrainer.train();

            MIMICTrainer mTrainer = new MIMICTrainer(200, 5, pop, ef, t);
            mTrainer.train();
        }

    }
}
