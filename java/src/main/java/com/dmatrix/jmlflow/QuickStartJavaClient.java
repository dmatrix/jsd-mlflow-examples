import java.util.List;
import java.util.Optional;

import org.mlflow.api.proto.Service.*;
import org.mlflow.api.proto.Service.Experiment;
import org.mlflow.tracking.MlflowClient;

/**
 * This is an example application which uses the MLflow Tracking API to create and manage
 * experiments and runs.
 */
public class QuickStartJavaClient {

    public static void main(String[] args) throws Exception {
        (new QuickStartJavaClient()).process(args);
    }

    public void process(String[] args) throws Exception {
        String expName = null;
        MlflowClient client = null;

        if (args.length < 1) {
            client = new MlflowClient();
        } else if (args.length == 2) {
            client = new MlflowClient(args[0]);
            expName= args[1];
        } else {
            client = new MlflowClient(args[0]);
        }
        if (expName == null) expName = "Exp_" + System.currentTimeMillis();

        long expId;
        Optional<Experiment> myExp = client.getExperimentByName(expName);
        if (!myExp.isPresent()) {
            System.out.println("Experiment Name does not exsist: " + expName);
            System.out.println("======== Creating Experiment...");
            expId = client.createExperiment(expName);
        } else {
            System.out.println("Experiment Name already exsists: " + expName);
            System.out.println(myExp.get());
            expId = myExp.get().getExperimentId();
            System.out.println("Using expermenta id to create a run: " + expId);
        }
        System.out.println("====== createExperiment: " + expName);

        System.out.println("createExperiment: expId=" + expId);

        System.out.println("====== getExperiment");
        GetExperiment.Response exp = client.getExperiment(expId);
        System.out.println("getExperiment: " + exp);

        System.out.println("====== listExperiments");
        List<Experiment> exps = client.listExperiments();
        System.out.println("#experiments: " + exps.size());
        exps.forEach(e -> System.out.println("  Exp: " + e));

        createRun(client, expId);

        System.out.println("====== getExperiment again");
        GetExperiment.Response exp2 = client.getExperiment(expId);
        System.out.println("getExperiment: " + exp2);

        System.out.println("====== getExperiment by name");
        Optional<Experiment> exp3 = client.getExperimentByName(expName);
        System.out.println("getExperimentByName: " + exp3);
    }

    public void createRun(MlflowClient client, long expId) {
        System.out.println("====== createRun");

        // Create run
        String sourceFile = "QuickStartJavaClient.java";

        RunInfo runCreated = client.createRun(expId, sourceFile);
        System.out.println("CreateRun: " + runCreated);
        String runId = runCreated.getRunUuid();

        for (int i=0; i < 5; i++) {
            // Log parameters
            client.logParam(runId, "min_samples_leaf", Integer.toString(i+1));
            client.logParam(runId, "max_depth",  Integer.toString(i+3));
            client.logParam(runId, "num_iterations",  Integer.toString(10 + i));

            // Log metrics
            client.logMetric(runId, "auc", 2.12F + i);
            client.logMetric(runId, "accuracy_score", 3.12F + i);
            client.logMetric(runId, "zero_one_loss", 4.12F + i);
        }

        // Update finished run
        client.setTerminated(runId, RunStatus.FINISHED);

        // Get run details
        Run run = client.getRun(runId);
        System.out.println("GetRun: " + run);
    }
}