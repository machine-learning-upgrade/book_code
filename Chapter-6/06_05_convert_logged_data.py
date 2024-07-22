import comet_llm
from comet_llm.query_dsl import TraceMetadata, Duration, Timestamp, UserFeedback
import json

def llm_data_to_artifact(start_time, end_time, artifact_name):
    comet_llm.init()
    api = comet_llm.API()
    traces = api.query(
        workspace="YOUR-WORKSPACE",
        project_name="YOUR-PROJECT",
        query=((Timestamp() > start_time) & (Timestamp() < end_time))
    )

    json_blob = []
    # api.query() returns a list of LLMTraces that match our query parameter
    for trace in traces:
        trace_data = trace._get_trace_data()
        data = {
            "inputs": trace_data['chain_inputs'],
            "outputs": trace_data['chain_outputs'],
            "metadata": trace_data['metadata']
        }
        json_blob.append(data)
        file_name = f"{start_time}-{end_time}.json"

        with open(file_name, 'w+') as f:
            json.dump(json_blob, f)

        experiment = comet_ml.Experiment()
        
        # Try to access existing artifact, if we've previously created one
        try:
            artifact = experiment.get_artifact(artifact_name)
        except:
            artifact = comet_ml.Artifact(artifact_name)

        artifact.add(file_name)
        experiment.log_artifact(artifact)