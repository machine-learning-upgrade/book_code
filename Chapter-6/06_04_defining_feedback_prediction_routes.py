import comet_llm
from comet_llm.query_dsl import TraceMetadata
from fastapi import HTTPException
from pydantic import BaseModel
from uuid import uuid4
from openai import OpenAI

client = OpenAI(
    api_key="YOUR-API-KEY",
    # base_url=”YOUR-API-URL”
    # Uncomment the above line to use your own model deployed with vLLM
)

async def log_prompt(data: dict):
    comet_llm.log_prompt(
        prompt=data['prompt'],
        output=data['output'],
        metadata={'id': data['id']}
    )

# Define the request model for the prediction route
class PredictionRequest(BaseModel):
    question: str

# Define the request model for the feedback route
class FeedbackRequest(BaseModel):
    conversation_id: str
    score: float

@app.post("/prediction/")
async def prediction(background_tasks:BackgroundTasks, request: PredictionRequest):
    try:
        # Generate a unique identifier for the conversation
        conversation_id = str(uuid4 ())
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request.question}],
        )
        answer = completion.choices[0].message.content
        log = {
            "prompt": request.question,
            "output": answer,
            "id": conversation_id
        }
        response = {"answer": answer, "conversation_id": conversation_id}
        background_tasks.add_task(log_prompt, data=log)
        return response
    except Exception as e:
        raise HTTPException (status_code=400, detail=str(e))


@app.post("/feedback")
async def feedback(background_tasks: BackgroundTasks,request: FeedbackRequest):
    try:
        api = comet_llm.API()

        # An LLMTrace is the object Comet uses to represent the query/response
        trace = api.query(
            workspace="YOUR-WORKSPACE",
            project_name="YOUR-PROJECT",
            query=(TraceMetadata("id") == request.conversation_id)
        )
        trace.log_user_feedback(request.score)
        return {"status": "Success"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))