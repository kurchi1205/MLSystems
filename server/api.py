import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import sys
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from serve import serve_step
from transformers import AutoTokenizer, LlamaForCausalLM
import uvicorn
import argparse


class Request(BaseModel):
    prompt: str


class RequestStatus:
    PENDING = "pending"
    COMPLETED = "completed"
    ERROR = "error"
    QUOTA_EXCEEDED = "quota_exceeded"


class GenerateResponse(BaseModel):
    text: Optional[str] = None
    status: str = None


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class SeqState:
    def __init__(self, prompt, request_id, embedding_only: bool = False):
        self.input_ids = None
        self.past_key_values = None
        self.decoded_tokens = ""
        self.has_prefilled = False
        self.generated_tokens = 0
        self.prompt = prompt
        self.status = RequestStatus.PENDING
        self.request_id = request_id
        self.embedding_only = embedding_only


class RequestPool:
    def __init__(self, init_quota: int = 1000, max_generated_tokens: int = 50):
        self.requests: Dict[str, SeqState] = {}
        self.active_requests: Dict[str, SeqState] = {}
        self.max_active_requests = 2
        self.quota = init_quota
        self.max_generated_tokens = max_generated_tokens
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def add_request(self, request: Request, embedding_only: bool = False) -> str:
        request_id = str(uuid.uuid4())
        request_data = SeqState(request.prompt, request_id, embedding_only)

        async with self.lock:
            self.requests[request_id] = request_data
            await self.queue.put(request_id)

        return request_id

    async def wait_for_completion(self, request_id: str, interval: float = 0.1) -> Dict:
        """wait for request completion"""
        while True:
            # TODO: wait for completion
            # return until the request is completed
            # for embedding, you should return EmbeddingResponse(embedding=seq.embedding)
            # for generate, you should return GenerateResponse(text=seq.decoded_tokens, status=seq.status)
            # ==== start your code here ====

            pass
            # ==== end of your code ====

    def stop_generation(self, tokenizer):
        # TODO: stop generation
        # stop generation if:
        # the generated tokens exceed the self.max_generated_tokens
        # or the last token is eos_token
        # or the sequence is an embedding only sequence (seq.embedding_only == True)
        # ==== start your code here ====

        stop_list = []
        # ==== end of your code ====
        return stop_list

    @torch.no_grad()
    async def process_request(self, model, tokenizer):
        while True:
            # TODO: get pending requests
            # if active requests are less than max_active_requests,
            # pop requests from the queue (if any) and put it into active requests
            # ==== start your code here ====


            # ==== end of your code ====

            if self.quota <= 0:
                # TODO: stop all requests if quota is exceeded
                # pop it from the active requests and requests
                # also set the status of the requests to RequestStatus.QUOTA_EXCEEDED
                # ==== start your code here ====

                pass
                # ==== end of your code ====

            if len(self.active_requests) > 0:
                # serve step
                request_data = list(self.active_requests.values())
                consumed_tokens = serve_step(model, tokenizer, request_data)
                # stop generation
                stop_list = self.stop_generation(tokenizer)
                for seq_state in stop_list:
                    seq_state.status = RequestStatus.COMPLETED
                # update quota
                self.quota -= consumed_tokens
                # clean up completed requests
                for req in request_data:
                    if req.status == RequestStatus.COMPLETED:
                        self.active_requests.pop(req.request_id)

            await asyncio.sleep(0.01)  # avoid high CPU usage


def parse_arguments():
    parser = argparse.ArgumentParser(description='API server with configurable init quota')
    parser.add_argument('--init-quota', 
                       type=int, 
                       default=sys.maxsize,
                       help='Initial quota value (default: infinite)')
    parser.add_argument('--max-generated-tokens', 
                       type=int, 
                       default=sys.maxsize,
                       help='Max generated tokens (default: infinite)')
    return parser.parse_args()


# Get command line arguments
args = parse_arguments()

# initialize request pool
request_pool = RequestPool(args.init_quota, args.max_generated_tokens)


# model
# NOTE: you need to apply for access to the model at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# which will be normally approved within several hours
# you can also use Qwen/Qwen2.5-1.5B-Instruct which does not require access
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(request_pool.process_request(model, tokenizer))
    yield


app = FastAPI(lifespan=lifespan)


# API endpoints
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: Request):
    request_id = await request_pool.add_request(request)
    return await request_pool.wait_for_completion(request_id)


@app.post("/get_embedding", response_model=EmbeddingResponse)
async def get_embedding(request: Request):
    request_id = await request_pool.add_request(request, embedding_only=True)
    return await request_pool.wait_for_completion(request_id)


if __name__ == "__main__":
    uvicorn.run("api:app", reload=True)
