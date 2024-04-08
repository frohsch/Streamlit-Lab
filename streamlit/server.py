from fastapi import FastAPI
from fastapi.responses import JSONResponse

import pandas as pd
from endpoint_util import EndpointUtil

local_folder_path = '/data/'
endpoint_name = "{ENDPOINT_NAME}"
bucket_name = "{BUCKET_NAME}"
endpoint = EndpointUtil(bucket_name, endpoint_name, local_folder_path)

app = FastAPI()


@app.get("/")
async def hello():
    return {"hello": "world"}


@app.get("/{user_id}/{threshold}")
async def root(user_id: str, threshold: str):
    pred_df = endpoint.call(int(user_id), float(threshold))
    print(f'pred_df = {type(pred_df)}')

    # Convert the recommendations to a JSON-serializable format
    pred_df = pred_df.to_dict(orient='records')

    return JSONResponse(content=pred_df)


@app.get('/get_movie_info')
async def get_movie_info():
    movies_df = pd.read_csv('movies.csv')
    return JSONResponse(content=movies_df.to_dict(orient='records'))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("server:app", host='0.0.0.0', port=8000, workers=1)  # reload=False 권장
