from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import modules

class TextRequest(BaseModel):
	text: str
	preproc: str = None

# Create an instance of the FastAPI class
app = FastAPI()
# Mount the "static" directory to serve static files
try:
	# Docker container
	app.mount("/static", StaticFiles(directory="static"), name="static")
except:
	# Local testing
	root_path = Path(__file__).resolve().parent.parent
	# Mount the root directory to serve static files
	app.mount("/static", StaticFiles(directory=root_path), name="static")

@app.get("/", response_class=HTMLResponse)
def route_example():

	text = """
		Summary:%20This%20paper%20presents%20a%20novel%20path%20integral%20formalism%20for%20Einstein%E2%80%99s%20theory%20of%20gravitation%20from%20the%20viewpoint%20of%20optimal%20control%20theory.%20Despite%20its%20close%20connection%20to%20the%20well-known%20variational%20principle%20of%20physicists,%20optimal%20control%20turns%20out%20to%20be%20more%20general.%20Within%20this%20context,%20a%20Lagrangian%20which%20is%20different%20from%20the%20Einstein-Hilbert%20Lagrangian%20is%20defined.%20Einstein%E2%80%99s%20field%20equations%20are%20recovered%20exactly%20with%20variations%20of%20the%20new%20action%20functional.%20The%20quantum%20theory%20is%20obtained%20using%20Ashtekar%20variables%20and%20the%20loop%20scalar%20product.%20As%20an%20illustrative%20example,%20the%20tunneling%20process%20of%20a%20black%20hole%20into%20another%20black%20hole%20or%20into%20a%20white%20hole%20is%20investigated%20with%20a%20toy%20model.
	"""

	return f'''
		<html>
		<head>
			<title>zbMATH Fine-Grained MSC Prediction API</title>
			<link rel="shortcut icon" type="image/png" href="/static/favicon_zbMATH.png"/>
		</head>
		<body>
			<h1>Fine-Grained MSC Prediction API</h1>
			<img src="/static/Logo_zbMATH-Open-201223.svg" alt="zbMATH" width="200">
			<h3>Example Queries</h3>
			<a href="/predict_text_keywords_qids?text={text}">Text to Keywords</a>
			<a href="/predict_text_keywords_augmented?text={text}&preproc=none">Text to Wikipedia*</a>
			<a href="/predict_linked_text_keywords?text={text}&preproc=none">Text with Wikipedia Links*</a>
			<a href="/predict_text_mscs?text={text}">Text to MSCs</a>
			<a href="/predict_keyword_mscs?keywords=path integral formalism,variational principle,optimal control,process of,black hole,principle of,Einstein-Hilbert Lagrangian,Ashtekar variables,theory of gravitation,tunneling process,white hole,path integral,quantum theory,field equations,integral formalism,theory of">Keywords to MSCs</a>
			<p>*Use parameter 'preproc' in ['none', 'stem', 'lemma']</p>
		</body>
		</html>
	'''


# KEYWORD EXTRACTION


# Predict keywords and Wikidata QIDs from (abstract) text
@app.get("/predict_text_keywords_qids")
async def route_predict_text_keywords_qids(text: str):
	keywords, qids = modules.extract_keywords(text)
	keyword_qid = [{"entity": keyword_qid[0], "qid": keyword_qid[1]} for keyword_qid in zip(keywords, qids)]
	return {"keyword_qid": keyword_qid} # {"keywords": keywords, "qids": qids}


# Predict keywords from (abstract) text augmented with spans, Wikidata QIDs and Wikipedia URLs
@app.get("/predict_text_keywords_augmented")
async def get_predict_text_keywords_augmented(text: str, preproc: str = None):
	keywords_augmented = modules.extract_keywords_augmented(text, preproc)
	return {'keyword_span_qid_url': keywords_augmented}


@app.post("/predict_text_keywords_augmented")
async def post_predict_text_keywords_augmented(request: TextRequest):
	keywords_augmented = modules.extract_keywords_augmented(request.text, request.preproc)
	return {'keyword_span_qid_url': keywords_augmented}


# Predict keywords from (abstract) text and return a linked html text
@app.get("/predict_linked_text_keywords", response_class=HTMLResponse)
async def route_predict_linked_text_keywords(text: str, preproc: str = None):
	linked_text_html = modules.get_linked_text_html(text, preproc)
	linked_text_full_html = f'''<html><body>{linked_text_html}</body></html>'''
	#return {'linked_text_html': linked_text_html}
	return linked_text_full_html


# MSC PREDICTION


# Predict MSCs from (abstract) text
@app.get("/predict_text_mscs")
async def route_predict_text_mscs(text: str):
	keywords, _ = modules.extract_keywords(text)
	mscs = modules.predict_mscs(keywords)
	return {"mscs": mscs}


# Predict MSCs from keywords
@app.get("/predict_keyword_mscs")
async def route_predict_keyword_mscs(keywords: str = Query(..., title="keywords")):
	# Parse the comma-separated string into a list
	keywords = [kw.strip() for kw in keywords.split(",")]
	mscs = modules.predict_mscs(keywords)
	return {"mscs": mscs}


# TODO: Predict keywords from MSCs


# Run the application using the FastAPI's included "uvicorn.run" function
if __name__ == "__main__":
	uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
