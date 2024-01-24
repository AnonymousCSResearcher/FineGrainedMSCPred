from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import modules


# Create an instance of the FastAPI class
app = FastAPI()
# Mount the "static" directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def route_example():
    return '''
        <html>
        <head>
            <title>zbMATH Fine-Grained MSC Prediction API</title>
            <link rel="shortcut icon" type="image/png" href="/static/favicon_zbMATH.png"/>
        </head>
        <body>
            <h1>Fine-Grained MSC Prediction API</h1>
            <img src="/static/Logo_zbMATH-Open-201223.svg" alt="zbMATH" width="200">
            <h3>Example Queries</h3>
            <a href="/predict_text_keywords_qids?text=Summary:%20This%20paper%20presents%20a%20novel%20path%20integral%20formalism%20for%20Einstein%E2%80%99s%20theory%20of%20gravitation%20from%20%27%20\%27the%20viewpoint%20of%20optimal%20control%20theory.%20Despite%20its%20close%20connection%20to%20the%20well-known%20variational%20%27%20\%27principle%20of%20physicists,%20optimal%20control%20turns%20out%20to%20be%20more%20general.%20Within%20this%20context,%20a%20Lagrangian%20%27%20\%27which%20is%20different%20from%20the%20Einstein-Hilbert%20Lagrangian%20is%20defined.%20Einstein%E2%80%99s%20field%20equations%20are%20%27%20\%27recovered%20exactly%20with%20variations%20of%20the%20new%20action%20functional.%20The%20quantum%20theory%20is%20obtained%20using%20%27%20\%27Ashtekar%20variables%20and%20the%20loop%20scalar%20product.%20As%20an%20illustrative%20example,%20the%20tunneling%20process%20of%20a%20%27%20\%27black%20hole%20into%20another%20black%20hole%20or%20into%20a%20white%20hole%20is%20investigated%20with%20a%20toy%20model.">Text to Keywords</a>
            <a href="/predict_text_mscs?text=Summary:%20This%20paper%20presents%20a%20novel%20path%20integral%20formalism%20for%20Einstein%E2%80%99s%20theory%20of%20gravitation%20from%20%27%20\%27the%20viewpoint%20of%20optimal%20control%20theory.%20Despite%20its%20close%20connection%20to%20the%20well-known%20variational%20%27%20\%27principle%20of%20physicists,%20optimal%20control%20turns%20out%20to%20be%20more%20general.%20Within%20this%20context,%20a%20Lagrangian%20%27%20\%27which%20is%20different%20from%20the%20Einstein-Hilbert%20Lagrangian%20is%20defined.%20Einstein%E2%80%99s%20field%20equations%20are%20%27%20\%27recovered%20exactly%20with%20variations%20of%20the%20new%20action%20functional.%20The%20quantum%20theory%20is%20obtained%20using%20%27%20\%27Ashtekar%20variables%20and%20the%20loop%20scalar%20product.%20As%20an%20illustrative%20example,%20the%20tunneling%20process%20of%20a%20%27%20\%27black%20hole%20into%20another%20black%20hole%20or%20into%20a%20white%20hole%20is%20investigated%20with%20a%20toy%20model.">Text to MSCs</a>
            <a href="/predict_keyword_mscs?keywords=path%20integral%20formalism,variational%20principle,optimal%20control,process%20of,black%20hole,principle%20of,Einstein-Hilbert%20Lagrangian,Ashtekar%20variables,theory%20of%20gravitation,tunneling%20process,white%20hole,path%20integral,quantum%20theory,field%20equations,integral%20formalism,theory%20of">Keywords to MSCs</a>
        </body>
        </html>
        '''

# Predict keywords and Wikidata QIDs from (abstract) text
@app.get("/predict_text_keywords_qids")
def route_predict_text_keywords_qids(text: str):
    keywords, qids = modules.extract_keywords(text)
    return {"keywords": keywords, "qids": qids}


# Predict MSCs from (abstract) text
@app.get("/predict_text_mscs")
def route_predict_text_mscs(text: str):
    keywords, _ = modules.extract_keywords(text)
    mscs = modules.predict_mscs(keywords)
    return {"mscs": mscs}


# Predict MSCs from keywords
@app.get("/predict_keyword_mscs")
def route_predict_keyword_mscs(keywords: str = Query(..., title="keywords")):
    # Parse the comma-separated string into a list
    keywords = [kw.strip() for kw in keywords.split(",")]
    mscs = modules.predict_mscs(keywords)
    return {"mscs": mscs}


# Run the application using the FastAPI's included "uvicorn.run" function
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
