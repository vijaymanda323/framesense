import asyncio
from main import app, lifespan

async def test_startup():
    print("Testing lifespan startup...")
    try:
        async with lifespan(app):
            print("Successfully loaded YOLO, Depth, and RAG!")
            rag = app.state.rag
            if rag.llm:
                print("RAG was initialized with LLM.")
            else:
                print("RAG disabled (missing GEMINI_API_KEY).")
    except Exception as e:
        print(f"Error during startup: {e}")

if __name__ == "__main__":
    asyncio.run(test_startup())
