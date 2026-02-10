from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
from psa_agent import solve_problem

app = FastAPI(title="Problem Solving Agent", description="Web interface for the PSA agent")

class ProblemRequest(BaseModel):
    problem: str

@app.post("/api/solve")
async def solve_problem_endpoint(request: ProblemRequest):
    """Solve a problem using the PSA agent."""
    try:
        # Run the agent in a thread pool since it might be blocking
        result = await asyncio.get_event_loop().run_in_executor(None, solve_problem, request.problem)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving problem: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the main webpage."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Problem Solving Agent</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #555;
                font-size: 1.1em;
            }
            textarea {
                width: 100%;
                min-height: 120px;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                resize: vertical;
                font-family: inherit;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s, box-shadow 0.2s;
                font-weight: 600;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                display: none;
            }
            .result h3 {
                margin-top: 0;
                color: #333;
                font-size: 1.3em;
            }
            .result pre {
                background: #2d3748;
                color: #e2e8f0;
                padding: 15px;
                border-radius: 6px;
                overflow-x: auto;
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.4;
            }
            .error {
                background: #fed7d7;
                border-left-color: #e53e3e;
                color: #c53030;
            }
            .examples {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .examples h3 {
                margin-top: 0;
                color: #333;
            }
            .example-btn {
                background: #e2e8f0;
                color: #4a5568;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                margin: 5px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.2s;
            }
            .example-btn:hover {
                background: #cbd5e0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Problem Solving Agent</h1>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                Enter a programming problem and watch the AI generate, test, and analyze the solution!
            </p>

            <form id="problemForm">
                <div class="form-group">
                    <label for="problem">Describe your programming problem:</label>
                    <textarea
                        id="problem"
                        name="problem"
                        placeholder="e.g., Write a function to find the maximum number in a list..."
                        required
                    ></textarea>
                </div>
                <button type="submit" id="submitBtn">Solve Problem</button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Generating solution... This may take a moment.</p>
            </div>

            <div class="result" id="result"></div>

            <div class="examples">
                <h3>Try these examples:</h3>
                <button class="example-btn" onclick="setExample('Write a function to check if a string is a palindrome')">Palindrome Checker</button>
                <button class="example-btn" onclick="setExample('Create a function that calculates the factorial of a number')">Factorial Function</button>
                <button class="example-btn" onclick="setExample('Implement a simple calculator that can add, subtract, multiply, and divide')">Simple Calculator</button>
                <button class="example-btn" onclick="setExample('Write a function to find the largest number in a list')">Find Maximum</button>
                <button class="example-btn" onclick="setExample('Create a function to reverse the words in a sentence')">Reverse Words</button>
            </div>
        </div>

        <script>
            const form = document.getElementById('problemForm');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const submitBtn = document.getElementById('submitBtn');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                const problem = document.getElementById('problem').value.trim();
                if (!problem) return;

                // Show loading state
                loading.style.display = 'block';
                result.style.display = 'none';
                submitBtn.disabled = true;
                submitBtn.textContent = 'Solving...';

                try {
                    const response = await fetch('/api/solve', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ problem }),
                    });

                    const data = await response.json();

                    if (response.ok) {
                        displayResult(data.result, false);
                    } else {
                        displayResult(data.detail || 'An error occurred', true);
                    }
                } catch (error) {
                    displayResult('Network error: ' + error.message, true);
                } finally {
                    // Hide loading state
                    loading.style.display = 'none';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Solve Problem';
                }
            });

            function displayResult(content, isError) {
                result.style.display = 'block';
                result.className = 'result' + (isError ? ' error' : '');
                result.innerHTML = `
                    <h3>${isError ? 'Error' : 'Solution'}</h3>
                    <pre>${content.replace(/</g, '<').replace(/>/g, '>')}</pre>
                `;
                result.scrollIntoView({ behavior: 'smooth' });
            }

            function setExample(example) {
                document.getElementById('problem').value = example;
            }

            // Add some visual feedback
            document.getElementById('problem').addEventListener('input', function() {
                this.style.borderColor = this.value.trim() ? '#667eea' : '#ddd';
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    print("Starting PSA Web App...")
    print("Visit http://localhost:8000 to use the agent")
    uvicorn.run(app, host="0.0.0.0", port=8000)
