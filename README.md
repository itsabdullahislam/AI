# My Python App

This is a simple Python application built with:

- Python 3.11 [Install python](https://www.python.org/downloads/)
- Conda [How to download Ananaconda](https://www.anaconda.com/download)
- CUDA 11.8 [Download Cuda kit](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- Ollama [Download Ollama](https://github.com/ollama/ollama)

## Installation

To install the required dependencies, it's recommended to create a new Conda environment:
```
conda create -p venv python==3.11 -y
conda activate venv/
```
Install other dependencies
```
pip install -r requirements.txt
```

I have used a custom LLM from [Gemma-7b-it-GGUF](https://huggingface.co/mlabonne/gemma-7b-it-GGUF/resolve/main/gemma-7b-it.Q5_K_M.gguf)
```
Invoke-WebRequest -Uri "https://huggingface.co/mlabonne/gemma-7b-it-GGUF/resolve/main/gemma-7b-it.Q5_K_M.gguf" -OutFile "./model/llm/gemma-7b-it.Q5_K_M.gguf"
```

## Pre-Requisites 

1. Use the Modelfile.txt to add into Ollama list of models.
2. Run the following commands in the terminal.
   ```
   ollama create gemma-updated -f ./models/modelfile.txt
   ```

## Run Apps
1. For FastAPI
   ```
   python main.py
   ```
2. Using Chainlit
   
   1. For ConversationalChain
   ``` 
      chainlit run app.py
    ```
   2. For LCEL
    ``` 
      chainlit run test-app.py
      ```
