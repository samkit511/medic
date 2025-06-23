@echo off
setlocal EnableDelayedExpansion

:: Prompt user to check if dependencies are downloaded
set "dependencies_downloaded="
set /p dependencies_downloaded="Have you already downloaded the dependencies? (yes/no): "

:: Check user response
if /i "!dependencies_downloaded!"=="no" (
    echo Installing dependencies...
    
    :: Install dependencies for Backend
    echo Installing Backend dependencies...
    cd /d C:\Users\samkit jain\Dropbox\PC\Desktop\research\chatbot\clinical-diagnosis-based-chatbots-using-few--shot-prompting-of-large-language-models-main\Backend
    pip install fastapi uvicorn pydantic pydantic-settings groq huggingface_hub PyPDF2 requests
    
    :: Install dependencies for Frontend
    echo Installing Frontend dependencies...
    cd /d C:\Users\samkit jain\Dropbox\PC\Desktop\research\chatbot\clinical-diagnosis-based-chatbots-using-few--shot-prompting-of-large-language-models-main\Frontend
    pip install streamlit requests
    
    echo Dependencies installation complete.
) else if /i "!dependencies_downloaded!"=="yes" (
    echo Skipping dependency installation as requested.
) else (
    echo Invalid response. Please enter 'yes' or 'no'. Exiting...
    pause
    exit /b
)

:: Start Backend (main.py) in a new command prompt window
start cmd /k "cd /d C:\Users\samkit jain\Dropbox\PC\Desktop\research\chatbot\clinical-diagnosis-based-chatbots-using-few--shot-prompting-of-large-language-models-main\Backend && python main.py"

:: Start Frontend (appp.py) in a new command prompt window
start cmd /k "cd /d C:\Users\samkit jain\Dropbox\PC\Desktop\research\chatbot\clinical-diagnosis-based-chatbots-using-few--shot-prompting-of-large-language-models-main\Frontend && streamlit run appp.py"

:: Pause to keep the script window open (optional)
pause