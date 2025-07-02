@echo off
setlocal EnableDelayedExpansion

:: Prompt user for project path, backend path, and frontend path
set /p project_path="Enter the full project path (e.g., C:\Users\YourName\Projects\medic): "
set /p backend_path="Enter the full backend path (e.g., %project_path%\Backend): "
set /p frontend_path="Enter the full frontend path (e.g., %project_path%\Frontend): "

:: Prompt user to check if dependencies are downloaded
set "dependencies_downloaded="
set /p dependencies_downloaded="Have you already downloaded the dependencies? (yes/no): "

:: Check user response
if /i "!dependencies_downloaded!"=="no" (
    echo Installing dependencies...
    
    :: Install dependencies for Backend
    echo Installing Backend dependencies...
    cd /d "!backend_path!"
    pip install fastapi uvicorn pydantic pydantic-settings groq huggingface_hub PyPDF2 requests
    
    :: Install dependencies for Frontend
    echo Installing Frontend dependencies...
    cd /d "!frontend_path!"
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
start cmd /k "cd /d !backend_path! && python main.py"

:: Start Frontend (appp.py) in a new command prompt window
start cmd /k "cd /d !frontend_path! && streamlit run main.py"

:: Pause to keep the script window open (optional)
pause