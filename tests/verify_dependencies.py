#!/usr/bin/env python
"""
Dependency Verification Script for Medical Chatbot

This script verifies that all required dependencies can be imported
and that the system meets the minimum requirements.

Usage:
    python verify_dependencies.py

This script should be run after installing requirements.txt to ensure
all dependencies are properly installed and compatible.
"""

import sys
import importlib
import pkg_resources
from typing import List, Dict, Any


def check_python_version():
    """Check if Python version meets minimum requirements."""
    min_python = (3, 8)
    current_python = sys.version_info[:2]
    
    print(f"Python version: {sys.version}")
    
    if current_python >= min_python:
        print(f"✅ Python version {current_python} meets minimum requirement {min_python}")
        return True
    else:
        print(f"❌ Python version {current_python} does not meet minimum requirement {min_python}")
        return False


def verify_core_dependencies():
    """Verify core application dependencies can be imported."""
    core_deps = [
        'fastapi',
        'uvicorn', 
        'pydantic',
        'requests',
        'python_dotenv'
    ]
    
    print("\n🔍 Verifying Core Dependencies:")
    results = {}
    
    for dep in core_deps:
        try:
            # Handle special cases
            module_name = 'dotenv' if dep == 'python_dotenv' else dep
            importlib.import_module(module_name)
            print(f"✅ {dep}")
            results[dep] = True
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            results[dep] = False
    
    return results


def verify_ai_dependencies():
    """Verify AI/ML dependencies can be imported."""
    ai_deps = [
        'transformers',
        'sentence_transformers', 
        'torch',
        'numpy',
        'groq',
        'huggingface_hub'
    ]
    
    print("\n🤖 Verifying AI/ML Dependencies:")
    results = {}
    
    for dep in ai_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
            results[dep] = True
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            results[dep] = False
    
    return results


def verify_security_dependencies():
    """Verify security and encryption dependencies."""
    security_deps = [
        'cryptography',
        'passlib',
        'jwt',
        'google.auth'
    ]
    
    print("\n🔒 Verifying Security Dependencies:")
    results = {}
    
    for dep in security_deps:
        try:
            # Handle special cases
            if dep == 'jwt':
                import PyJWT as jwt
            else:
                importlib.import_module(dep)
            print(f"✅ {dep}")
            results[dep] = True
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            results[dep] = False
    
    return results


def verify_database_dependencies():
    """Verify database dependencies."""
    db_deps = [
        'sqlite3',  # Built-in
        'cryptography'
    ]
    
    print("\n🗄️ Verifying Database Dependencies:")
    results = {}
    
    for dep in db_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
            results[dep] = True
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            results[dep] = False
    
    return results


def verify_testing_dependencies():
    """Verify testing framework dependencies."""
    test_deps = [
        'pytest',
        'httpx',
        'faker'
    ]
    
    print("\n🧪 Verifying Testing Dependencies:")
    results = {}
    
    for dep in test_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
            results[dep] = True
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            results[dep] = False
    
    return results


def verify_optional_dependencies():
    """Verify optional dependencies."""
    optional_deps = [
        'PIL',  # Pillow
        'pytesseract',
        'pandas'
    ]
    
    print("\n📋 Verifying Optional Dependencies:")
    results = {}
    
    for dep in optional_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
            results[dep] = True
        except ImportError as e:
            print(f"⚠️  {dep}: {e} (optional)")
            results[dep] = False
    
    return results


def check_system_requirements():
    """Check system-level requirements."""
    print("\n🖥️ System Requirements:")
    
    # Check available memory (if psutil is available)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb >= 4:
            print("✅ RAM meets minimum requirement (4GB)")
        else:
            print("⚠️  RAM below recommended minimum (4GB)")
            
    except ImportError:
        print("⚠️  psutil not available - cannot check memory")
    
    # Check Python architecture
    architecture = sys.maxsize > 2**32 and "64-bit" or "32-bit"
    print(f"✅ Python architecture: {architecture}")


def generate_report(all_results: Dict[str, Dict[str, bool]]):
    """Generate a summary report."""
    print("\n" + "="*60)
    print("📊 DEPENDENCY VERIFICATION SUMMARY")
    print("="*60)
    
    total_deps = 0
    successful_deps = 0
    
    for category, results in all_results.items():
        if category == "python_version":
            continue
            
        category_success = sum(results.values())
        category_total = len(results)
        total_deps += category_total
        successful_deps += category_success
        
        status = "✅" if category_success == category_total else "⚠️ "
        print(f"{status} {category.replace('_', ' ').title()}: {category_success}/{category_total}")
    
    print("\n" + "-"*60)
    overall_percentage = (successful_deps / total_deps * 100) if total_deps > 0 else 0
    print(f"📈 Overall Success Rate: {successful_deps}/{total_deps} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 90:
        print("🎉 Excellent! Your system is ready for deployment.")
    elif overall_percentage >= 70:
        print("👍 Good! Most dependencies are available. Check optional ones if needed.")
    else:
        print("⚠️  Warning! Several dependencies are missing. Please install missing packages.")
    
    return overall_percentage >= 70


def main():
    """Main verification function."""
    print("🏥 Medical Chatbot - Dependency Verification")
    print("="*60)
    
    # Check Python version first
    python_ok = check_python_version()
    
    if not python_ok:
        print("\n❌ Python version too old. Please upgrade to Python 3.8 or higher.")
        sys.exit(1)
    
    # Run all verification checks
    all_results = {
        'python_version': python_ok,
        'core_dependencies': verify_core_dependencies(),
        'ai_dependencies': verify_ai_dependencies(),
        'security_dependencies': verify_security_dependencies(), 
        'database_dependencies': verify_database_dependencies(),
        'testing_dependencies': verify_testing_dependencies(),
        'optional_dependencies': verify_optional_dependencies()
    }
    
    # Check system requirements
    check_system_requirements()
    
    # Generate report
    success = generate_report(all_results)
    
    if success:
        print("\n✅ Verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Verification failed. Please install missing dependencies.")
        print("\nTo install all dependencies, run:")
        print("pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
