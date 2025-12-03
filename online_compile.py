#!/usr/bin/env python3
"""
Online LaTeX Compilation for Research Paper
Compiles the research paper using online LaTeX services
"""

import webbrowser
import os
import time

def open_online_compiler():
    """Open online LaTeX compilers in web browser"""
    print("🌐 ONLINE LATEX COMPILATION OPTIONS")
    print("=" * 50)

    compilers = [
        {
            "name": "Overleaf (Recommended)",
            "url": "https://www.overleaf.com/",
            "description": "Professional online LaTeX editor with real-time collaboration"
        },
        {
            "name": "LaTeX Base",
            "url": "https://latexbase.com/",
            "description": "Simple online LaTeX compiler"
        },
        {
            "name": " Papeeria",
            "url": "https://papeeria.com/",
            "description": "Online LaTeX editor with Git integration"
        }
    ]

    print("Available online compilers:")
    for i, compiler in enumerate(compilers, 1):
        print(f"{i}. {compiler['name']}")
        print(f"   {compiler['description']}")
        print(f"   URL: {compiler['url']}")
        print()

    choice = input("Choose a compiler (1-3) or 'all' to open all: ").strip().lower()

    if choice == 'all':
        print("Opening all compilers...")
        for compiler in compilers:
            webbrowser.open(compiler['url'])
            time.sleep(1)  # Small delay between openings
    elif choice in ['1', '2', '3']:
        compiler = compilers[int(choice) - 1]
        print(f"Opening {compiler['name']}...")
        webbrowser.open(compiler['url'])
    else:
        print("Invalid choice. Opening Overleaf by default...")
        webbrowser.open(compilers[0]['url'])

def create_upload_instructions():
    """Create instructions for uploading files to online compilers"""
    print("\n📤 UPLOAD INSTRUCTIONS")
    print("=" * 50)

    files_to_upload = [
        "research_paper.tex",
        "references.bib"
    ]

    print("Files to upload:")
    for file in files_to_upload:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - NOT FOUND")

    print("\nSteps:")
    print("1. Go to the online compiler website")
    print("2. Create a new project or upload files")
    print("3. Upload research_paper.tex as the main file")
    print("4. Upload references.bib for bibliography")
    print("5. Click 'Compile' or 'Recompile'")
    print("6. Download the generated PDF")

def create_manual_installation_guide():
    """Create manual installation guide"""
    print("\n🔧 MANUAL INSTALLATION GUIDE")
    print("=" * 50)

    print("If online compilation doesn't work, install LaTeX locally:")
    print()

    print("Option 1: MiKTeX (Recommended for Windows)")
    print("- Download: https://miktex.org/download")
    print("- Choose the 64-bit installer")
    print("- Follow the installation wizard")
    print("- Restart your terminal/command prompt")
    print()

    print("Option 2: TeX Live (Comprehensive)")
    print("- Download: https://www.tug.org/texlive/")
    print("- Choose the Windows installer")
    print("- Run the installer (large download ~4GB)")
    print()

    print("After installation, run:")
    print("python compile_paper.py")

def create_quick_test():
    """Create a quick LaTeX test"""
    print("\n🧪 QUICK LATEX TEST")
    print("=" * 50)

    test_tex = """\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\title{LSTM Trading AI Research Paper}
\\author{Jonus Nattapong}
\\begin{document}
\\maketitle
\\section{Abstract}
This is a test compilation of the LSTM Trading AI research paper.
\\end{document}"""

    with open("test_latex.tex", "w", encoding="utf-8") as f:
        f.write(test_tex)

    print("Created test_latex.tex for testing LaTeX installation")
    print("Try compiling this simple file first")

def main():
    print("LSTM Trading AI - Online LaTeX Compilation")
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check if research paper files exist
    required_files = ["research_paper.tex", "references.bib"]
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all research paper files are present.")
        return

    print("✅ All research paper files found")

    # Main options
    print("\nChoose an option:")
    print("1. Open online LaTeX compilers")
    print("2. Show upload instructions")
    print("3. Manual installation guide")
    print("4. Create quick test file")
    print("5. All of the above")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        open_online_compiler()
    elif choice == "2":
        create_upload_instructions()
    elif choice == "3":
        create_manual_installation_guide()
    elif choice == "4":
        create_quick_test()
    elif choice == "5":
        open_online_compiler()
        create_upload_instructions()
        create_manual_installation_guide()
        create_quick_test()
    else:
        print("Invalid choice. Showing all options...")
        create_upload_instructions()
        create_manual_installation_guide()
        create_quick_test()

    print("\n" + "=" * 60)
    print("💡 TIPS:")
    print("• Overleaf is the easiest option - no installation required")
    print("• Upload research_paper.tex and references.bib")
    print("• The paper should compile without errors")
    print("• Download the PDF when compilation is complete")
    print("=" * 60)

if __name__ == "__main__":
    main()