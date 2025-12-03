#!/usr/bin/env python3
"""
LaTeX Compilation Script for Research Paper
Compiles the LSTM Trading AI research paper
"""

import os
import subprocess
import sys

def compile_latex():
    """Compile LaTeX document to PDF"""
    print("🔬 LSTM Trading AI Research Paper Compiler")
    print("=" * 50)

    # Check if LaTeX files exist
    tex_file = "research_paper.tex"
    bib_file = "references.bib"

    if not os.path.exists(tex_file):
        print(f"❌ Error: {tex_file} not found")
        return False

    if not os.path.exists(bib_file):
        print(f"❌ Error: {bib_file} not found")
        return False

    print("📄 Found LaTeX files:")
    print(f"   • {tex_file}")
    print(f"   • {bib_file}")

    try:
        # Check if pdflatex is available
        result = subprocess.run(["pdflatex", "--version"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Error: pdflatex not found. Please install LaTeX distribution:")
            print("   • Windows: MiKTeX or TeX Live")
            print("   • macOS: MacTeX")
            print("   • Linux: TeX Live")
            return False

        print("✅ LaTeX installation found")

        # Compile document (multiple passes for bibliography)
        print("\n🔨 Compiling LaTeX document...")

        # First pass
        print("   Pass 1/3: Initial compilation...")
        result1 = subprocess.run(["pdflatex", tex_file],
                               capture_output=True, text=True)
        if result1.returncode != 0:
            print(f"❌ Error in pass 1: {result1.stderr}")
            return False

        # Bibliography compilation
        print("   Pass 2/3: Bibliography processing...")
        result2 = subprocess.run(["bibtex", "research_paper"],
                               capture_output=True, text=True)
        if result2.returncode != 0:
            print(f"⚠️  Warning in bibliography: {result2.stderr}")

        # Second pass
        print("   Pass 3/3: Final compilation...")
        result3 = subprocess.run(["pdflatex", tex_file],
                               capture_output=True, text=True)
        if result3.returncode != 0:
            print(f"❌ Error in pass 3: {result3.stderr}")
            return False

        # Third pass (sometimes needed for references)
        result4 = subprocess.run(["pdflatex", tex_file],
                               capture_output=True, text=True)

        # Check if PDF was created
        pdf_file = "research_paper.pdf"
        if os.path.exists(pdf_file):
            file_size = os.path.getsize(pdf_file)
            print("\n✅ Compilation successful!")
            print(f"   📄 Generated: {pdf_file} ({file_size:,} bytes)")
            return True
        else:
            print("❌ Error: PDF file was not generated")
            return False

    except FileNotFoundError:
        print("❌ Error: LaTeX compiler not found in PATH")
        print("   Please ensure pdflatex and bibtex are installed and in your PATH")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def clean_auxiliary_files():
    """Clean up auxiliary LaTeX files"""
    auxiliary_extensions = ['.aux', '.bbl', '.blg', '.log', '.out', '.toc', '.lof', '.lot']

    cleaned = 0
    for ext in auxiliary_extensions:
        filename = f"research_paper{ext}"
        if os.path.exists(filename):
            os.remove(filename)
            cleaned += 1

    if cleaned > 0:
        print(f"🧹 Cleaned {cleaned} auxiliary files")
    else:
        print("ℹ️  No auxiliary files to clean")

def main():
    print("Compiling research paper...")

    success = compile_latex()

    if success:
        print("\n" + "="*50)
        print("🎉 SUCCESS!")
        print("Your research paper has been compiled successfully.")
        print("📄 File: research_paper.pdf")
        print("="*50)

        # Ask about cleaning auxiliary files
        response = input("\nClean auxiliary LaTeX files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            clean_auxiliary_files()

    else:
        print("\n" + "="*50)
        print("❌ COMPILATION FAILED")
        print("Please check the error messages above.")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()