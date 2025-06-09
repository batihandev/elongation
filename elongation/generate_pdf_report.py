from fpdf import FPDF
import os

def safe_text(text):
    return text.encode("latin-1", "replace").decode("latin-1")

def generate_pdf_report(test_results, output_path="elongation_summary_report.pdf"):
    """
    test_results: List of dicts, each with keys:
        - name: str
        - yield_strength: int (MPa)
        - rebar_diameter: float (mm)
        - elongation_plot: str (path to .png)
        - force_plot: str (path to .png)
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for result in test_results:
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, safe_text(f"Test Report: {result['name']}"), ln=True)

        # Metadata
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, safe_text(f"Yield Strength: {result['yield_strength']} MPa"), ln=True)
        pdf.cell(0, 10, safe_text(f"Rebar Diameter: {result['rebar_diameter']} mm"), ln=True)
        pdf.ln(5)

        # Elongation Plot
        if os.path.exists(result["elongation_plot"]):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, safe_text("Elongation Curve"), ln=True)
            pdf.image(result["elongation_plot"], w=180)
            pdf.ln(5)

        # Force-Elongation Plot
        if os.path.exists(result["force_plot"]):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, safe_text("Force-Elongation Curve"), ln=True)
            pdf.image(result["force_plot"], w=180)

    pdf.output(output_path)
    print(f"✅ Report saved to: {output_path}")
