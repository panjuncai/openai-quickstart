from fpdf import FPDF;pdf = FPDF();pdf.add_page();pdf.set_font("Arial", size=12);text = open("test.txt").read();[pdf.cell(0, 10, txt=line, ln=True) for line in text.split("
")];pdf.output("test.pdf")
