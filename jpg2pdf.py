import sys
import os
from fpdf import FPDF
import img2pdf


def process_jpg(image_bytes):
    pdf = FPDF()
    pdf.set_auto_page_break(0)
    pdf.add_page()
    pdf_bytes = img2pdf.convert(image_bytes)
    with open('output.pdf', 'wb') as f:
        f.write(pdf_bytes)
    return pdf

