from fpdf import FPDF
import os
import re

def create_combined_exum_highlighted_pdf(text, similarity_dicts, output_path):
    all_highlight_tokens = set()
    for sim in similarity_dicts:
        all_highlight_tokens.update(sim['tokens'])

    paragraphs = text.split('\n')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    for para in paragraphs:
        words = para.strip().split()
        if not words:
            pdf.ln()
            continue

        for word in words:
            plain = re.sub(r'\W+', '', word.lower())

            if plain in all_highlight_tokens:
                pdf.set_text_color(255, 255, 255)
                pdf.set_fill_color(255, 204, 0)
                pdf.cell(pdf.get_string_width(word + ' ') + 1, 7, word + ' ', ln=0, fill=True)
                pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(0, 0, 0)
            else:
                pdf.cell(pdf.get_string_width(word + ' ') + 1, 7, word + ' ', ln=0)

        pdf.ln(7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)

    return output_path
