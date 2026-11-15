import sys, os, subprocess

class PDFBuilder:
    def __init__(self, filename="document.ps"):
        self.filename = filename
        self.pages = []
        self.current_page = []
        self.width = 595   # A4
        self.height = 842  # A4
        self.margin = 50
        self.y = self.height - self.margin
        self.page_num = 1

    def new_page(self):
        if self.current_page:
            self.pages.append(self.current_page)
            self.current_page = []
        self.page_num += 1
        self.y = self.height - self.margin

    def add_op(self, op):
        self.current_page.append(op)

    def check_space(self, needed):
        if self.y - needed < self.margin + 30:
            self.new_page()

    def title(self, text):
        self.check_space(50)
        self.add_op(f"/Helvetica-Bold findfont 18 scalefont setfont\n0.1 0.2 0.4 setrgbcolor\n{self.margin} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 26

    def subtitle(self, text):
        self.check_space(35)
        self.add_op(f"/Helvetica-Bold findfont 13 scalefont setfont\n0.2 0.3 0.5 setrgbcolor\n{self.margin} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 20

    def heading(self, text):
        self.check_space(30)
        self.y -= 8
        self.add_op(f"/Helvetica-Bold findfont 12 scalefont setfont\n0.15 0.25 0.35 setrgbcolor\n{self.margin} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 16

    def subheading(self, text):
        self.check_space(25)
        self.y -= 4
        self.add_op(f"/Helvetica-Bold findfont 10.5 scalefont setfont\n0.2 0.2 0.2 setrgbcolor\n{self.margin} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 14

    def paragraph(self, text, font="Helvetica", size=9.5, leading=13, color=(0.1, 0.1, 0.1)):
        self.add_op(f"/{font} findfont {size} scalefont setfont\n{color[0]} {color[1]} {color[2]} setrgbcolor\n")
        words = text.split()
        lines = []
        cur_line = []
        cur_len = 0
        max_width = self.width - 2 * self.margin
        # approx char width for 9.5pt Helvetica is ~5.0 pt
        avg_char = size * 0.52

        for w in words:
            w_len = len(w) * avg_char + (avg_char if cur_line else 0)
            if cur_len + w_len > max_width and cur_line:
                lines.append(" ".join(cur_line))
                cur_line = [w]
                cur_len = len(w) * avg_char
            else:
                cur_line.append(w)
                cur_len += w_len
        if cur_line:
            lines.append(" ".join(cur_line))

        for l in lines:
            self.check_space(leading)
            self.add_op(f"{self.margin} {self.y} moveto\n({self.escape(l)}) show\n")
            self.y -= leading
        self.y -= 4

    def bullet(self, bold_prefix, text, size=9.2, leading=12.5):
        max_width = self.width - 2 * self.margin - 16
        avg_char = size * 0.52
        words = text.split()
        lines = []
        cur_line = []
        cur_len = len(bold_prefix) * avg_char * 1.1

        for w in words:
            w_len = len(w) * avg_char + (avg_char if cur_line else 0)
            if cur_len + w_len > max_width and cur_line:
                lines.append(" ".join(cur_line))
                cur_line = [w]
                cur_len = len(w) * avg_char
            else:
                cur_line.append(w)
                cur_len += w_len
        if cur_line:
            lines.append(" ".join(cur_line))

        self.check_space(leading * len(lines) + 4)
        # Bullet point symbol
        self.add_op(f"/Helvetica-Bold findfont {size} scalefont setfont\n0.2 0.4 0.6 setrgbcolor\n")
        self.add_op(f"{self.margin + 4} {self.y} moveto\n(\\\\267) show\n")
        self.add_op(f"{self.margin + 16} {self.y} moveto\n({self.escape(bold_prefix)}) show\n")
        prefix_offset = len(bold_prefix) * avg_char * 1.05
        self.add_op(f"/Helvetica findfont {size} scalefont setfont\n0.1 0.1 0.1 setrgbcolor\n")
        if lines:
            self.add_op(f"{self.margin + 16 + prefix_offset} {self.y} moveto\n({self.escape(lines[0])}) show\n")
            self.y -= leading
            for l in lines[1:]:
                self.add_op(f"{self.margin + 16} {self.y} moveto\n({self.escape(l)}) show\n")
                self.y -= leading
        self.y -= 3

    def callout(self, title, lines):
        h = len(lines) * 13 + 30
        self.check_space(h + 10)
        top_y = self.y
        bot_y = self.y - h
        # Draw background rectangle
        self.add_op(f"0.94 0.96 0.98 setrgbcolor\nnewpath\n{self.margin} {bot_y} moveto\n{self.width - self.margin} {bot_y} lineto\n{self.width - self.margin} {top_y} lineto\n{self.margin} {top_y} lineto\nclosepath fill\n")
        # Draw left bar
        self.add_op(f"0.15 0.35 0.65 setrgbcolor\nnewpath\n{self.margin} {bot_y} moveto\n{self.margin + 4} {bot_y} lineto\n{self.margin + 4} {top_y} lineto\n{self.margin} {top_y} lineto\nclosepath fill\n")
        # Header
        self.y -= 14
        self.add_op(f"/Helvetica-Bold findfont 10.5 scalefont setfont\n0.15 0.35 0.65 setrgbcolor\n{self.margin + 12} {self.y} moveto\n({self.escape(title)}) show\n")
        self.y -= 14
        # Body lines
        self.add_op(f"/Helvetica-Oblique findfont 9 scalefont setfont\n0.15 0.15 0.15 setrgbcolor\n")
        for l in lines:
            self.add_op(f"{self.margin + 12} {self.y} moveto\n({self.escape(l)}) show\n")
            self.y -= 13
        self.y = bot_y - 12

    def separator(self):
        self.check_space(15)
        self.y -= 5
        self.add_op(f"0.8 0.8 0.85 setrgbcolor\n1 setlinewidth\nnewpath\n{self.margin} {self.y} moveto\n{self.width - self.margin} {self.y} lineto\nstroke\n0 0 0 setrgbcolor\n")
        self.y -= 10

    def escape(self, s):
        return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    def build_postscript(self):
        if self.current_page:
            self.pages.append(self.current_page)
        total_pages = len(self.pages)
        out = []
        out.append("%!PS-Adobe-3.0\n")
        out.append(f"%%BoundingBox: 0 0 {self.width} {self.height}\n")
        out.append("%%Title: Geometria Semantica y Memoria Topologica\n")
        out.append("%%Creator: Semantic Geometry Chile\n")
        out.append(f"%%Pages: {total_pages}\n")
        out.append("%%EndComments\n")

        for idx, page in enumerate(self.pages, 1):
            out.append(f"%%Page: {idx} {idx}\n")
            # Header
            out.append(f"/Helvetica findfont 8 scalefont setfont\n0.5 0.5 0.5 setrgbcolor\n")
            out.append(f"{self.margin} {self.height - 30} moveto\n(Semantic Geometry Chile \\267 Monografia Cientifica y Dictamen Tecnico) show\n")
            out.append(f"0.85 0.85 0.85 setrgbcolor\n0.5 setlinewidth\nnewpath\n{self.margin} {self.height - 35} moveto\n{self.width - self.margin} {self.height - 35} lineto\nstroke\n")
            # Page Body
            out.extend(page)
            # Footer
            out.append(f"0.85 0.85 0.85 setrgbcolor\n0.5 setlinewidth\nnewpath\n{self.margin} 40 moveto\n{self.width - self.margin} 40 lineto\nstroke\n")
            out.append(f"/Helvetica findfont 8 scalefont setfont\n0.5 0.5 0.5 setrgbcolor\n")
            out.append(f"{self.margin} 28 moveto\n(Inferencia Tensorial de Silicio a 15W \\267 Arquitectura de Sistemas y Fisica del Computo) show\n")
            out.append(f"{self.width - self.margin - 65} 28 moveto\n(Pagina {idx} de {total_pages}) show\n")
            out.append("showpage\n")
        out.append("%%EOF\n")
        return "".join(out)

    def save_pdf(self, pdf_path):
        ps_data = self.build_postscript()
        ps_temp = pdf_path.replace(".pdf", ".ps")
        with open(ps_temp, "w", encoding="latin-1", errors="replace") as f:
            f.write(ps_data)
        subprocess.run(["ps2pdf", ps_temp, pdf_path], check=True)
        if os.path.exists(ps_temp):
            os.remove(ps_temp)
        print(f"PDF generado exitosamente en: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

print("PDFBuilder ready.")
