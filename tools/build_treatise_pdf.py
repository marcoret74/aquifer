import sys, os, subprocess

class PDFBuilder:
    def __init__(self, filename="document.ps"):
        self.filename = filename
        self.pages = []
        self.current_page = []
        # Formato Carta (US Letter: 8.5 x 11 pulgadas = 612 x 792 puntos PostScript)
        self.width = 612
        self.height = 792
        self.margin_x = 50
        self.margin_top = 65
        self.margin_bottom = 55
        self.y = self.height - self.margin_top
        self.page_num = 1

    def new_page(self):
        if self.current_page:
            self.pages.append(self.current_page)
            self.current_page = []
        self.page_num += 1
        self.y = self.height - self.margin_top

    def add_op(self, op):
        self.current_page.append(op)

    def check_space(self, needed):
        if self.y - needed < self.margin_bottom:
            self.new_page()

    def title(self, text):
        self.check_space(48)
        self.add_op(f"/Helvetica-Bold findfont 17 scalefont setfont\n0.1 0.2 0.4 setrgbcolor\n{self.margin_x} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 24

    def subtitle(self, text):
        self.check_space(32)
        self.add_op(f"/Helvetica-Bold findfont 12 scalefont setfont\n0.2 0.3 0.5 setrgbcolor\n{self.margin_x} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 18

    def heading(self, text):
        self.check_space(28)
        self.y -= 6
        self.add_op(f"/Helvetica-Bold findfont 11 scalefont setfont\n0.15 0.25 0.35 setrgbcolor\n{self.margin_x} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 15

    def subheading(self, text):
        self.check_space(22)
        self.y -= 3
        self.add_op(f"/Helvetica-Bold findfont 9.5 scalefont setfont\n0.2 0.2 0.2 setrgbcolor\n{self.margin_x} {self.y} moveto\n({self.escape(text)}) show\n0 0 0 setrgbcolor\n")
        self.y -= 13

    def paragraph(self, text, font="Helvetica", size=9.0, leading=12.5, color=(0.1, 0.1, 0.1)):
        self.add_op(f"/{font} findfont {size} scalefont setfont\n{color[0]} {color[1]} {color[2]} setrgbcolor\n")
        words = text.split()
        lines = []
        cur_line = []
        cur_len = 0
        max_width = self.width - 2 * self.margin_x
        avg_char = size * 0.51

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
            self.add_op(f"{self.margin_x} {self.y} moveto\n({self.escape(l)}) show\n")
            self.y -= leading
        self.y -= 3

    def bullet(self, bold_prefix, text, size=8.8, leading=12.0):
        max_width = self.width - 2 * self.margin_x - 16
        avg_char = size * 0.51
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
        self.add_op(f"/Helvetica-Bold findfont {size} scalefont setfont\n0.2 0.4 0.6 setrgbcolor\n")
        self.add_op(f"{self.margin_x + 4} {self.y} moveto\n(\\\\267) show\n")
        self.add_op(f"{self.margin_x + 16} {self.y} moveto\n({self.escape(bold_prefix)}) show\n")
        prefix_offset = len(bold_prefix) * avg_char * 1.05
        self.add_op(f"/Helvetica findfont {size} scalefont setfont\n0.1 0.1 0.1 setrgbcolor\n")
        if lines:
            self.add_op(f"{self.margin_x + 16 + prefix_offset} {self.y} moveto\n({self.escape(lines[0])}) show\n")
            self.y -= leading
            for l in lines[1:]:
                self.add_op(f"{self.margin_x + 16} {self.y} moveto\n({self.escape(l)}) show\n")
                self.y -= leading
        self.y -= 2

    def callout(self, title, lines):
        h = len(lines) * 12.5 + 28
        self.check_space(h + 8)
        top_y = self.y
        bot_y = self.y - h
        self.add_op(f"0.94 0.96 0.98 setrgbcolor\nnewpath\n{self.margin_x} {bot_y} moveto\n{self.width - self.margin_x} {bot_y} lineto\n{self.width - self.margin_x} {top_y} lineto\n{self.margin_x} {top_y} lineto\nclosepath fill\n")
        self.add_op(f"0.15 0.35 0.65 setrgbcolor\nnewpath\n{self.margin_x} {bot_y} moveto\n{self.margin_x + 4} {bot_y} lineto\n{self.margin_x + 4} {top_y} lineto\n{self.margin_x} {top_y} lineto\nclosepath fill\n")
        self.y -= 13
        self.add_op(f"/Helvetica-Bold findfont 10 scalefont setfont\n0.15 0.35 0.65 setrgbcolor\n{self.margin_x + 12} {self.y} moveto\n({self.escape(title)}) show\n")
        self.y -= 13
        self.add_op(f"/Helvetica-Oblique findfont 8.5 scalefont setfont\n0.15 0.15 0.15 setrgbcolor\n")
        for l in lines:
            self.add_op(f"{self.margin_x + 12} {self.y} moveto\n({self.escape(l)}) show\n")
            self.y -= 12.5
        self.y = bot_y - 10

    def separator(self):
        self.check_space(12)
        self.y -= 4
        self.add_op(f"0.8 0.8 0.85 setrgbcolor\n0.8 setlinewidth\nnewpath\n{self.margin_x} {self.y} moveto\n{self.width - self.margin_x} {self.y} lineto\nstroke\n0 0 0 setrgbcolor\n")
        self.y -= 8

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
        out.append("%%PageOrder: Ascend\n")
        out.append("<</PageSize [612 792]>> setpagedevice\n")
        out.append("%%EndComments\n")

        for idx, page in enumerate(self.pages, 1):
            out.append(f"%%Page: {idx} {idx}\n")
            out.append("<</PageSize [612 792]>> setpagedevice\n")
            # Encabezado (posicionado de forma segura a 36 puntos del borde superior)
            header_y = self.height - 38
            header_line_y = self.height - 44
            out.append(f"/Helvetica findfont 7.5 scalefont setfont\n0.45 0.45 0.5 setrgbcolor\n")
            out.append(f"{self.margin_x} {header_y} moveto\n(Semantic Geometry Chile \\267 Monografia Cientifica y Dictamen Tecnico) show\n")
            out.append(f"0.8 0.83 0.88 setrgbcolor\n0.5 setlinewidth\nnewpath\n{self.margin_x} {header_line_y} moveto\n{self.width - self.margin_x} {header_line_y} lineto\nstroke\n")
            
            # Contenido de la página
            out.extend(page)
            
            # Pie de página (posicionado a 34 puntos del borde inferior)
            footer_line_y = 42
            footer_text_y = 30
            out.append(f"0.8 0.83 0.88 setrgbcolor\n0.5 setlinewidth\nnewpath\n{self.margin_x} {footer_line_y} moveto\n{self.width - self.margin_x} {footer_line_y} lineto\nstroke\n")
            out.append(f"/Helvetica findfont 7.5 scalefont setfont\n0.45 0.45 0.5 setrgbcolor\n")
            out.append(f"{self.margin_x} {footer_text_y} moveto\n(Inferencia Tensorial de Silicio a 15W \\267 Arquitectura de Sistemas y Fisica del Computo) show\n")
            out.append(f"{self.width - self.margin_x - 65} {footer_text_y} moveto\n(Pagina {idx} de {total_pages}) show\n")
            out.append("showpage\n")
        out.append("%%EOF\n")
        return "".join(out)

    def save_pdf(self, pdf_path):
        ps_data = self.build_postscript()
        ps_temp = pdf_path.replace(".pdf", ".ps")
        with open(ps_temp, "w", encoding="latin-1", errors="replace") as f:
            f.write(ps_data)
        subprocess.run(["ps2pdf", "-sPAPERSIZE=letter", ps_temp, pdf_path], check=True)
        if os.path.exists(ps_temp):
            os.remove(ps_temp)
        print(f"PDF Formato Carta generado en: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

