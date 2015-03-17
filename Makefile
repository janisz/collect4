.PHONY: specyfikacja_wstepna.pdf all clean

all: specyfikacja_wstepna.pdf 

# MAIN LATEXMK RULE
#
# # -pdf tells latexmk to generate PDF directly (instead of DVI).
# # -pdflatex="" tells latexmk to call a specific backend with specific options.
# # -use-make tells latexmk to call make for generating missing files.
#
# # -interactive=nonstopmode keeps the pdflatex backend from stopping at a
# # missing file reference and interactively asking you for an alternative.

specyfikacja_wstepna.pdf: ./doc/specyfikacja_wstepna.tex ./doc/img/opis_sieci.pdf
	cd doc; latexmk -f -pdf -pdflatex="pdflatex -interactive=nonstopmode" -use-make specyfikacja_wstepna.tex

doc/img/opis_sieci.pdf doc/img/opis_sieci.dot:
	dot -Tpdf doc/img/opis_sieci.dot -o doc/img/opis_sieci.pdf

clean:
	cd doc; latexmk -CA

