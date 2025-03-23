REPORT = report
REFERENCES = references

TEX = xelatex
BIB = bibtex

AUXFILES = *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.lol *.synctex.gz

all: $(REPORT).pdf $(REPORT)-cn.pdf

$(REPORT).pdf: $(REPORT).tex $(REFERENCES).bib algorithms.tex optimizations.tex
	$(TEX) $(REPORT).tex
	$(BIB) $(REPORT)
	$(TEX) $(REPORT).tex
	$(TEX) $(REPORT).tex

$(REPORT)-cn.pdf: $(REPORT)-cn.tex $(REFERENCES).bib
	$(TEX) $(REPORT)-cn.tex
	$(BIB) $(REPORT)
	$(TEX) $(REPORT)-cn.tex
	$(TEX) $(REPORT)-cn.tex

clean:
	rm -f $(AUXFILES)

clean-all:
	rm -f $(AUXFILES) $(REPORT).pdf $(REPORT)-cn.pdf

.PHONY: all clean
