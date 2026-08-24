# Relatório de resultados — Oracle_N.
# Requer tectonic (https://tectonic-typesetting.github.io).

PDF := relatorio.pdf
SRC := relatorio.tex preambulo.tex $(wildcard secoes/*.tex)
RUN := ../results/experiments/2026-08-23T14-19-59_2286fcc

.PHONY: all figuras clean

all: $(PDF)

$(PDF): $(SRC) $(wildcard figuras/*.png)
	tectonic -X compile relatorio.tex

# Recopia as figuras a partir do run de referência.
figuras:
	cp $(RUN)/figures/*.png figuras/

clean:
	rm -f $(PDF) relatorio.aux relatorio.log relatorio.out relatorio.toc
