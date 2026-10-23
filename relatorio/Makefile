# Relatório de resultados — Oracle_N.
# Requer tectonic (https://tectonic-typesetting.github.io).

PDF := relatorio.pdf
SRC := relatorio.tex preambulo.tex $(wildcard secoes/*.tex)
RUN := ../results/experiments/2026-08-23T14-19-59_2286fcc
# Bateria de imagens (ADR 0022): a figura sai do primeiro dos dois runs.
IMG_RUN := ../results/experiments/2026-10-23T01-02-10_8a9d9bd

.PHONY: all figuras clean

all: $(PDF)

$(PDF): $(SRC) $(wildcard figuras/*.png)
	tectonic -X compile relatorio.tex

# Recopia as figuras a partir dos runs de referência.
figuras:
	cp $(RUN)/figures/*.png figuras/
	cp $(IMG_RUN)/figures/fig9_image_oracle_curves.png figuras/

clean:
	rm -f $(PDF) relatorio.aux relatorio.log relatorio.out relatorio.toc
