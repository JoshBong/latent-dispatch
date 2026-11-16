TECTONIC ?= $(if $(wildcard tools/tectonic),./tools/tectonic,tectonic)

.PHONY: pdf setup clean
pdf:
	@mkdir -p build
	$(TECTONIC) --keep-logs --outdir build main.tex

setup:
	python3 scripts/install_tectonic.py

clean:
	rm -rf build
