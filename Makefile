.PHONY: ing
SHELL=/bin/bash -O expand_aliases
DATADIR=/home/ejp/src/syncological/data


ing_exp1:
	-mkdir $(DATADIR)/ing_exp1
	-rm $(DATADIR)/ing_exp1/*
	parallel -j 12 -v \
		--joblog '$(DATADIR)/ing_exp1/log' \
		--nice 19 --delay 5 \
		'python bin/ing.py $(DATADIR)/ing_exp1/rate-{1}_ei-{2}_j-{3} -t 1.0 --stim 0.75 --rate {1} --w_ie {2} --seed {3}' ::: \
		5 10 30 ::: 0.1 0.12 0.14 0.16 0.18 0.20 ::: {1..20}

