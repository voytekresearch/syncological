.PHONY: exp1
SHELL=/bin/bash -O expand_aliases
DATADIR=/home/ejp/src/syncological/data

exp1: ing_exp1 ping_exp1 async_exp1


ing_exp1:
	-mkdir $(DATADIR)/ing_exp1
	-rm $(DATADIR)/ing_exp1/*
	parallel -j 12 -v \
		--joblog '$(DATADIR)/ing_exp1/log' \
		--nice 19 --delay 5 \
		'python bin/ing.py $(DATADIR)/ing_exp1/rate-{1}_ie-{2}_j-{3} -t 1.0 --stim 0.75 --rate {1} --w_ie {2} --seed {3}' ::: \
		5 10 30 ::: \
		0.1 0.2 0.3 0.4 0.5 0.52 0.54 0.56 0.58 0.60 0.70 0.8 0.9 1.0 ::: \
		{1..20}

ping_exp1:
	-mkdir $(DATADIR)/ping_exp1
	-rm $(DATADIR)/ping_exp1/*
	parallel -j 12 -v \
		--joblog '$(DATADIR)/ping_exp1/log' \
		--nice 19 --delay 5 \
		'python bin/ping.py $(DATADIR)/ping_exp1/rate-{1}_ei-{2}_ie-{3}_j-{4} -t 1.0 --stim 0.75 --rate {1} --w_ei {2} --w_ie {3} --seed {4}' ::: \
		5 10 30 ::: \
		1.0 1.01 1.02 1.03 1.04 1.04 1.06 1.08 1.09 1.1 1.2 1.3 1.4 1.5 ::: \
		0.1 0.2 0.3 0.4 0.5 0.52 0.54 0.56 0.58 0.60 0.70 0.8 0.9 1.0 ::: \
		{1..20}

async_exp1:
	-mkdir $(DATADIR)/async_exp1
	-rm $(DATADIR)/async_exp1/*
	parallel -j 12 -v \
		--joblog '$(DATADIR)/async_exp1/log' \
		--nice 19 --delay 5 \
		'python bin/async.py $(DATADIR)/async_exp1/rate-{1}_j-{2} -t 1.0 --stim 0.75 --rate {1} --seed {2}' ::: \
		5 10 30 ::: \
		{1..20}
