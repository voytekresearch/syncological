SHELL=/bin/bash -O expand_aliases
.PHONY: exp1
DATADIR=/home/ejp/src/syncological/data
TYPEDATADIR=/Users/type/Code/syncological/data
TYPEHARDPY=/Users/type/anaconda/bin/

exp1: ing_exp1 ping_exp1 async_exp1


# --------------------------------------------------------------------------
#  Stimulus reacts to w_ei, w_ie and rate
ing_exp1:
	-mkdir $(DATADIR)/ing_exp1
	-rm $(DATADIR)/ing_exp1/*
	parallel -j 12 -v \
		--joblog '$(DATADIR)/ing_exp1/log' \
		--nice 19 --delay 2 \
		'python bin/ing.py $(DATADIR)/ing_exp1/rate-{1}_ie-{2}_j-{3} -t 1.0 --stim 0.75 --rate {1} --w_ie {2} --seed {3}' ::: \
		5 10 30 ::: \
		0.1 0.2 0.3 0.4 0.5 0.52 0.54 0.56 0.58 0.60 0.70 0.8 0.9 1.0 ::: \
		{1..20}

# Osc and stim synchony reacts to I drive changes
ing_exp2:
	-mkdir $(TYPEDATADIR)/ing_exp2
	-rm $(TYPEDATADIR)/ing_exp2/*
	parallel -j 12 -v \
		--joblog '$(TYPEDATADIR)/ing_exp2/log' \
		--nice 19 --delay 2 \
		'$(TYPEHARDPY)/python bin/ing.py $(TYPEDATADIR)/ing_exp2/I_e-{1}-{1}_j-{2} -t 1.0 --stim 0.75 --I_e {1} {1} --seed {2}' ::: \
		0.1 0.2 0.3 0.4 0.6 0.8 ::: \
		{1..20}

# Synchony reacts to I stability (via I_i_sigma)
ing_exp3:
	-mkdir $(TYPEDATADIR)/ing_exp3
	-rm $(TYPEDATADIR)/ing_exp3/*
	parallel -j 12 -v \
		--joblog '$(TYPEDATADIR)/ing_exp3/log' \
		--nice 19 --delay 2 \
		'$(TYPEHARDPY)/python bin/ing.py $(TYPEDATADIR)/ing_exp3/I_i_sigma-{1}_j-{2} -t 1.0 --stim 0.75 --I_i_sigma {1} --seed {2}' ::: \
		0.01 0.02 0.04 0.06 0.08 0.1 ::: \
		{1..20}


# --------------------------------------------------------------------------
#  Stimulus reacts to w_ei, w_ie and rate
ping_exp1:
	-mkdir $(DATADIR)/ping_exp1
	-rm $(DATADIR)/ping_exp1/*
	parallel -j 12 -v \
		--joblog '$(DATADIR)/ping_exp1/log' \
		--nice 19 --delay 2 \
		'python bin/ping.py $(DATADIR)/ping_exp1/rate-{1}_ei-{2}_ie-{3}_j-{4} -t 1.0 --stim 0.75 --rate {1} --w_ei {2} --w_ie {3} --seed {4}' ::: \
		5 10 30 ::: \
		1.0 1.01 1.02 1.03 1.04 1.04 1.06 1.08 1.09 1.1 1.2 1.3 1.4 1.5 ::: \
		0.1 0.2 0.3 0.4 0.5 0.52 0.54 0.56 0.58 0.60 0.70 0.8 0.9 1.0 ::: \
		{1..20}

# Osc and stim synchony reacts to drive changes
ping_exp2:
	-mkdir $(TYPEDATADIR)/ping_exp2
	-rm $(TYPEDATADIR)/ping_exp2/*
	parallel -j 12 -v \
		--joblog '$(TYPEDATADIR)/ping_exp2/log' \
		--nice 19 --delay 2 \
		'$(TYPEHARDPY)/python bin/ping.py $(TYPEDATADIR)/ping_exp2/I_e-{1}-{1}_j-{2} -t 1.0 --stim 0.75 --I_e {1} {1} --seed {2}' ::: \
		0.3 0.4 0.5 0.6 0.8 1.0 ::: \
		{1..20}


# --------------------------------------------------------------------------
#  Stimulus reacts to rate under chaos
async_exp1:
	-mkdir $(TYPEDATADIR)/async_exp1
	-rm $(TYPEDATADIR)/async_exp1/*
	parallel -j 12 -v \
		--joblog '$(TYPEDATADIR)/async_exp1/log' \
		--nice 19 --delay 2 \
		'$(TYPEHARDPY)/python bin/async.py $(TYPEDATADIR)/async_exp1/rate-{1}_j-{2} -t 1.0 --stim 0.75 --rate {1} --seed {2}' ::: \
		5 10 30 ::: \
		{1..20}
