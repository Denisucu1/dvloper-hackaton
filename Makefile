# Team adn - local workflow. Everything that is not `image`/`push` runs without Docker.
PY      ?= python3
REG     ?= harbor.hackaton-2026.dvloper.io
TEAM    ?= adn
TAG     ?= t1
IMAGE   ?= $(REG)/$(TEAM)/bot
GAME    ?= tic_tac_toe
ARENA   := arena-lite
LABEL   ?= run
BENCH_ARGS ?=

.PHONY: help test bench prep-trace prep-trace-real smoke prep practice list image run push digest clean

prep-trace:
	$(PY) scripts/prep_trace.py

prep-trace-real:
	$(PY) scripts/prep_trace.py --real --game $(GAME) --verbose

bench:
	$(PY) scripts/bench.py --label "$(LABEL)" $(BENCH_ARGS)

help:
	@echo "make test            - unit tests (stdlib unittest, no deps)"
	@echo "make prep-trace      - visual timeline of the 10-minute prep (offline, no key)"
	@echo "make prep-trace-real GAME=pig - same, against the real gateway"
	@echo "make bench           - benchmark + delta vs the previous run of the same config"
	@echo "make smoke           - scripted protocol run against main.py (no network)"
	@echo "make list            - practice games in arena-lite"
	@echo "make prep GAME=pig   - one prep run with the real dev key"
	@echo "make practice        - prep + full match vs the random bot"
	@echo "make image TAG=t1    - docker build --platform linux/amd64"
	@echo "make push  TAG=t1    - push the frozen tag"
	@echo "make digest TAG=t1   - what the registry serves"

test:
	$(PY) -m unittest discover -s tests -v

smoke:
	$(PY) scripts/smoke.py

list:
	cd $(ARENA) && $(PY) -m arena practice --list

prep:
	cd $(ARENA) && $(PY) -m arena prep --bot "cmd:$(PY) ../main.py" --game $(GAME) --env-file ../dev.env --seconds 600

practice:
	cd $(ARENA) && $(PY) -m arena practice --prep \
		--bot-a "cmd:$(PY) ../main.py" --bot-b "cmd:$(PY) examples/random_bot.py" \
		--env-file ../dev.env --game $(GAME)

image:
	docker build --platform linux/amd64 -t $(IMAGE):$(TAG) .

run:
	docker run -i --rm $(IMAGE):$(TAG)

push: image
	docker push $(IMAGE):$(TAG)

digest:
	docker buildx imagetools inspect $(IMAGE):$(TAG)

clean:
	rm -rf results logs memory __pycache__ harness/__pycache__ tests/__pycache__
