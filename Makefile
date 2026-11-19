PY ?= python3
.PHONY: all scene sequence test report bundle compare video factory web

all:        ## full pipeline (renders, tests, report, bundle)
	./run_all.sh
scene:      ## rebuild scene.blend from MPFB2 + MakeHuman assets (MH_ASSETS=...)
	$(BLENDER_PATH) -b --python-exit-code 1 -P tools/build_scene.py -- --assets $(MH_ASSETS) --out scene.blend
sequence:
	$(PY) wrapper/eyes.py sequence --out renders
test:
	$(PY) -m pytest -q tests
report:
	$(PY) tools/make_report.py
bundle:
	$(PY) tools/make_bundle.py
compare:    ## spec vs physio vs physio_full side-by-sides
	$(BLENDER_PATH) -b scene.blend --python-exit-code 1 -P tools/compare.py -- renders/compare
video:
	$(PY) tools/make_video.py
factory:    ## XR near-eye dataset (N=200 smoke run by default)
	$(BLENDER_PATH) -b scene.blend --python-exit-code 1 -P tools/factory.py -- --n $${N:-200} --seed 0 --out data
web:
	npx serve web
