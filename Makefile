.PHONY: test benchmark calibrate ens328 bundle verify

test:
	python -m pytest -q tests

benchmark:
	calivox benchmark configs/datscan_3d.yaml --skip-existing --write runs/benchmark_3d.md
	calivox benchmark configs/datscan_2p5d.yaml --skip-existing --write runs/benchmark_2p5d.md

calibrate:
	calivox calibrate runs --write runs/calibrated.md

ens328:
	calivox ensemble runs --members configs/ens328.yaml --out runs/ENS328.json

bundle: ens328
	calivox freeze runs/ENS328.json --out bundles/ENS328
