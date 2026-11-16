#!/usr/bin/env python3
from pathlib import Path
import shutil

TARGET = Path('scripts/generalization_validation_script/step35a_train_e2p8_domain_stress.py')

if not TARGET.is_file():
    raise FileNotFoundError(
        f'Run this from the dat-scan-classifier project root. Missing: {TARGET}'
    )

text = TARGET.read_text()
backup = TARGET.with_suffix(TARGET.suffix + '.bak_before_v1_2')
if not backup.exists():
    shutil.copy2(TARGET, backup)

old_import = 'from dataclasses import asdict\n'
new_import = 'from dataclasses import asdict, replace\n'
if old_import in text:
    text = text.replace(old_import, new_import, 1)
elif new_import not in text:
    raise RuntimeError('Could not patch dataclasses import safely.')

old_cfg_import = (
    'from src.configs.training_config import '
    'ExperimentConfig, save_experiment_config, seed_everything\n'
)
new_cfg_import = (
    'from src.configs.training_config import (\n'
    '    DEFAULT_TRAINING_CONFIG,\n'
    '    ExperimentConfig,\n'
    '    save_experiment_config,\n'
    '    seed_everything,\n'
    ')\n'
)
if old_cfg_import in text:
    text = text.replace(old_cfg_import, new_cfg_import, 1)
elif 'DEFAULT_TRAINING_CONFIG' not in text:
    raise RuntimeError('Could not patch training_config import safely.')

text = text.replace('    _shared_config_for_fold,\n', '', 1)

old_block = '''    # Reuse the authoritative central training policy with the same fold-specific
    # seed logic as common_cv_experiment.
    shared_args = argparse.Namespace(
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        amp=args.amp,
    )
    shared = _shared_config_for_fold(shared_args, args.fold)
    seed_everything(shared.reproducibility)
'''

new_block = '''    # Build the fold-specific shared configuration directly from the central
    # source of truth. Do NOT call common_cv_experiment._shared_config_for_fold:
    # that is a private CLI helper and its expected argparse fields can change.
    shared = DEFAULT_TRAINING_CONFIG

    fold_seed = int(shared.reproducibility.seed) + int(args.fold)
    reproducibility = replace(
        shared.reproducibility,
        seed=fold_seed,
    )

    dataloader = shared.dataloader
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be >= 0.")
        dataloader = replace(
            dataloader,
            num_workers=int(args.num_workers),
        )

    training = shared.training
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max-epochs must be >= 1.")
        training = replace(
            training,
            max_epochs=int(args.max_epochs),
        )

    if args.amp is not None:
        training = replace(
            training,
            use_amp=bool(args.amp),
        )

    shared = replace(
        shared,
        reproducibility=reproducibility,
        dataloader=dataloader,
        training=training,
    )
    shared.validate()
    seed_everything(shared.reproducibility)
'''

if old_block in text:
    text = text.replace(old_block, new_block, 1)
elif 'Do NOT call common_cv_experiment._shared_config_for_fold' not in text:
    raise RuntimeError(
        'Could not locate the Step35A shared-config block. '
        'No unsafe partial patch was written.'
    )

if '_shared_config_for_fold(' in text:
    raise RuntimeError('Private helper call still remains after patch; refusing to write.')

TARGET.write_text(text)

print('PATCH PASS')
print(f'Patched : {TARGET}')
print(f'Backup  : {backup}')
print('Private _shared_config_for_fold dependency removed.')
print('Now run py_compile before resubmitting.')
