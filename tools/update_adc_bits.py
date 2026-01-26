import json
from pathlib import Path


def update_adc_bits(path: Path, new_values):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    def recurse(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'adc_bits':
                    # Replace choice values if present
                    if isinstance(v, dict) and 'values' in v:
                        v['values'] = new_values
                else:
                    recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(data)

    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    cfg_path = Path('0 - DADS dataset extraction') / 'build_config.json'
    if not cfg_path.exists():
        print(f'Config file not found: {cfg_path}')
    else:
        update_adc_bits(cfg_path, [24, 16])
        print(f'Updated adc_bits in {cfg_path} to [24, 16]')
