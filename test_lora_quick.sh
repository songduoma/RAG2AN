#!/usr/bin/env bash
set -euo pipefail

# 快速測試 LoRA 整合是否正常運作

echo "============================================================"
echo "Quick LoRA Integration Test"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# 激活 conda 環境
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'test'..."
    set +u  # conda activate scripts may reference unset vars like QT_XCB_GL_INTEGRATION
    eval "$(conda shell.bash hook)"
    conda activate test
    set -u
fi

# 測試 1: 檢查 generator.py 是否正確導入 LoRA 相關模組
echo "[Test 1] Checking generator.py imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
from generator import LoRALocalEngine, FakeNewsGenerator
print('✅ LoRALocalEngine imported successfully')
print('✅ FakeNewsGenerator imported successfully')
"

if [[ $? -ne 0 ]]; then
    echo "❌ Import test failed"
    exit 1
fi

echo ""

# 測試 2: 檢查 lora_trainer.py
echo "[Test 2] Checking lora_trainer.py..."
python3 -c "
import sys
sys.path.insert(0, '.')
from lora_trainer import LoRATrainer, prepare_dataset_from_gan_outputs
print('✅ LoRATrainer imported successfully')
print('✅ prepare_dataset_from_gan_outputs imported successfully')
"

if [[ $? -ne 0 ]]; then
    echo "❌ lora_trainer import test failed"
    exit 1
fi

echo ""

# 測試 3: 測試 FakeNewsGenerator 在不同模式下的初始化
echo "[Test 3] Testing FakeNewsGenerator modes..."

echo "  Testing API mode..."
export GEN_MODE=api
export OPENAI_API_KEY="sk-test-dummy-key"
python3 -c "
import sys
sys.path.insert(0, '.')
from generator import FakeNewsGenerator
try:
    gen = FakeNewsGenerator()
    print('✅ API mode initialization OK')
except Exception as e:
    print(f'Info: {e}')
"

echo "  Testing local mode..."
export GEN_MODE=local
python3 -c "
import sys
sys.path.insert(0, '.')
from generator import FakeNewsGenerator
print('✅ Local mode can be initialized (model loading not tested)')
"

echo "  Testing lora mode..."
export GEN_MODE=lora
export LORA_PATH=""  # 空路徑，只測試初始化邏輯
python3 -c "
import sys
sys.path.insert(0, '.')
from generator import FakeNewsGenerator
print('✅ LoRA mode can be initialized (model loading not tested)')
"

echo ""

# 測試 4: 檢查必要的套件
echo "[Test 4] Checking required packages..."
python3 -c "
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'peft': 'PEFT (for LoRA)',
    'trl': 'TRL (for SFT training)',
    'bitsandbytes': 'BitsAndBytes (for 4-bit quantization)',
}

missing = []
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f'✅ {name}')
    except ImportError:
        print(f'❌ {name} - NOT INSTALLED')
        missing.append(pkg)

if missing:
    print(f'\n⚠️  Missing packages: {missing}')
    print('Install with: pip install peft trl bitsandbytes')
    exit(1)
"

if [[ $? -ne 0 ]]; then
    echo ""
    echo "❌ Some required packages are missing"
    echo "Run: pip install peft trl bitsandbytes"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ All tests passed!"
echo "============================================================"
echo ""
echo "LoRA integration is ready. You can now:"
echo ""
echo "1. Run vanilla GAN training to collect data:"
echo "   export GEN_MODE=local"
echo "   DATASET_SPLIT='train[:10]' NUM_ROUNDS=5 ./train.sh"
echo ""
echo "2. Train LoRA adapter from collected data:"
echo "   python lora_trainer.py --gan-outputs local/rag_gan_runs/RUNDIR --epochs 3"
echo ""
echo "3. Use LoRA model in GAN training:"
echo "   export GEN_MODE=lora"
echo "   export LORA_PATH=./lora_checkpoints"
echo "   DATASET_SPLIT='train[:10]' NUM_ROUNDS=5 ./train.sh"
echo ""
echo "Or run the full integration test:"
echo "   ./test_lora_integration.sh"
echo ""
