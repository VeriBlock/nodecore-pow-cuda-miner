@echo off
SET APP_TITLE=VeriBlock NVidia GPU Miner
TITLE %APP_TITLE%

:Execute

VeriBlock-NodeCore-PoW-CUDA -o testnet-pool-gpu.veriblock.org:8501 -u YOUR_ADDRESS_HERE -d 0 -l false

echo ----------------------------------------------------------------------
echo ^| GPU Miner is restarting in 10 seconds!                             ^|
echo ^| Press CTRL+C to cancel this or ENTER to proceed immediately...     ^|
echo ----------------------------------------------------------------------
timeout /t 10 > NUL
echo Restarting GPU Miner...
goto Execute
