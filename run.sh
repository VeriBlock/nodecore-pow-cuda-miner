while true;
do
  ./nodecore_pow_cuda -o testnet-pool-gpu.veriblock.org:8501 -u address -d 0 -bs 512 -tbs 1024;
  echo "Restarting PoW miner!";
  sleep 5;
done;
