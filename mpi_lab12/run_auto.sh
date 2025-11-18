#!/bin/bash

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PROCS_LIST="1 2 4 8"
SCRIPT="hybrid_matvec_wsl2.py"
LOG="results.log"

echo "=== Hybrid MPI + OpenMP Benchmark ===" > $LOG
echo "Date: $(date)" >> $LOG
echo "-------------------------------------" >> $LOG

for p in $PROCS_LIST; do
    echo "" | tee -a $LOG
    echo "Running with $p MPI process(es)..." | tee -a $LOG
    mpirun --oversubscribe -np $p python3 $SCRIPT | tee -a $LOG
    echo "-------------------------------------" >> $LOG
done

echo "" >> $LOG
echo "All tests completed." | tee -a $LOG

