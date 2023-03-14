#!/bin/bash
for i in {0..9}
do
    python ipip-ffm.py $i
done
for i in {0..499}
do
    python sim_data.py $i
done
for i in {0..1199}
do
    python recovery_and_comparison.py $i
done
for i in {0..1199}
do
    python c2st_analyses.py $i
done
python make_figures.py
