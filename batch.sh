#!/bin/bash

declare -a areas=("borgo a mozzano" "santa fiora" "magliano in toscana" "porcari" "roccalbegna" "fauglia" "sambuca pistoiese" "villa basilica" "castel del piano" "semproniano")

for area in "${areas[@]}"; do
    echo $area
    python Sim.py -D "$area" --types dijkstra_1
done
