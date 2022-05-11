#!/bin/bash

declare -a areas=("sambuca pistoiese" "villa basilica" "borgo a mozzano" "castel del piano" "santa fiora 23 " "magliano in toscana" "porcari" "roccalbegna" "fauglia" "semproniano")

for area in "${areas[@]}"; do
    echo $area
    python Sim.py -D "$area"
done
