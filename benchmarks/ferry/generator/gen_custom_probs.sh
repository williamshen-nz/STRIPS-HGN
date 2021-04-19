#!/bin/bash

set -e

DEST_DIR="../"
DEST_DIR_TRAIN="${DEST_DIR}train"
DEST_DIR_TEST="${DEST_DIR}test"

make
mkdir -p "${DEST_DIR}" "${DEST_DIR_TRAIN}"
mkdir -p "${DEST_DIR}" "${DEST_DIR_TEST}"

# gen <num_locations> <num_cars>
gen() {
    num_locations="$1"
    num_cars="$2"
    dest_name="ferry-l${num_locations}-c${num_cars}.pddl"
    echo "Generating train problem $dest_name"
    ./ferry -l $num_locations -c $num_cars > "${DEST_DIR_TRAIN}/${dest_name}"
}

# gen_test <num_locations> <num_cars>
gen_test() {
    num_locations="$1"
    num_cars="$2"
    dest_name="ferry-l${num_locations}-c${num_cars}.pddl"
    echo "Generating test problem $dest_name"
    ./ferry -l $num_locations -c $num_cars > "${DEST_DIR_TEST}/${dest_name}"
}

# Train on small number of location and cars
gen 2 1
gen 2 2
gen 2 3
gen 2 4

gen 3 1
gen 3 2
gen 3 3
gen 3 4

gen 4 1
gen 4 2
gen 4 3
gen 4 4

# Test on larger number of locations and cars
gen_test 2 5
gen_test 2 10
gen_test 2 15
gen_test 2 20

gen_test 3 5
gen_test 3 10
gen_test 3 15
gen_test 3 20

gen_test 4 5
gen_test 4 10
gen_test 4 15
gen_test 4 20

gen_test 5 5
gen_test 5 10
gen_test 5 15
gen_test 5 20

gen_test 6 5
gen_test 6 10
gen_test 6 15
gen_test 6 20

gen_test 7 5
gen_test 7 10
gen_test 7 15
gen_test 7 20

gen_test 8 5
gen_test 8 10
gen_test 8 15
gen_test 8 20

gen_test 9 5
gen_test 9 10
gen_test 9 15
gen_test 9 20

gen_test 9 5
gen_test 9 10
gen_test 9 15
gen_test 9 20

gen_test 10 5
gen_test 10 10
gen_test 10 15
gen_test 10 20