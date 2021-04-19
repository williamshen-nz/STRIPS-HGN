#!/bin/bash

set -e

DEST_DIR="../"
DEST_DIR_TRAIN="${DEST_DIR}train/"
DEST_DIR_TEST="${DEST_DIR}test/"

make
mkdir -p "${DEST_DIR}" "${DEST_DIR_TRAIN}"
mkdir -p "${DEST_DIR}" "${DEST_DIR_TEST}"

# gen <cities> <planes> <people> <seed>
gen() {
    n_cities="$1"
    n_planes="$2"
    n_people="$3"
    seed="$4"
    dest_name="zenotravel-cities${n_cities}-planes${n_planes}-people${n_people}-${seed}.pddl"
    echo "Generating train problem $dest_name"
    ./ztravel $seed $n_cities $n_planes $n_people > "${DEST_DIR_TRAIN}/${dest_name}"
}

# gen_test <cities> <planes> <people> <seed>
gen_test() {
    n_cities="$1"
    n_planes="$2"
    n_people="$3"
    seed="$4"
    dest_name="zenotravel-cities${n_cities}-planes${n_planes}-people${n_people}-${seed}.pddl"
    echo "Generating test problem $dest_name"
    ./ztravel $seed $n_cities $n_planes $n_people > "${DEST_DIR_TEST}/${dest_name}"
}

# Training Problems
# <cities> <planes> <people> <seed>
# 10 x 2 cities
gen 2 1 2 1864
gen 2 1 3 8798
gen 2 2 2 7284
gen 2 2 3 9145
gen 2 3 2 1325
gen 2 3 3 3417
gen 2 3 4 7627
gen 2 4 2 4892
gen 2 4 3 1657
gen 2 4 4 6874

# 10 x 3 cities
gen 3 1 2 8152
gen 3 1 3 4791
gen 3 2 2 9581
gen 3 2 3 8752
gen 3 2 4 6913
gen 3 2 5 7306
gen 3 3 2 8942
gen 3 3 3 1826
gen 3 3 4 2981
gen 3 3 5 4582

# Testing problems
# {2, 3, 4 cities} x {2, 3, 4, 5 planes} x {3, 4, 5, 6, 7} people = 60 problems
# 2 cities (20 problems), {3, 4, 5, 6, 7} people
# => 2 planes
gen_test 2 2 3 2168
gen_test 2 2 4 8940
gen_test 2 2 5 1564
gen_test 2 2 6 8432
gen_test 2 2 7 7462
# => 3 planes
gen_test 2 3 3 4872
gen_test 2 3 4 2987
gen_test 2 3 5 2164
gen_test 2 3 6 6510
gen_test 2 3 7 3468
# => 4 planes
gen_test 2 4 3 7462
gen_test 2 4 4 2248
gen_test 2 4 5 8992
gen_test 2 4 6 1067
gen_test 2 4 7 6599
# => 5 planes
gen_test 2 5 3 9854
gen_test 2 5 4 2145
gen_test 2 5 5 5664
gen_test 2 5 6 6522
gen_test 2 5 7 4850

# 3 cities (20 problems), {3, 4, 5, 6, 7} people
# => 2 planes
gen_test 3 2 3 9462
gen_test 3 2 4 8405
gen_test 3 2 5 9812
gen_test 3 2 6 8261
gen_test 3 2 7 1956
# => 3 planes
gen_test 3 3 3 8975
gen_test 3 3 4 5712
gen_test 3 3 5 9867
gen_test 3 3 6 1573
gen_test 3 3 7 4929
# => 4 planes
gen_test 3 4 3 8943
gen_test 3 4 4 9527
gen_test 3 4 5 3947
gen_test 3 4 6 4985
gen_test 3 4 7 6173
# => 5 planes
gen_test 3 5 3 1564
gen_test 3 5 4 2347
gen_test 3 5 5 7182
gen_test 3 5 6 1956
gen_test 3 5 7 4813

# 4 cities (20 problems), {3, 4, 5, 6, 7} people
# Network starts failing around here for lots of people
# => 2 planes
gen_test 4 2 3 4571
gen_test 4 2 4 5642
gen_test 4 2 5 8952
gen_test 4 2 6 3248
gen_test 4 2 7 7627
# => 3 planes
gen_test 4 3 3 2056
gen_test 4 3 4 5912
gen_test 4 3 5 8691
gen_test 4 3 6 6549
gen_test 4 3 7 1842
# => 4 planes
gen_test 4 4 3 3078
gen_test 4 4 4 2417
gen_test 4 4 5 6891
gen_test 4 4 6 1235
gen_test 4 4 7 4853
# => 5 planes
gen_test 4 5 3 3894
gen_test 4 5 4 9648
gen_test 4 5 5 3690
gen_test 4 5 6 2792
gen_test 4 5 7 5762

## 5 cities (16 problems), {3, 4, 5, 6} people
## Network starts failing around here for lots of people
## => 2 planes
#gen_test 5 2 3 8971
#gen_test 5 2 4 4653
#gen_test 5 2 5 1687
#gen_test 5 2 6 4610
## => 3 planes
#gen_test 5 3 3 5463
#gen_test 5 3 4 9054
#gen_test 5 3 5 6874
#gen_test 5 3 6 3247
## => 4 planes
#gen_test 5 4 3 7156
#gen_test 5 4 4 4062
#gen_test 5 4 5 6574
#gen_test 5 4 6 9412
## => 5 planes
#gen_test 5 5 3 8424
#gen_test 5 5 4 6240
#gen_test 5 5 5 3521
#gen_test 5 5 6 1854

## IPC Problems
## Some smaller problems
#ztravel 1000 3 1 2 1000 > pfile1
#ztravel 1202 3 1 3 1000 > pfile2
#ztravel 1345 3 2 4 1000 > pfile3
#ztravel 5637 3 2 5 1000 > pfile4
#ztravel 1976 4 2 4 1000 > pfile5
#ztravel 1212 4 2 5 1000 > pfile6
#ztravel 3000 4 2 6 1000 > pfile7
#
## Now start to increase complexity a bit
#ztravel 2222 5 3 6 1000 > pfile8
#ztravel 3434 5 3 7 1000 > pfile9
#ztravel 9987 5 3 8 1000 > pfile10
#ztravel 1653 6 3 7 1000 > pfile11
#ztravel 1700 6 3 8 1000 > pfile12
#ztravel 2000 6 3 10 1000 > pfile13
#
## Are we still playing?
#ztravel 4312 10 5 10 1000 > pfile14
#ztravel 3452 12 5 15 1000 > pfile15
#ztravel 3412 14 5 15 1000 > pfile16
#ztravel 3152 16 5 20 1000 > pfile17
#ztravel 1452 18 5 20 1000 > pfile18
#ztravel 3451 20 5 25 1000 > pfile19
#ztravel 3222 22 5 25 1000 > pfile20
