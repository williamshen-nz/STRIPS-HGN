#!/bin/bash

set -e

DEST_DIR="../problems"
mkdir -p $DEST_DIR

make

# gen <size>
gen() {
    num_balls="$1"
    dest_name="gripper-n${num_balls}.pddl"
    echo "Generating problem $dest_name"
    ./gripper -n $num_balls > "${DEST_DIR}/${dest_name}"
}

# Problems up to 50 balls
for i in `seq 1 50`
do
  gen $i
done
