(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 )
(:init
(arm-empty)
(on b1 b2)
(on b2 b4)
(on-table b3)
(on b4 b3)
(on b5 b1)
(clear b5)
)
(:goal
(and
(on b1 b5)
(on b5 b2))
)
)


;; pddl-generators:
;; command: /home/skunk/ibm-ugr/pddl-generator/pddl_generators/blocksworld-4ops/../__blocksworld/blocksworld.sh 4ops 5 34000 /home/skunk/ibm-ugr/pddlsl/experiments/data/blocksworld/train/2e237657707f4819bb0dcc0d59c18c43.pddl
;; dict: {"blocks": 5}
;; date: 2023-08-08T04:47:15.319856
;; end:

