(define (problem BW-7-6874-58)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b1)
        (on-table b3)
        (on b4 b7)
        (on b5 b2)
        (on-table b6)
        (on b7 b5)
        (clear b3)
        (clear b4)
        (clear b6)
    )
    (:goal
        (and
            (on b1 b5)
            (on-table b2)
            (on b3 b6)
            (on b4 b1)
            (on b5 b2)
            (on-table b6)
            (on b7 b4)
        )
    )
)