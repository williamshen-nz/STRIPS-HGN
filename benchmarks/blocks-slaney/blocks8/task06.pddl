(define (problem BW-8-3326-6)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init
        (handempty)
        (on b1 b4)
        (on b2 b7)
        (on b3 b1)
        (on-table b4)
        (on b5 b8)
        (on-table b6)
        (on b7 b3)
        (on b8 b6)
        (clear b2)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b4)
            (on-table b2)
            (on-table b3)
            (on b4 b7)
            (on-table b5)
            (on b6 b2)
            (on-table b7)
            (on b8 b1)
        )
    )
)