(define (problem BW-8-3326-53)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init
        (handempty)
        (on b1 b6)
        (on-table b2)
        (on b3 b1)
        (on-table b4)
        (on b5 b7)
        (on b6 b2)
        (on b7 b8)
        (on-table b8)
        (clear b3)
        (clear b4)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b6)
            (on b2 b3)
            (on b3 b1)
            (on-table b4)
            (on b5 b2)
            (on b6 b7)
            (on b7 b8)
            (on-table b8)
        )
    )
)