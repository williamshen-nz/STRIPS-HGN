(define (problem BW-8-3326-23)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init
        (handempty)
        (on b1 b2)
        (on-table b2)
        (on-table b3)
        (on b4 b6)
        (on b5 b7)
        (on-table b6)
        (on b7 b4)
        (on b8 b1)
        (clear b3)
        (clear b5)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b8)
            (on b2 b1)
            (on-table b3)
            (on b4 b2)
            (on b5 b3)
            (on b6 b4)
            (on b7 b6)
            (on b8 b5)
        )
    )
)