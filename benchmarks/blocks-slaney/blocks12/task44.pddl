(define (problem BW-12-9546-44)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 - block)
    (:init
        (handempty)
        (on b1 b12)
        (on b2 b11)
        (on b3 b5)
        (on b4 b1)
        (on b5 b9)
        (on-table b6)
        (on b7 b3)
        (on b8 b2)
        (on b9 b8)
        (on b10 b6)
        (on b11 b4)
        (on b12 b10)
        (clear b7)
    )
    (:goal
        (and
            (on b1 b11)
            (on b2 b1)
            (on-table b3)
            (on b4 b7)
            (on b5 b8)
            (on b6 b5)
            (on b7 b10)
            (on b8 b9)
            (on b9 b3)
            (on b10 b2)
            (on b11 b6)
            (on b12 b4)
        )
    )
)