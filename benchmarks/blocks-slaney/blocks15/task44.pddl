(define (problem BW-15-4678-44)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 - block)
    (:init
        (handempty)
        (on b1 b4)
        (on b2 b8)
        (on b3 b13)
        (on b4 b3)
        (on b5 b10)
        (on b6 b14)
        (on-table b7)
        (on b8 b5)
        (on-table b9)
        (on b10 b7)
        (on b11 b6)
        (on b12 b15)
        (on b13 b12)
        (on b14 b1)
        (on b15 b2)
        (clear b9)
        (clear b11)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b1)
            (on b4 b5)
            (on b5 b9)
            (on b6 b4)
            (on b7 b15)
            (on b8 b13)
            (on-table b9)
            (on b10 b6)
            (on b11 b8)
            (on-table b12)
            (on b13 b7)
            (on-table b14)
            (on-table b15)
        )
    )
)