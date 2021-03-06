(define (problem BW-15-4678-30)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 - block)
    (:init
        (handempty)
        (on b1 b2)
        (on-table b2)
        (on-table b3)
        (on b4 b8)
        (on b5 b10)
        (on b6 b9)
        (on-table b7)
        (on b8 b5)
        (on b9 b14)
        (on-table b10)
        (on b11 b15)
        (on-table b12)
        (on b13 b7)
        (on b14 b3)
        (on b15 b4)
        (clear b1)
        (clear b6)
        (clear b11)
        (clear b12)
        (clear b13)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b1)
            (on b3 b5)
            (on b4 b3)
            (on b5 b8)
            (on b6 b7)
            (on b7 b13)
            (on-table b8)
            (on b9 b11)
            (on-table b10)
            (on b11 b12)
            (on-table b12)
            (on b13 b10)
            (on b14 b2)
            (on b15 b4)
        )
    )
)