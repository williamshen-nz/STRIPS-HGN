(define (problem BW-15-4678-6)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 - block)
    (:init
        (handempty)
        (on b1 b15)
        (on b2 b1)
        (on b3 b2)
        (on b4 b5)
        (on b5 b8)
        (on-table b6)
        (on b7 b4)
        (on b8 b13)
        (on b9 b11)
        (on b10 b6)
        (on b11 b10)
        (on-table b12)
        (on-table b13)
        (on b14 b7)
        (on b15 b14)
        (clear b3)
        (clear b9)
        (clear b12)
    )
    (:goal
        (and
            (on b1 b9)
            (on b2 b6)
            (on b3 b13)
            (on b4 b15)
            (on b5 b14)
            (on-table b6)
            (on-table b7)
            (on b8 b5)
            (on b9 b3)
            (on b10 b7)
            (on b11 b2)
            (on b12 b4)
            (on-table b13)
            (on b14 b10)
            (on b15 b8)
        )
    )
)