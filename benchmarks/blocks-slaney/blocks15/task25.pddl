(define (problem BW-15-4678-25)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b8)
        (on b3 b6)
        (on-table b4)
        (on b5 b7)
        (on b6 b1)
        (on-table b7)
        (on b8 b4)
        (on b9 b3)
        (on b10 b2)
        (on-table b11)
        (on b12 b15)
        (on-table b13)
        (on b14 b5)
        (on b15 b9)
        (clear b10)
        (clear b11)
        (clear b12)
        (clear b13)
        (clear b14)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b11)
            (on b3 b2)
            (on b4 b13)
            (on-table b5)
            (on b6 b10)
            (on b7 b4)
            (on b8 b5)
            (on b9 b12)
            (on-table b10)
            (on b11 b1)
            (on b12 b6)
            (on b13 b15)
            (on-table b14)
            (on b15 b14)
        )
    )
)