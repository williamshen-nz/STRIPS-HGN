(define (problem BW-14-9843-39)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b8)
        (on-table b3)
        (on b4 b1)
        (on-table b5)
        (on b6 b4)
        (on b7 b5)
        (on b8 b6)
        (on b9 b13)
        (on-table b10)
        (on b11 b9)
        (on b12 b11)
        (on b13 b10)
        (on-table b14)
        (clear b2)
        (clear b3)
        (clear b7)
        (clear b12)
        (clear b14)
    )
    (:goal
        (and
            (on b1 b9)
            (on-table b2)
            (on b3 b13)
            (on-table b4)
            (on b5 b8)
            (on b6 b5)
            (on b7 b6)
            (on b8 b12)
            (on b9 b2)
            (on b10 b4)
            (on b11 b1)
            (on b12 b10)
            (on b13 b11)
            (on-table b14)
        )
    )
)