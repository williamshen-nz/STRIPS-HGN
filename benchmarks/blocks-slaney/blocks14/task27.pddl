(define (problem BW-14-9843-27)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b12)
        (on b3 b13)
        (on b4 b6)
        (on-table b5)
        (on b6 b5)
        (on-table b7)
        (on b8 b14)
        (on b9 b4)
        (on b10 b9)
        (on-table b11)
        (on-table b12)
        (on b13 b1)
        (on b14 b2)
        (clear b3)
        (clear b7)
        (clear b8)
        (clear b10)
        (clear b11)
    )
    (:goal
        (and
            (on b1 b3)
            (on b2 b1)
            (on b3 b13)
            (on b4 b6)
            (on b5 b2)
            (on b6 b10)
            (on b7 b9)
            (on-table b8)
            (on-table b9)
            (on b10 b12)
            (on b11 b4)
            (on b12 b7)
            (on-table b13)
            (on b14 b5)
        )
    )
)