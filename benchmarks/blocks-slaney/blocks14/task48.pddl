(define (problem BW-14-9843-48)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 - block)
    (:init
        (handempty)
        (on b1 b11)
        (on b2 b14)
        (on b3 b9)
        (on b4 b5)
        (on-table b5)
        (on b6 b3)
        (on b7 b6)
        (on b8 b7)
        (on-table b9)
        (on b10 b4)
        (on b11 b2)
        (on b12 b8)
        (on-table b13)
        (on-table b14)
        (clear b1)
        (clear b10)
        (clear b12)
        (clear b13)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b1)
            (on b4 b10)
            (on b5 b9)
            (on b6 b2)
            (on b7 b12)
            (on b8 b11)
            (on b9 b8)
            (on-table b10)
            (on b11 b7)
            (on b12 b14)
            (on b13 b5)
            (on b14 b3)
        )
    )
)