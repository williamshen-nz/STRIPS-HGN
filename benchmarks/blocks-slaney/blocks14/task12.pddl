(define (problem BW-14-9843-12)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 - block)
    (:init
        (handempty)
        (on b1 b9)
        (on b2 b7)
        (on b3 b5)
        (on b4 b12)
        (on b5 b4)
        (on b6 b1)
        (on b7 b3)
        (on-table b8)
        (on-table b9)
        (on-table b10)
        (on-table b11)
        (on b12 b6)
        (on b13 b14)
        (on b14 b2)
        (clear b8)
        (clear b10)
        (clear b11)
        (clear b13)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b6)
            (on b3 b4)
            (on b4 b5)
            (on b5 b2)
            (on-table b6)
            (on b7 b11)
            (on-table b8)
            (on-table b9)
            (on b10 b8)
            (on b11 b13)
            (on b12 b14)
            (on b13 b3)
            (on-table b14)
        )
    )
)