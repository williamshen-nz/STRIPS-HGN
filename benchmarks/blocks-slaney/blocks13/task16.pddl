(define (problem BW-13-2654-16)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b5)
        (on b3 b11)
        (on b4 b1)
        (on b5 b3)
        (on b6 b9)
        (on b7 b2)
        (on b8 b12)
        (on-table b9)
        (on b10 b7)
        (on b11 b6)
        (on b12 b13)
        (on b13 b10)
        (clear b4)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b13)
            (on b2 b8)
            (on b3 b2)
            (on b4 b10)
            (on b5 b11)
            (on-table b6)
            (on b7 b3)
            (on b8 b5)
            (on b9 b4)
            (on b10 b6)
            (on-table b11)
            (on-table b12)
            (on b13 b12)
        )
    )
)