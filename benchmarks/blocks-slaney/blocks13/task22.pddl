(define (problem BW-13-2654-22)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 - block)
    (:init
        (handempty)
        (on b1 b13)
        (on-table b2)
        (on b3 b10)
        (on b4 b2)
        (on b5 b7)
        (on-table b6)
        (on-table b7)
        (on b8 b6)
        (on b9 b8)
        (on-table b10)
        (on b11 b5)
        (on b12 b1)
        (on b13 b3)
        (clear b4)
        (clear b9)
        (clear b11)
        (clear b12)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b11)
            (on b3 b9)
            (on b4 b1)
            (on b5 b7)
            (on-table b6)
            (on-table b7)
            (on b8 b4)
            (on b9 b12)
            (on b10 b3)
            (on-table b11)
            (on-table b12)
            (on-table b13)
        )
    )
)