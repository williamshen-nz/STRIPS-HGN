(define (problem BW-11-6452-17)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 - block)
    (:init
        (handempty)
        (on b1 b5)
        (on b2 b7)
        (on-table b3)
        (on-table b4)
        (on b5 b6)
        (on b6 b11)
        (on b7 b3)
        (on b8 b10)
        (on b9 b2)
        (on b10 b1)
        (on b11 b4)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b6)
            (on b2 b8)
            (on-table b3)
            (on b4 b2)
            (on b5 b9)
            (on b6 b4)
            (on b7 b3)
            (on b8 b5)
            (on b9 b11)
            (on b10 b1)
            (on b11 b7)
        )
    )
)