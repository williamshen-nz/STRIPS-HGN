(define (problem BW-11-6452-36)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 - block)
    (:init
        (handempty)
        (on-table b1)
        (on-table b2)
        (on b3 b1)
        (on b4 b9)
        (on b5 b4)
        (on b6 b8)
        (on b7 b11)
        (on b8 b5)
        (on b9 b7)
        (on b10 b2)
        (on b11 b3)
        (clear b6)
        (clear b10)
    )
    (:goal
        (and
            (on b1 b6)
            (on b2 b1)
            (on b3 b7)
            (on b4 b5)
            (on b5 b8)
            (on-table b6)
            (on b7 b10)
            (on-table b8)
            (on b9 b3)
            (on b10 b4)
            (on b11 b2)
        )
    )
)