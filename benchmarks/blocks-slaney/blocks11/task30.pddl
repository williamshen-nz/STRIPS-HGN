(define (problem BW-11-6452-30)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 - block)
    (:init
        (handempty)
        (on b1 b5)
        (on-table b2)
        (on-table b3)
        (on b4 b8)
        (on b5 b3)
        (on b6 b11)
        (on-table b7)
        (on-table b8)
        (on b9 b7)
        (on b10 b4)
        (on b11 b2)
        (clear b1)
        (clear b6)
        (clear b9)
        (clear b10)
    )
    (:goal
        (and
            (on b1 b6)
            (on b2 b10)
            (on-table b3)
            (on b4 b1)
            (on b5 b9)
            (on b6 b8)
            (on-table b7)
            (on b8 b5)
            (on-table b9)
            (on-table b10)
            (on-table b11)
        )
    )
)