(define (problem BW-11-6452-46)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 - block)
    (:init
        (handempty)
        (on b1 b8)
        (on b2 b1)
        (on b3 b6)
        (on b4 b9)
        (on-table b5)
        (on b6 b11)
        (on b7 b4)
        (on b8 b7)
        (on b9 b10)
        (on-table b10)
        (on b11 b2)
        (clear b3)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b10)
            (on b2 b4)
            (on b3 b8)
            (on b4 b7)
            (on b5 b11)
            (on-table b6)
            (on-table b7)
            (on b8 b9)
            (on b9 b6)
            (on b10 b5)
            (on b11 b2)
        )
    )
)