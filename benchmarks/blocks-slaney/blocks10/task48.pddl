(define (problem BW-10-7268-48)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 - block)
    (:init
        (handempty)
        (on b1 b7)
        (on b2 b9)
        (on b3 b1)
        (on b4 b10)
        (on-table b5)
        (on b6 b2)
        (on b7 b5)
        (on-table b8)
        (on-table b9)
        (on b10 b8)
        (clear b3)
        (clear b4)
        (clear b6)
    )
    (:goal
        (and
            (on b1 b3)
            (on b2 b5)
            (on b3 b4)
            (on b4 b2)
            (on b5 b9)
            (on b6 b10)
            (on b7 b1)
            (on b8 b7)
            (on-table b9)
            (on b10 b8)
        )
    )
)