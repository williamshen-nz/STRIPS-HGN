(define (problem BW-12-9546-4)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 - block)
    (:init
        (handempty)
        (on b1 b10)
        (on b2 b9)
        (on-table b3)
        (on b4 b1)
        (on-table b5)
        (on-table b6)
        (on-table b7)
        (on b8 b2)
        (on b9 b3)
        (on b10 b6)
        (on b11 b4)
        (on b12 b7)
        (clear b5)
        (clear b8)
        (clear b11)
        (clear b12)
    )
    (:goal
        (and
            (on b1 b7)
            (on b2 b9)
            (on b3 b12)
            (on b4 b10)
            (on b5 b8)
            (on-table b6)
            (on b7 b4)
            (on b8 b3)
            (on b9 b1)
            (on b10 b5)
            (on-table b11)
            (on b12 b11)
        )
    )
)