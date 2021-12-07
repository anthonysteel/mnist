;;; MNIST data set:
;;;    http://yann.lecun.com/exdb/mnist/
(define l1 '((1 2 3 -10) (10 11 -15 -18) (-1 0 1 5) (1 1 1 -11)))
(define l2 '((1 10 -5) (-10 11 -15 0) ((10 5) (1 3 4) -1)))

(define (atom? x) (not (pair? x)))

;;; Applies ReLU function to atom or list
(define (relu x)
  (cond ((null? x) ())
        ((list? x) (cons (relu (car x)) (relu (cdr x))))
        ((atom? x) (max 0 x))
        (else (cons (max 0 (car x)) (relu (cdr x))))))
