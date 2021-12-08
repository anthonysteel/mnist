;;; Mnist data set:
;;;    http://yann.lecun.com/exdb/mnist/
(defvar l1 '((1 2 3 -10) (10 11 -15 -18) (-1 0 1 5) (1 1 1 -11)))
(defvar l2 '((1 10 -5) (-10 11 -15 0) ((10 5) (1 3 4) -1)))

;;; Applies ReLU function to atom or list
(defun relu (x)
  (cond ((null x) ())
        ((listp x) (cons (relu (car x)) (relu (cdr x))))
        ((atom x) (max 0 x))
        (t (cons (max 0 (car x)) (relu (cdr x))))))

