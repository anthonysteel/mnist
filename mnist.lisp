;;; Mnist data set:
;;;    http://yann.lecun.com/exdb/mnist
;;; Download training/test sets images and labels. Place in a ./data

(defvar l1 '((1 2 3 -10) (10 11 -15 -18) (-1 0 1 5) (1 1 1 -11)))
(defvar l2 '((1 10 -5) (-10 11 -15 0) ((10 5) (1 3 4) -1)))

;;; Applies ReLU function to atom or list
(defun relu (x)
  (cond ((null x) ())
        ((listp x) (cons (relu (car x)) (relu (cdr x))))
        ((atom x) (max 0 x))
        (t (cons (max 0 (car x)) (relu (cdr x))))))

;;; Load first MNIST image as a list
(defun load-image (path)
  (let ((data (open path :element-type '(unsigned-byte 8))))
    (dotimes (i #x12)
      (read-byte data))
    (loop for i from 0 to (* 28 28)
	  collect (read-byte data))))

