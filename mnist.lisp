;;; MNIST data set:
;;;    http://yann.lecun.com/exdb/mnist
;;; Download training/test sets images and labels. Place in a ./data
;;;    Overview of idx3 filetype (see MNIST cite for more details):

;;;    [offset] [type]          [value]          [description]
;;;    0000     32 bit integer  0x00000803(2051) magic number
;;;    0004     32 bit integer  60000            number of images
;;;    0008     32 bit integer  28               number of rows
;;;    0012     32 bit integer  28               number of columns
;;;    0016     unsigned byte   ??               pixel
;;;    0017     unsigned byte   ??               pixel
;;;    ........
;;;    xxxx     unsigned byte   ??               pixel

;;; Applies ReLU function to atom or list
(defun relu (x)
  (cond ((null x) ())
        ((listp x) (cons (relu (car x)) (relu (cdr x))))
        ((atom x) (max 0 x))
        (t (cons (max 0 (car x)) (relu (cdr x))))))

;; Reshape a matrix into a matrix with dimensions n and m
(defun reshape (mat m n)
  ;; Flatten a matrix into an array
  (defun flatten (mat)
    (cond ((null mat) nil)
	  ((listp mat) (append (flatten (car mat)) (flatten (cdr mat))))
	  (t (list mat))))
  ;; Reshape a list into a matrix with dimensions n and m
  (defun reshape-list (l m n)
    (cond ((or (= m 0) (null l)) nil)
	  (t (cons (subseq l 0 n) (reshape-list (nthcdr n l) (- m 1) n)))))
  (let ((arr (flatten mat)))
    (reshape-list arr m n)))

;;; Load first MNIST image as a list
(defun load-mnist-image (path)
  (defun load-mnist-data (path)
    (let ((data (open path :element-type '(unsigned-byte 8))))
      (dotimes (i #x12) ;; Don't load magic number, number of images, rows and columns
	(read-byte data))
      (loop for i from 0 to (* 28 28)
	    collect (read-byte data))))
  (reshape (load-mnist-data path) 28 28))

;; Lists for testing
(defvar l1 '((1 2 3 -10) (10 11 -15 -18) (-1 0 1 5) (1 1 1 -11)))
(defvar l2 '((1 10 -5) (-10 11 -15 0) ((10 5) (1 3 4) -1)))
(defvar l3 '((1 2 3 4) (5 6 7 8)))
(defvar macbook-path "data/train-images-idx3-ubyte")
(defvar macmini-path "Documents/mnist/data/train-images-idx3-ubyte")
(defvar img (load-mnist-image macbook-path))
(defvar img (load-mnist-image macmini-path))
