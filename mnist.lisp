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

;;; Initialize an n x m matrix with random weights
(defun init-rand-weights (n m)
  (defun init-rand-iter (n)
    (cond ((= n 0) nil)
	  (t (cons (random 1.0)
		   (init-rand-iter (- n 1))))))
  (cond ((= m 0) nil)
	(t (cons (init-rand-iter (- n 1))
		 (init-rand-weights n (- m 1))))))

;;; Checks if a list is a matrix. A list is considered a matrix when
;;; the number of entries in each row is the same. For example:
;;;   ((1 2 3) (4 5 6)) is a matrix
;;;   ((1 2) (1)) is not
(defun matrixp (m)
  (if (null (cdr m))
      (length (car m))
      (and (= (length (car m))
	      (length (cadr m)))
	   (matrixp (cdr m)))))

;;; Get the size of the matrix, returns NULL if not a matrix
(defun size (m)
  (if (matrixp m)
      (list (length m) (length (car m)))
      nil))

;;; Get row, indexing from zero
(defun get-row (n mat)
  (nth n mat))

;;; Get column, indexing from zero
(defun get-col (n mat)
  (if (null mat)
      nil
      (cons (nth n (car mat)) (get-col n (cdr mat)))))

;;; Scalar product two lists together, returns NULL if lists are
;;; not of equal length
(defun dot (l1 l2)
  (cond ((not (= (length l1) (length l2))) nil)
	((null l1) 0)
	(t (+ (* (car l1) (car l2)) (dot (cdr l1) (cdr l2))))))

;;; Predicate to test if a matrix is a square matrix
(defun squarep (mat)
  (= (first (size mat)) (second (size mat))))
  
;;; Multiply two matrices, returns NULL if either list is not a
;;; matrix or inner dimensions don't match
(defun matmul (mat m2)
  (defun matmul-iter (m1 m2)
    )
  (let ((m1-column-length (second (size m1)))
	(m2-row-length (first (size m2))))
    (cond ((or (not (matrixp m1)) (not (matrixp m2))) nil)
	  ((not (= m1-column-length m2-row-length)) nil)
	  (t (matmul-iter m1 m2)))))

;;; Reshape a matrix into a matrix with dimensions n and m
(defun reshape (mat n m)
  ;;; Flatten a matrix into an array
  (defun flatten (mat)
    (cond ((null mat) nil)
	  ((listp mat) (append (flatten (car mat)) (flatten (cdr mat))))
	  (t (list mat))))
  ;;; Reshape a list into a matrix with dimensions n and m
  (defun reshape-list (l n m)
    (cond ((or (= n 0) (null l)) nil)
	  (t (cons (subseq l 0 m) (reshape-list (nthcdr n l) (- n 1) m)))))
  (let ((arr (flatten mat)))
    (reshape-list arr n m)))

;;; Load first MNIST image as a list
(defun load-mnist-image (path)
  ;;; Load byte data from MNIST file
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
(defvar la '(1 2 6 8))
(defvar lb '(9 10 15 12))
(defvar macbook-path "data/train-images-idx3-ubyte")
(defvar macmini-path "Documents/mnist/data/train-images-idx3-ubyte")
(defvar data (load-mnist-data macmini-path))
(defvar img (load-mnist-image macmini-path))
