;;; Solving MNIST from scratch
;;;
;;; MNIST data set:
;;;    http://yann.lecun.com/exdb/mnist
;;;
;;; Download training/test sets for images and labels. Place in ./data
;;;
;;;    Overview of idx3 filetype (see MNIST cite for more details):
;;;
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

;;; Initialize an n-element list with init-val
(defun init-list-aux (l init-val n)
  (if (= n 0)
      l
      (init-list-aux (cons init-val l) init-val (- n 1))))
(defun init-list (init-val n)
  (init-list-aux () init-val n))

;;; Calculate LogSumExp
(defun lse (x)
  (let ((c (reduce #'max x))
	(sum-list (lambda (l) (reduce #'+ l)))
	(exp-list (lambda (l) (mapcar #'exp l)))
	(minus-lists (lambda (l1 l2) (mapcar #'- l1 l2))))
    (+ c (log (funcall sum-list
		       (funcall exp-list
				(funcall minus-lists
					 (flatten x)
					 (init-list c (length (flatten x))))))))))
					 
;;; Initialize an m x n matrix with random weights
(defun init-rand-weights (m n)
  (defun random-from-range (start end)
    (+ start (random (+ (- end start)))))
  (defun init-rand-iter (m n)
    (cond ((= n 0) nil)
	  (t (cons (/ (random-from-range -1.0 1.0) (* m n))
		   (init-rand-iter m (- n 1))))))
  (cond ((= m 0) nil)
	(t (cons (init-rand-iter m n)
		 (init-rand-weights (- m 1) n)))))

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

;;; Predicate to test if a matrix is a square matrix
(defun squarep (mat)
  (= (first (size mat)) (second (size mat))))

;;; Get row, indexing from zero
(defun get-row (n mat)
  (nth n mat))

;;; Get column, indexing from zero
(defun get-col (n mat)
  (if (null mat)
      nil
      (cons (nth n (car mat)) (get-col n (cdr mat)))))

;;; Get ith and jth element from a matrix
(defun get-elem (i j mat)
  (nth j (get-row i mat)))

;;; Scalar product two lists together, returns NULL if lists are
;;; not of equal length
(defun dot (l1 l2)
  (cond ((not (= (length l1) (length l2))) nil)
	((null l1) 0)
	(t (+ (* (car l1) (car l2)) (dot (cdr l1) (cdr l2))))))

;;; Multiply two matrices, returns NULL if either list is not a
;;; matrix or inner dimensions don't match
(defun matmul (m1 m2)
  (defun matmul-iter-cols (m1 m2 m n)
    (cond ((= n (second (size m2))) nil)
	  (t (cons (dot (get-row m m1) (get-col n m2))
		   (matmul-iter-cols m1 m2 m (+ n 1))))))
  (defun matmul-iter-rows (m1 m2 m)
    (cond ((= m (first (size m1))) nil)
	  (t (cons (matmul-iter-cols m1 m2 m 0)
		   (matmul-iter-rows m1 m2 (+ m 1))))))
  (let ((m1-column-length (second (size m1)))
	(m2-row-length (first (size m2))))
    (cond ((or (not (matrixp m1)) (not (matrixp m2))) nil)
	  ((not (= m1-column-length m2-row-length)) nil)
	  (t (matmul-iter-rows m1 m2 0)))))))

;;; Swap in place the ith and jth elements of a list
(defun swap-in-place (i j l)
  (rotatef (nth i l) (nth j l)))

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

;;; Forward pass
(defun forward (x l1 l2 act)
  (matmul (funcall act (matmul x l1)) l2))

;;; Backward pass
(defun backward  

;;; Load nth MNIST image as a matrix
(defun nth-image (n path)
  ;;; Load byte data from MNIST file
  (defun load-data (path)
    (let ((data (open path :element-type '(unsigned-byte 8))))
      (dotimes (i #x12) ;; Don't load magic number, number of images, rows and columns
	(read-byte data))
      (dotimes (i (* n (* 28 28)))
	(read-byte data))
      (loop for i from 0 to (* 28 28)
	    collect (read-byte data))))
  (reshape (load-data path) 28 28))

;;; Load random image from MNIST training set of 60,000 images
(defun random-image (path)
  (nth-image (random 60000) path))

;;; Load nth label from MNIST label training set
(defun nth-label (n path)
  (let ((data (open path :element-type '(unsigned-byte 8))))
    (dotimes (i #x8)
      (read-byte data))
    (dotimes (i (* n))
      (read-byte data))
    (read-byte data)))

;;; Draw image in terminal with 1's and 0's
(defun draw (img)
  (loop for row in img
	do (loop for num in row
		 do (if (> num 0)
			(write 1)
			(write 0)))
	   (terpri)))

;;; Load nth image and label from MNIST training set
(defun nth-image-and-label (n image-path label-path)
  (list (nth-image n image-path) (nth-label n label-path)))

;;; Load random image and label from MNIST training set
(defun random-image-and-label (image-path label-path)
  (nth-image-and-label (random 60000) image-path label-path))

;; Lists for testing
(defvar l1 '((1 2 3 -10) (10 11 -15 -18) (-1 0 1 5) (1 1 1 -11)))
(defvar l2 '((1 10 -5) (-10 11 -15 0) ((10 5) (1 3 4) -1)))
(defvar l3 '((1 2 3 4) (5 6 7 8)))
(defvar la '(1 2 6 8))
(defvar lb '(9 10 15 12))
(defvar m1 '((1 2) (-1 5)))
(defvar m2 '((4 1) (2 2)))
(setq m1 '((1 2 4) (-1 5 5)))
(setq m2 '((4 1 5 9) (2 2 8 10) (1 -1 -2 )))
(defvar layer1 (init-rand-weights 784 128))
(defvar layer2 (init-rand-weights 128 10))
(defvar image-path "data/train-images-idx3-ubyte")
(defvar image-path "Documents/mnist/data/train-images-idx3-ubyte")
(defvar label-path "data/train-labels-idx1-ubyte")
(defvar label-path "Documents/mnist/data/train-images-idx3-ubyte")
(defvar data (load-mnist-data image-path))
(defvar img (load-mnist-nth-image 1 image-path))
(setq img (load-mnist-nth-image 100 image-path))

;; Matrices for training
(setq l1 (init-rand-weights 784 128))
(setq l2 (init-rand-weights 128 10))
(defvar x (reshape (random-image image-path) 1 (* 28 28)))
(setq x '(-0.06594814 -0.05582632 -0.06729704 -0.02831291  0.0
          -0.040037   -0.05968991 -0.03470023 -0.03858397 -0.02367364))
