(ns app.core
  (:require [clojure.math :as math]))

(defn model [x w b]
  (+ (reduce + (map * x w)) b))

(defn sigmoid [x]
  (/ 1 (+ 1 (math/exp (* -1 x)))))

(defn forward [x w b act]
  (act (model x w b)))

(defn cost [td w b act]
  (/ (reduce +
             (map (fn [tr]
                 (math/pow (-
                            (forward
                             (take-while (fn [x] (not= x :y)) tr)
                             w b act)
                            (last tr))
                           2))
   td)) (* 1.0 (count td))))

(defn graddesc
  ([td w b act eps lr]
   (graddesc td w b act eps lr (cost td w b act) (dec (count w))))
  ([td w b act eps lr c i]
   (if (= i -1)
     [w (- b (* (/ (- (cost td w (+ b eps) act) c) eps) lr))]
     (recur
      td
      (assoc w i (- (w i) (* lr
                             (/ (- (cost
                                    td
                                    (assoc w i (+ (w i) eps))
                                    b act) c) eps)
                             )))
      b act eps lr c (dec i)))
   ))

(defn train [td wb act eps lr n]
  (if (zero? n)
    wb
    (recur td (graddesc td (wb 0) (wb 1) act eps lr) act eps lr (dec n))))

(def wb
  (train [[1 1 :y 1]
          [1 0 :y 0]
          [0 1 :y 0]
          [0 0 :y 0]]
         [[1.0 1.0] 1.0]
         sigmoid
         1e-1
         1e-1
         100000))

(forward [1 1] (wb 0) (wb 1) sigmoid)
(forward [1 0] (wb 0) (wb 1) sigmoid)
(forward [0 1] (wb 0) (wb 1) sigmoid)
(forward [0 0] (wb 0) (wb 1) sigmoid)

         
   
  

