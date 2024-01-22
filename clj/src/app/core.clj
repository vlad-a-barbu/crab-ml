(ns app.core
  (:require [clojure.math :as math]))

(defn model [x w b]
  (+ (reduce + (map * x w)) b))

(defn sigmoid [x]
  (/ 1 (+ 1 (math/exp (* -1 x)))))

(defn cost [td w b]
  (/ (reduce +
             (map (fn [tr]
                 (math/pow (-
                            (model (take-while (fn [x] (not= x :y)) tr) w b)
                            (last tr))
                           2))
                  td)) (* 1.0 (count td))))

