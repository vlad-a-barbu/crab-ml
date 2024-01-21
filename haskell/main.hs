model :: Floating a => [a] -> [a] -> a -> a
model x w b = sum (zipWith (*) x w) + b

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

forward :: Floating a => (a -> a) -> [a] -> [a] -> a -> a
forward act x w b = act $ model x w b

dist :: Floating a => [a] -> a -> (a -> a) -> [a] -> a
dist w b act (y:xs) = (forward act xs w b - y) ** 2
  
cost :: Floating a => [[a]] -> [a] -> a -> (a -> a) -> a
cost td w b act = sum (map (dist w b act) td) / fromIntegral (length td)

replace [] _ = []
replace (_:xs) (0,a) = a:xs
replace (x:xs) (n,a) = if n < 0 then x:xs else x: replace xs (n-1,a)
 
graddesc :: (Floating a, Integral b) => [[a]] -> [a] -> a -> (a -> a) -> a -> a -> a -> b -> ([a], a)
graddesc td w b act c eps lr (-1) = (w, b - (cost td w (b + eps) act - c) / eps * lr)
graddesc td w b act c eps lr wi = do
  let w' = replace w (wi, (w !! fromIntegral wi) + eps)
  let dc = (cost td w' b act - c) / eps
  let w'' = replace w (wi, (w !! fromIntegral wi) - dc * lr)
  graddesc td w'' b act c eps lr (wi - 1)

strw :: (Floating a, Show a) => [a] -> String
strw [] = "[]"
strw xs = "[" ++ go xs ++ "]"
 where
   go [x] = show x
   go (x:xs) = show x ++ "," ++ go xs

logstep :: (Floating a, Show a) => [a] -> a -> a -> IO ()
logstep w b c = do
  putStrLn $ "w = " ++ strw w ++ "; b = " ++ show b ++ "; c = " ++ show c ++ ";"

train :: (Floating a, Integral b, Show a) => [[a]] -> [a] -> a -> (a -> a) -> a -> a -> b -> IO([a], a)
train td w b act eps lr 0 = return (w, b)
train td w b act eps lr n = do
  let c = cost td w b act
  logstep w b c
  let (w', b') = graddesc td w b act c eps lr (fromIntegral $ length w - 1)
  train td w' b' act eps lr (n - 1)

f x = [x * 3, x]
generate n = map f [1..n]

main :: IO()
main = do
  (w, b) <- train
    (generate 10)
    [1.0]
    1.0
    id
    0.0001 0.0001
    100000
  let fw = forward id
  print $ fw [11.0] w b
  
