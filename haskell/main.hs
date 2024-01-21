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
graddesc td w b act c eps lr wi = do
  let w' = replace w (wi, (w !! fromIntegral wi) + eps)
  let dc = (cost td w' b act - c) / eps
  let w'' = replace w (wi, (w !! fromIntegral wi) - dc * lr)
  let db = (cost td w (b + eps) act - c) / eps
  let b' = b - db * lr
  if wi < 1 then (w'', b') else graddesc td w'' b act c eps lr (wi - 1)

train :: (Floating a, Integral b) => [[a]] -> [a] -> a -> (a -> a) -> a -> a -> b -> ([a], a)
train td w b act eps lr n = do
  let c = cost td w b act
  let nw = fromIntegral $ length w
  let (w', b') = graddesc td w b act c eps lr (nw - 1)
  if n < 1 then (w', b') else train td w' b' act eps lr (n - 1)

main :: IO()
main = do
  let (w, b) = train
        [[0.0, 1.0, 1.0],
         [1.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [1.0, 0.0, 0.0]]
        [1.0, 1.0]
        (-1.0)
        sigmoid
        0.1 0.1
        1000000
  let fw = forward sigmoid
  putStrLn ("[1.0, 1.0]:\t" ++ show (fw [1.0, 1.0] w b))
  putStrLn ("[1.0, 0.0]:\t" ++ show (fw [1.0, 0.0] w b))
  putStrLn ("[0.0, 1.0]:\t" ++ show (fw [0.0, 1.0] w b))
  putStrLn ("[0.0, 0.0]:\t" ++ show (fw [0.0, 0.0] w b))
  
