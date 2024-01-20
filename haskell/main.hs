model :: Floating a => [a] -> [a] -> a -> a
model x w b = sum (zipWith (*) x w) + b

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

dist :: Floating a => [a] -> a -> [a] -> a
dist w b td = do
  let (y:xs) = td
  let yh = sigmoid $ model xs w b
  let d = yh - y
  d * d
  
cost :: Floating a => [[a]] -> [a] -> a -> a
cost td w b = sum (map (dist w b) td) / fromIntegral (length td)

replace [] _ = []
replace (_:xs) (0,a) = a:xs
replace (x:xs) (n,a) =
  if n < 0
    then (x:xs)
    else x: replace xs (n-1,a)

graddesc :: (Floating a, Integral b) => [[a]] -> [a] -> a -> a -> a -> a -> b -> [a]
graddesc td w b c eps lr i = do
  let n = fromIntegral $ length w
  let w' = replace w (i, (w !! fromIntegral i) + eps)
  let dc = (cost td w' b - c) / eps
  let w'' = replace w (i, (w !! fromIntegral i) - dc * lr)
  if i == n - 1
    then w''
    else graddesc td w'' b c eps lr (i + 1)

train :: (Floating a, Integral b) => [[a]] -> [a] -> a -> a -> a -> b -> ([a], a)
train td w b eps lr n = do
  let c = cost td w b
  let w' = graddesc td w b c eps lr 0
  let db = (cost td w (b + eps) - c) / eps
  let b' = b - db * lr
  if n < 1
    then (w', b')
    else train td w' b' eps lr (n - 1)

main :: IO()
main = do
  let (w, b) = train
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 0.0, 0.0]]
        [6.0, 6.0]
        (-0.9)
        0.1 0.1
        1000000
  putStrLn ("[1.0, 1.0]:\t"++show (model [1.0, 1.0] w b))
  putStrLn ("[1.0, 0.0]:\t"++show (model [1.0, 0.0] w b))
  putStrLn ("[0.0, 1.0]:\t"++show (model [0.0, 1.0] w b))
  putStrLn ("[0.0, 0.0]:\t"++show (model [0.0, 0.0] w b))
  return ()
  
