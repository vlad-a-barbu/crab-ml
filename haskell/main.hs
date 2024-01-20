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

main :: IO()
main = do
  print $ cost [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]] [6.11, 4.67] (-2.37)
 
