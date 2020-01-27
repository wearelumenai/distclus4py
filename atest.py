from distclus import Streaming, MCMC
from numpy import array
import time


dataset = array([
    [4, 1, 2] for i in range(100)
])

algo = MCMC(
    init_k=2, b=500, amp=1, seed=654126513379
)
# algo = Streaming(
#     mu=0.5,
#     sigma=0.1,
#     outRatio=2,
#     outAfter=7,
#     buffer_size=10000
# )
algo.push(dataset[:10])
# algo.run(rasync=True)
algo.play()
algo.push(dataset[10:])

print(algo.centroids)

# monitoring during 1 seconds since algo is running in a asynchronous mode
for k in range(10):
    print(k)
    print(len(algo.centroids), 'centers', end='|')
    print(k)

algo.stop()
