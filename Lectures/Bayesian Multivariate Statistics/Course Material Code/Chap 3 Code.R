#3.1. Sampling from a grid-approximate posterior
p_grid <- seq(from = 0, to = 1,length.out = 1000)
prob_p <- rep(1,1000)
prob_data <- dbinom(6, size = 9, prob = p_grid)
posterior <- prob_data * prob_p
posterior <- posterior / sum(posterior)
samples <- sample(p_grid, prob = posterior, size=1e4, replace=TRUE)
plot(samples)
library(rethinking)

dens(samples)
plot(samples)


#3.2. Sampling to summarize
#3.2.1 Intervals of defined boundaries
sum(posterior[p_grid < 0.5])
sum(samples < 0.5) / 1e4

#3.2.2 Intervals of defined mass
quantile(samples, 0.8)
quantile(samples, c(0.1,0.9))

p_grid = seq(from = 0, to = 1, length.out = 1000)
prior = rep(1,1e3)
likelihood <- dbinom(3, size=3, prob=p_grid)
posterior <- prior * likelihood
posterior <- posterior / sum(posterior)
samples <- sample(p_grid, size = 1e4, prob = posterior, replace = TRUE)
PI(samples, 0.5)
HPDI(samples, 0.5)

#3.2.3 Point estimates
p_grid[which.max(posterior)]
chainmode(samples, adj=0.01)
mean(samples)
median(samples)

sum(posterior*abs(0.5 - p_grid))
loss <- sapply(p_grid, function(d) sum(posterior*abs(d - p_grid)))
p_grid[which.min(loss)]
# absolute loss leads to the median as the point estimate
# quadratic loss leads to the mean as the point estimate


#3.3 Sampling to simulate prediction
#3.3.1 Dummy data
dbinom(0:2, size = 2, prob=0.7)
rbinom(1, size = 2, prob=.7)
rbinom(10, size = 2, prob=.7)

dummy_w <- rbinom(1e5, size = 2, prob = .7)
table(dummy_w)
table(dummy_w)/length(dummy_w)

dummy_w <- rbinom(1e5, size = 9, prob = .7)
simplehist(dummy_w, xlab = 'dummy water count')

w <- rbinom(1e4, size = 9, prob = .6)
simplehist(w)
w <- rbinom(1e4, size = 9, prob = samples)
simplehist(w)
