library(rethinking)
library(tidyverse)
library(tidyr)
#4.1 Why normal distributions are normal
#4.1.1 Normal by addition

pos <- replicate(1000, sum(runif(16,-1,1)))
plot(pos)
simplehist(pos)
hist(pos)

#4.1.2 Normal by multiplication
prod(1+runif(12,0,0.1))
growth <- replicate(10000, prod(1+runif(12,0,0.1)))
dens(growth, norm.comp = TRUE)

big <- replicate(10000, prod(1+runif(12,0,0.5)))
small <- replicate(10000, prod(1+runif(12,0,0.01)))

dens(big, norm.comp = TRUE)
dens(small, norm.comp = TRUE)

#4.1.3 Normal by log-multiplication
log.big <- replicate(10000, log(prod(1+runif(12,0,0.5))))
dens(log.big, norm.comp = TRUE)

#4.3 A Gaussian model of height
#4.3.1 The data
library(rethinking)
data("Howell1")
d <- Howell1
str(d)
glimpse(d)
precis(d)
summary(d)
d2 <- d[d$age >= 18,]

#4.3.2 The model
dens(d2$height)
curve(dnorm(x, 178, 20), from = 100, to = 250)
curve(dnorm(x, 0, 50), from = -10, to = 60)

sample_mu <- rnorm(1e4, 178, 20)
sample_sigma <- runif(1e4, 0, 50)
prior_h <- rnorm(1e4, sample_mu, sample_sigma)
dens(prior_h)

#4.3.3 Grid approximation of the posterior distribution
mu.list <- seq(140,160, length.out = 200)
sigma.list <- seq(4,9, length.out = 200)
post <- expand.grid(mu=mu.list, sigma=sigma.list)
post$LL <- sapply(1:nrow(post), function(i) sum(dnorm(d2$height, mean = post$mu[i], sd = post$sigma[i], log = TRUE)))
post$prod <- post$LL + dnorm(post$mu, 178, 20, TRUE) + dunif(post$sigma, 0, 50, TRUE)
post$prob <- exp(post$prod - max(post$prod))
contour_xyz(post$mu,post$sigma, post$prod)
image_xyz(post$mu, post$sigma, post$prod)

#4.3.4 Sampling from the posterior
sample.rows <- sample(1:nrow(post), size = 1e4, replace = TRUE, prob = post$prob)
sample.mu <- post$mu[sample.rows]
sample.sigma <- post$sigma[sample.rows]
plot(sample.mu, sample.sigma, cex=0.5, pch=16, col=col.alpha(rangi2,0.1))
dens(sample.mu)
dens(sample.sigma)

HPDI(sample.mu)
HPDI(sample.sigma)

d3 <- sample(d2$height, size = 20)
mu.list <- seq(150, 170, length.out = 200)
sigma.list <- seq(4, 20, length.out = 200)
post2 <- expand.grid(mu=mu.list, sigma=sigma.list)
post2$LL <- sapply(1:nrow(post2), function(i) sum(dnorm(d3, mean = post2$mu[i], sd = post2$sigma[i], log=TRUE)))
post2$prod <- post2$LL + dnorm(post2$mu, 178, 20, TRUE) + dnorm(post2$sigma, 0, 50, TRUE)
post2$prob <- exp(post2$prod - max(post2$prod))
sample2.rows <- sample(1:nrow(post2), size = 1e4, replace = TRUE, prob = post2$prob)
sample2.mu <- post2$mu[sample2.rows]
sample2.sigma <- post2$sigma[sample2.rows]
plot (sample2.mu, sample2.sigma, cex=0.5, pch=16, col=col.alpha(rangi2, 0.1), xlab='mu',ylab='sigma')

dens(sample2.sigma, norm.comp = TRUE)

#4.3.5 Finding the posterior distribution with quap

library(rethinking)
data("Howell1")
d <- Howell1
d2 <- d[d$age>=18,]
flist <- alist(
  height ~ dnorm(mu, sigma),
  mu ~ dnorm(178,20),
  sigma ~ dunif(0,50)
)

library(rethinking)
m4.1 <- map(flist, data = d2)
precis(m4.1)

m4.2 <- map(alist(
  height ~ dnorm(mu, sigma),
  mu ~ dnorm(178,0.1),
  sigma ~ dunif(0,50)
), data = d2)
precis(m4.2)

#4.3.6 Sampling from a quap
vcov(m4.1)
diag(vcov(m4.1))
cov2cor(vcov(m4.1))

post <- extract.samples(m4.1, n = 1e4)
class(post)
head(post)
precis(post)
plot(post)

# The code below do the same thing as extract.samples
library(MASS)
post <- mvrnorm(n = 1e4, mu = coef(m4.1), Sigma = vcov(m4.1))
class(post)


#4.4 Adding a predictor
plot(d2$height ~ d2$weight)
#4.4.1 The linera model strategy
set.seed(2971)
N <- 100
a <- rnorm(N, 178, 20)
b <- rnorm(N, 0, 10)
plot(NULL, xlim = range(d2$weight), ylim = c(-100,400), xlab = "weight", ylab = "height")
abline(h = 0, lty = 2)
abline(h = 272, lty = 1, lwd = 0.5)
mtext("b ~ dnorm(0,10)")
xbar <- mean(d2$weight)
for (i in 1:N) curve( a[i] + b[i]*(x-xbar), from = min(d2$weight), to = max(d2$weight), add = TRUE, 
                      col = col.alpha("black",0.2))

b <- rlnorm(1e4, meanlog = 0, sdlog = 1)
dens(b, xlim = c(0,5), adj = 0.1)

set.seed(2971)
N <- 100
a <- rnorm(N, 178, 20)
b <- rlnorm(N, 0, 1)
plot(NULL, xlim = range(d2$weight), ylim = c(-100,400), xlab = "weight", ylab = "height")
abline(h = 0, lty = 2)
abline(h = 272, lty = 1, lwd = 0.5)
mtext("log(b) ~ dnorm(0,1)")
xbar <- mean(d2$weight)
for (i in 1:N) curve( a[i] + b[i]*(x-xbar), from = min(d2$weight), to = max(d2$weight), 
                      add = TRUE, col = col.alpha("black",0.2))

#4.4.2 Finding the posterior distribution
library(rethinking)
data("Howell1")
d <- Howell1
d2 <- d[d$age >=18, ]

xbar <- mean(d2$weight)

m4.3 <- map(flist = alist(
  height ~ dnorm(mu, sigma),
  mu <- a + b*(weight - xbar),
  a ~ dnorm(178, 20),
  b ~ dlnorm(0,1),
  sigma ~ dunif(0, 50)
),
data = d2)

#4.4.3 Interpreting the posterior distribution
#4.4.3.1 Tables of marginal distributions
precis(m4.3)
round(vcov(m4.3), 3)
pairs(m4.3)

#4.4.3.2 Plotting posterior inference against the data
plot(height ~ weight, data = d2, col = rangi2)
post <- extract.samples(m4.3)
a_map <- mean(post$a)
b_map <- mean(post$b)
curve(a_map + b_map*(x-xbar), add=TRUE)

#4.4.3.3 Adding uncertainty around the mean
post <- extract.samples(m4.3)
post[1:5,]

N <- 10
dN <- d2[1:N,]
mN <- map(flist = alist(
  height ~ dnorm(mu, sigma),
  mu <- a + b*(weight - mean(weight)),
  a ~ dnorm(178,20),
  b ~ dlnorm(0,1),
  sigma ~ dunif(0,50 )
), data = dN)

#extract 20 samples from the posterior
post <- extract.samples(mN, n = 20)
#display raw data and sample size
plot(dN$weight, dN$height, xlim = range(d2$weight), ylim = range(d2$height),
     col = rangi2, xlab='weight', ylab='height')
mtext(concat("N = ", N))
#plot the lines, with transparency
for (i in 1:20)
  curve(post$a[i] + post$b[i] * (x - mean(dN$weight)),
        col="black", add= TRUE)

#4.4.3.4 Plotting regression intervals and contours
post <- extract.samples(m4.3)
mu_at_50 <- post$a + post$b * (50 - xbar)
dens(mu_at_50, col = rangi2, lwd=2, xlab="mu|weight=50")
HPDI (mu_at_50, .89)

mu <- link(m4.3)
str(mu)

weight.seq <- seq(from=25, to=70, by=1)
mu <- link(m4.3, data=data.frame(weight = weight.seq))
str(my)

#type="n" to hide row data
plot(height ~ weight, d2, type='n')
for (i in 1:100)
  points(weight.seq, mu[i,],pch=16, col=col.alpha(rangi2, 0.1))

mu.mean <- apply(mu, 2, mean)
mu.HPDI <- apply(mu, 2, HPDI, prob=.89)
plot(height ~ weight, data = d2, col=col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.HPDI, weight.seq)

#4.4.3.5 Prediction intervals
sim.height <- sim(m4.3, data = list(weight = weight.seq))
str(sim.height)
height.PI <- apply(sim.height, 2, PI, prob = .89)
plot(height ~ weight, d2, col=col.alpha(rangi2, .5))
lines(weight.seq, mu.mean)
shade(mu.HPDI, weight.seq)
shade(height.PI, weight.seq)

sim.height <- sim(m4.3, data = list(weight = weight.seq), n=1e4)
height.PI <- apply(sim.height,2,PI,prob=.89)
plot(height ~ weight, d2, col=col.alpha(rangi2, .5))
lines(weight.seq, mu.mean)
shade(mu.HPDI, weight.seq)
shade(height.PI, weight.seq)

#4.5 Curves from the line
#4.5.1 Polynomial regression
library(rethinking)
data("Howell1")
d <- Howell1
str(d)
plot(d$height ~ d$weight)

d$weight_s <- (d$weight - mean(d$weight)) / sd(d$weight)
d$weight_s2 <- d$weight_s^2
m4.5 <- map(flist = alist(
  height ~ dnorm(mu, sigma),
  mu <- a + b1*weight_s + b2*weight_s2,
  a ~ dnorm(178,20),
  b1 ~ dlnorm(0,1),
  b2 ~ dnorm(0,1),
  sigma ~ dunif(0,50)
  ), data = d)
precis(m4.5)

weight.seq <- seq(from = -2.2, to =2, length.out = 30)
pred_dat <- list(weight_s = weight.seq, weight_s2 = weight.seq^2)
mu <- link(m4.5, data = pred_dat)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob=.89)
sim.height <- sim(m4.5, data = pred_dat)
height.PI <- apply(sim.height, 2, PI, prob = .89)

plot(height ~ weight_s, d, col=col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)