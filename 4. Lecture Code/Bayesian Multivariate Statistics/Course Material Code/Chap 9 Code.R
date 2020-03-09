# 9.1 Good King Markov and His island Kindom
library(rstan)
library(rethinking)

#Rcode9.1
num_weeks <- 1e5
positions <- rep(0,num_weeks)
current <- 10
for (i in 1:num_weeks) {
  # record current position
  positions[i] <- current

  # flip coin to generate proposal
  proposal <- current + sample(c(-1,1),size=1)
  # now make sure he loops around the archipelago
  if (proposal < 1) proposal <- 10
  if (proposal > 10) proposal <- 1
  
  # move?
  prob_move <- proposal / current
  current <- ifelse(runif(1) < prob_move, proposal, current)
}


# 9.2 Metropolis, Gibbs, and Sadness
#Rcode9.2
D <- 10
T <- 1e3
Y <- rmvnorm(T,rep(0,D),diag(D))
rad_dist <- function(Y) sqrt(sum(Y^2))
rd <- sapply(1:T, function(i) rad_dist( Y[i, ]))
dens(rd)


# 9.4 Easy HMC: ULAM
#Rcode9.9
data(rugged)
d <- rugged; rm(rugged)
d$log_gdp <- log(d$rgdppc_2000)
dd <- d[complete.cases(d$rgdppc_2000), ]
dd$log_gdp_std <- dd$log_gdp / mean(dd$log_gdp)
dd$rugged_std <- dd$rugged / max(dd$rugged)
dd$cid <- ifelse(dd$cont_africa==1, 1, 2)

#Rcode9.10
m8.5 <- quap(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a[cid] + b[cid]*(rugged_std - 0.215),
  a[cid] ~ dnorm(1,0.1),
  b[cid] ~ dnorm(0,0.3),
  sigma ~ dexp(1)
), data = dd)
precis(m8.5, depth = 2)

#9.4.1 Preparation
#Rcode9.11
dat_slim <- list(log_gdp_std = dd$log_gdp_std,
                 rugged_std = dd$rugged_std,
                 cid = as.integer(dd$cid))
str(dat_slim)

#9.4.2 Sampling from the posterior
m9.1 <- ulam(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a[cid] + b[cid] * (rugged_std - 0.215),
  a[cid] ~ dnorm(1, 0.1),
  b[cid] ~ dnorm(0, 0.3),
  sigma ~ dexp(1)
), data = dat_slim, chains=4, cores = 4)

stancode(m9.1)

precis(m9.1, depth = 2)

#9.4.3 Sampling again, in parallel
m9.1a <- ulam(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a[cid] + b[cid] * (rugged_std - 0.215),
  a[cid] ~ dnorm(1, 0.1),
  b[cid] ~ dnorm(0, 0.3),
  sigma ~ dexp(1)
), data = dat_slim, chains=4, cores = 4, iter=1000)

show(m9.1a)

precis(m9.1a, depth = 2)

#9.4.4 Visualization
pairs(m9.1)

#9.4.5 Checking the chain
traceplot(m9.1)

#9.5 Care and feeding of your Markov Chain
#9.5.3 Taming a wild chain
y <- c(-1,1)
set.seed(11)
m9.2 <- ulam(alist(
  y ~ dnorm(mu, sigma),
  mu <- alpha,
  alpha ~ dnorm(0, 1000),
  sigma ~ dexp(0.0001)
), data = list(y=y), chains = 2)

precis(m9.2)
pairs(m9.2@stanfit)
traceplot(m9.2)




set.seed(11)
m9.3 <- ulam(alist(
  y ~ dnorm(mu, sigma),
  mu <- alpha,
  alpha ~ dnorm(1,10),
  sigma ~ dexp(1)
), data = list(y=y), chains = 2)
precis(m9.3)
traceplot(m9.3)

#9.5.4 Non-identifiable parameters
set.seed(41)
y <- rnorm(100, mean = 0, sd=1)

m9.4 <- ulam(alist(
  y ~ dnorm(mu, sigma),
  mu <- a1 + a2,
  a1 ~ dnorm(0, 1000),
  a2 ~ dnorm(0, 1000),
  sigma ~ dexp(1)
), data = list(y=y), chains = 2)
precis(m9.4)
traceplot(m9.4)


m9.5 <- ulam(alist(
  y ~ dnorm(mu, sigma),
  mu <- a1 + a2,
  a1 ~ dnorm(0,10),
  a2 ~ dnorm(0,10),
  sigma ~ dexp(1)
), data = list(y=y), chains = 2)
precis(m9.5)
traceplot(m9.5)
