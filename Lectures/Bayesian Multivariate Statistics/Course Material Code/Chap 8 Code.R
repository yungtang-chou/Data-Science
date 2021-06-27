library(rethinking)
library(rstan)

# 8.1 Building an interaction
# 8.1.1 Making two models

# R code 8.1
data('rugged')
d <- rugged
d$log_gdp <- log(d$rgdppc_2000)
dd <- d[complete.cases(d$rgdppc_2000),]

dd$log_gdp_std <- dd$log_gdp / mean(dd$log_gdp)
dd$rugged_std <- dd$rugged / max(dd$rugged)

d.A1 <- dd[dd$cont_africa == 1,] # Africa
d.A0 <- dd[dd$cont_africa == 0,] # Not Africa

# R code 8.2
m8.1 <- quap(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a + b * (rugged_std - 0.215),
  a ~ dnorm(1,1),
  b ~ dnorm(0,1),
  sigma ~ dexp(1)
), data = d.A1)

# R code 8.3
set.seed(7)
prior <- extract.prior(m8.1)
plot(NULL, xlim=c(0,1),ylim=c(0.5,1.5), xlab='ruggedness', ylab='log GDP')
abline(h=min(dd$log_gdp_std), lty=2)
abline(h=max(dd$log_gdp_std), lty=2)

rugged_seq <- seq(from = -0.1, to = 1.1, length.out = 30)
mu <- link(m8.1, post=prior, data = data.frame(rugged_std = rugged_seq))
for ( i in 1:50) lines(rugged_seq, mu[i,], col=col.alpha("black",0.3))

# R code 8.4
sum(abs(prior$b) > 0.6) / length(prior$b)

# R code 8.5
m8.1 <- quap(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a + b*(rugged_std - 0.215),
  a ~ dnorm(1,0.1),
  b ~ dnorm(0,0.3),
  sigma ~ dexp(1)
), data = d.A1)

# R code 8.6
m8.2 <- quap(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a + b*(rugged_std - 0.215),
  a ~ dnorm(1,0.1),
  b ~ dnorm(0,0.25),
  sigma ~ dexp(1)
), data = d.A0)


# 8.1.2 Adding an indicator variable doesn't work
# R code 8.7
m8.3 = quap(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a + b*(rugged_std - .215),
  a ~ dnorm(1,.1),
  b ~ dnorm(0,.3),
  sigma ~ dexp(1)
), data = dd)

# R code 8.8
# make variable to index Africa (1) or not (2)
dd$cid <- ifelse(dd$cont_africa == 1, 1, 2)

# R code 8.9 - 8.10
m8.4 <- quap(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a[cid] + b*(rugged_std - 0.215),
  a[cid] ~ dnorm(1,.1),
  b ~ dnorm(0,.3),
  sigma ~ dexp(1)
), data = dd)

compare(m8.3, m8.4)

dat = list(
  cid = as.integer(dd$cid),
  rugged_std = dd$rugged_std,
  N = NROW(dd)
)

# R code 8.11
precis(m8.4)
precis(m8.4, depth = 2)

# R code 8.12
rugged.seq <- seq(from=-0.1, to = 1.1, length.out = 30)
mu.NotAfrica <- link(m8.4, data = data.frame(cid=2, rugged_std=rugged.seq))
mu.Africa <- link(m8.4, data = data.frame(cid=1, rugged_std = rugged.seq))

mu.NotAfrica_mu <- apply(mu.NotAfrica, 2, mean)
mu.NotAfrica_ci <- apply(mu.NotAfrica, 2, PI, prob=.97)
mu.Africa_mu <- apply(mu.Africa,2, mean)
mu.Africa_ci <- apply(mu.Africa,2, PI, prob=0.97)

# 8.1.3 Adding an interaction does work.
# R code 8.13 - 8.16
m8.5 <- quap(alist(
  log_gdp_std ~ dnorm(mu, sigma),
  mu <- a[cid] + b[cid] * (rugged_std - 0.215),
  a[cid] ~ dnorm(1,0.1),
  b[cid] ~ dnorm(0,.3),
  sigma ~ dexp(1)
), data = dd)

precis(m8.5, depth = 2)

compare(m8.3, m8.4, m8.5)

waic_list <- WAIC(m8.5, pointwise = TRUE)

# 8.1.4 Plotting the interaction
# R code 8.17

par(mfcol=c(1,2))

#plot africa with all data
plot(d.A1$rugged_std, d.A1$log_gdp_std, pch=16, col=rangi2, 
     xlab = 'ruggedness (standardized)', ylab = 'log GDP (as proportion of mean', 
     xlim = c(0,1))
mu <- link(m8.5, data =data.frame(cid =1, rugged_std = rugged_seq))
mu_mean <- apply(mu, 2, mean)
mu_ci <- apply(mu, 2, PI, prob=.97)
lines(rugged_seq, mu_mean, lwd=2)
shade(mu_ci, rugged_seq, col=col.alpha(rangi2, .3))
mtext("African nations")

#plot nonafrica with all data
plot(d.A0$rugged_std, d.A0$log_gdp_std, pch=1, col='black',
     xlab = 'ruggedness (standardized)', ylab= 'log GDP (as proportion of mean)',
     xlim = c(0,1))
mu <- link(m8.5, data=data.frame(cid=2, rugged_std = rugged_seq))
mu_mean <- apply(mu,2,mean)
mu_ci <- apply(mu,2, PI, prob=.97)
lines(rugged_seq, mu_mean, lwd=2)
shade(mu_ci, rugged_seq)
mtext("Non-African nations")

# 8.2 Symmetry of interactions
# R code 8.18
rugged_seq <- seq(from=-0.2, to=1.2, length.out = 30)
muA <- link(m8.5, data=data.frame(cid=1, rugged_std = rugged_seq))
muN <- link(m8.5, data =data.frame(cid=2, rugged_std = rugged_seq))
delta <- muA - muN


# 8.3 Continuous interactions
# 8.3.1 A winter flower
# R code 8.19
library(rethinking)
data(tulips)
d <- tulips
str(d)

# 8.3.2 The models
# R code 8.20
d$blooms_std <- d$blooms / max(d$blooms)
d$water_cent <- d$water - mean(d$water)
d$shade_cent <- d$shade - mean(d$shade)


# R code 8.23
m8.6 <- quap(alist(
  blooms_std ~ dnorm(mu, sigma),
  mu <- a + bw * water_cent + bs * shade_cent,
  a ~ dnorm(.5,.25),
  bw ~ dnorm(0, .25),
  bs ~ dnorm(0, .25),
  sigma ~ dexp(1)
), data = d)

# R code 8.24
m8.7 <- quap(alist(
  blooms_std ~ dnorm(mu, sigma),
  mu <- a + bw*water_cent + bs * shade_cent + bws*water_cent*shade_cent,
  a ~ dnorm(0.5,0.25),
  bw ~ dnorm(0,0.25),
  bs ~ dnorm(0,0.25),
  bws ~ dnorm(0,0.25),
  sigma ~ dexp(1)
), data = d)

# 8.3.3 Plotting posterior predictions
# R code 8.25
par(mfrow=c(1,3))
for (s in -1:1){
  idx <- which(d$shade_cent == s)
  plot(d$water_cent[idx], d$blooms_std[idx], xlim=c(-1,1), ylim=c(0,1),
       xlab='Water', ylab='blooms', pch=16, col=rangi2)
  mu <- link(m8.6, data=data.frame(shade_cent=s, water_cent=-1:1))
  for (i in 1:20) lines( -1:1, mu[i, ], col=col.alpha("black",0.3))
}


# 8.3.4 Plotting prior predictions
# R code 8.26
set.seed(7)
prior <- extract.prior(m8.6)
