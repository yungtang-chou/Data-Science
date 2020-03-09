# Chapter 13 Models with Memory
library(rstan)
library(rethinking)
library(loo)

# 13.1 Example: Multilevel Tadpoles

data(reedfrogs)
d <- reedfrogs ; rm(reedfrogs)
str(d)

d$tank <- 1:nrow(d)
dat <- list(
  S = d$surv,
  N = d$density,
  tank = d$tank
)

m13.1 <- ulam(alist(
  S ~ dbinom(N,p),
  logit(p) <- a[tank],
  a[tank] ~ dnorm(0,1.5)
), data = dat, chains = 4, cores = 4, log_lik = T)

precis(m13.1, depth = 2)

m13.2 <- ulam(alist(
  S ~ dbinom(N,p),
  logit(p) <- a[tank],
  a[tank] ~ dnorm(abar,sigma),
  abar ~ dnorm(0,1.5),
  sigma ~ dexp(1)
), data = dat, chains = 4, cores = 4, log_lik = T)

rethinking::compare(m13.1, m13.2)

# extract stan sample
post <- extract.samples(m13.2)
d$propsurv.est <- logistic(apply(post$a, 2 , mean))

# display raw proportions surviving in each tank
plot (d$propsurv, ylim=c(0,1), pch=16, xant='n',
      xlab='tank', ylab='proportion survival', col = rangi2)
axis(1, at=c(1,16,32,48), labels = c(1,16,32,48))

# overlay posterior means
points(d$propsurv.est)

# mark posterior mean probability across tanks
abline(h = mean(inv_logit(post$abar)), lty=2)

# draw vertical dividers between tank densities
abline(v=16.5, lwd=0.5)
abline(v=32.5, lwd=0.5)
text(8,0,'small tanks')
text(16+8, 0, 'medium tanks')
text(32+8, 0, 'large tanks')


# show first 100 populations in the posterior
plot(NULL, xlim=c(-3,4), ylim=c(0,0.35),
     xlab='log-odds survive', ylab='Density')
for (i in 1:100){
  curve(dnorm(x, post$abar[i], post$sigma[i]), add=T,
        col = col.alpha('black',0.2))
}

# sample 8000 imaginary tanks from the posterior distribution
sim_tanks <- rnorm(8000, post$abar, post$sigma)
# transform to probability and visualize
dens(inv_logit(sim_tanks), lwd=2, adj=0.1)

# 13.2 Varying efects and the underfitting/overfitting trade-off
# 13.2.1 The model
abar <- 1.5
sigma <- 1.5
nponds <- 60
Ni <- as.integer(rep(c(5,10,25,35), each=15))

set.seed(5005)
a_pond <- rnorm(nponds, mean = abar, sd=sigma)
dsim <- data.frame(pond = 1:nponds, Ni=Ni, true_a = a_pond)

# 13.2.2 Simulate survivors
dsim$Si <- rbinom(nponds, prob = logistic(dsim$true_a), size = dsim$Ni)

# 13.2.3 Compute the no-pooling estimates
dsim$p_nopool <- dsim$Si / dsim$Ni

# 13.2.4 Compute partial-pooling estimates
dat <- list(
  Si = dsim$Si,
  Ni = dsim$Ni,
  pond = dsim$pond
)

m13.3 <- ulam(
  alist(
    Si ~ dbinom(Ni, p),
    logit(p) <- a_pond[pond],
    a_pond[pond] ~ dnorm(a_bar, sigma),
    a_bar ~ dnorm(0, 1.5),
    sigma ~ dexp(1)
  ), data = dat, chains = 4, cores = 4)

precis(m13.3, depth = 2)

post <- extract.samples(m13.3)
dsim$p_partpool <- apply(inv_logit(post$a_pond), 2, mean)
dsim$p_true <- inv_logit(dsim$true_a)

nopool_error <- abs(dsim$p_nopool - dsim$p_true)
partpool_error <- abs(dsim$p_partpool - dsim$p_true)

plot(1:60, nopool_error, xlab='pond', ylab='absolute error',
     col = rangi2, pch=16)
points(1:60, partpool_error)

# 13.3 More than one type of cluster
data("chimpanzees")
d <- chimpanzees; rm(chimpanzees)
d$treatment <- 1 + d$prosoc_left + 2*d$condition

dat_list <- list(
  pulled_left = d$pulled_left,
  actor = d$actor,
  block_id = d$block,
  treatment = as.integer(d$treatment)
)

set.seed(13)
m13.4 <- ulam(
  alist(
    pulled_left ~ dbinom(1,p),
    logit(p) <- a[actor] + g[block_id] + b[treatment],
    b[treatment] ~ dnorm(0, 0.5),
    a[actor] ~ dnorm(a_bar, sigma_a),
    g[block_id] ~ dnorm(0, sigma_g),
    a_bar ~ dnorm(0, 1.5),
    sigma_a ~ dexp(1),
    sigma_g ~ dexp(1)
  ), data = dat_list, chains = 4, cores = 4, log_lik = T
)

precis(m13.4, depth = 2)
plot(precis(m13.4, depth = 2))

m13.5 <- ulam(
  alist(
    pulled_left ~ dbinom(1,p),
    logit(p) <- a[actor] + b[treatment],
    b[treatment] ~ dnorm(0, 0.5),
    a[actor] ~ dnorm(a_bar, sigma_a),
    a_bar ~ dnorm(0,1.5),
    sigma_a ~ dexp(1)
  ), data = dat_list, cores = 4, chains = 4, log_lik = T
)

rethinking::compare(m13.4, m13.5)

# 13.3.2 Even more clusters
m13.6 <- ulam(
  alist(
    pulled_left ~ dbinom(1,p),
    logit(p) <- a[actor] + g[block_id] + b[treatment],
    b[treatment] ~ dnorm(0, sigma_b),
    a[actor] ~ dnorm(a_bar, sigma_a),
    g[block_id] ~ dnorm(0, sigma_g),
    a_bar ~ dnorm(0, 1.5),
    sigma_b ~ dexp(1),
    sigma_a ~ dexp(1),
    sigma_g ~ dexp(1)
  ), data = dat_list, chains = 4, cores = 4, log_lik = T
)
coeftab(m13.4, m13.6)


# 13.4 Divergent transitions and non-centered priors
# 13.4.1 Adjust target acceptance
divergent(m13.4)
set.seed(13)
m13.4b <- ulam(m13.4, cores = 4, chains = 4, control = list(adapt_delta=0.99))
divergent(m13.4b)

# 13.4.2 Reparameterization

set.seed(13)
m13.4nc <- ulam(
  alist(
    pulled_left ~ dbinom(1,p),
    logit(p) <- a_bar + z[actor]*sigma_a + x[block_id]*sigma_g + b[treatment],
    b[treatment] ~ dnorm(0, 0.5),
    z[actor] ~ dnorm(0, 0.5),
    x[block_id] ~ dnorm(0, 1),
    a_bar ~ dnorm(0, 0.5),
    sigma_a ~ dexp(1),
    sigma_g ~ dexp(1)
  ), data = dat_list, chains = 4, cores = 4
)
neff_c <- precis(m13.4, depth = 2)[['n_eff']]
neff_nc <- precis(m13.4nc, depth = 2)[['n_eff']]
par_names <- rownames(precis(m13.4, depth = 2))
neff_table <- cbind(neff_c, neff_nc)
rownames(neff_table) <- par_names
round(t(neff_table))

precis(m13.4, depth=2)
precis(m13.4nc, depth=2)
# 13.5 Multilevel posterior predictions
# 13.5.1 Posterior prediction for same clusters
chimp <- 2
d_pred <- list(
  actor = rep(chimp,4),
  treatment = 1:4,
  block_id = rep(1,4)
)

p <- link(m13.4, data= d_pred)
p_mu <- apply(p, 2, mean)
p_ci <- apply(p, 2, PI)

post <- extract.samples(m13.4)
str(post)
dens(post$a[,5])
p_link <- function(treatment, actor=1, block_id=1){
  logodds <- with(post, a[,actor] + g[,block_id] + b[,treatment])
  return(inv_logit(logodds))
}
p_raw <- sapply(1:4, function(i) p_link(i, actor=2, block_id=1))
p_mu <- apply(p_raw, 2, mean)
p_ci <- apply(p_raw, 2, PI)

# 13.5.2 Posterior prediction for new clusters
p_link_abar <- function(treatment){
  logodds <- with(post, a_bar + b[,treatment])
  return(inv_logit(logodds))
}

p_raw <- sapply(1:4, function(i) p_link_abar(i))
p_mu <- apply(p_raw, 2, mean)
p_ci <- apply(p_raw, 2, PI)
plot(NULL, xlab='treatment', ylab='proportion pulled left',
     ylim=c(0,1), xaxt='n', xlim=c(1,4))
axis(1, at=1:4, labels=c("R/N","L/N","R/P","L/P"))
lines(1:4, p_mu)
shade(p_ci, 1:4)

a_sim <- with(post, rnorm(post$a_bar), a_bar, sigma_a)
p_link_sim <- function(treatment){
  logodds <- with(post, a_sim + b[,treatment])
  return(inv_logit(logodds))
}
p_raw_sim <- sapply(1:4, function(i) p_link_sim(i))

plot(NULL, xlab='treatment', ylab='proportion pulled left',
     ylim=c(0,1), xaxt='n', xlim=c(1,4))
axis(1, at=1:4, labels = c("R/N","L/N","R/P","L/P"))
for (i in 1:100) lines(1:4, p_raw_sim[i,], col=col.alpha("black",0.25))
