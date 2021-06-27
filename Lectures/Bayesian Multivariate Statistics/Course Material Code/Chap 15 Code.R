# Chap 15 Missing Data and Other Opportunities
# 15.1 Measurement Error
library(rethinking)
data("WaffleDivorce")
d <- WaffleDivorce

# points
plot(d$Divorce ~ d$MedianAgeMarriage, ylim=c(4,15),
     xlab='Median age marriange', ylab='Divorce Rate')
# Standard Errors
for (i in 1:nrow(d)){
  ci <- d$Divorce[i] + c(-1,1)*d$Divorce.SE[i]
  x <- d$MedianAgeMarriage[i]
  lines(c(x,x),ci)
}

# 15.1.1 Error on the outcome
dlist <- list(
  D_obs = standardize(d$Divorce),
  D_sd = d$Divorce.SE / sd(d$Divorce),
  M = standardize(d$Marriage),
  A = standardize(d$MedianAgeMarriage),
  N = nrow(d)
)

m15.1 <- ulam(
  alist(
    D_obs ~ dnorm(D_true, D_sd),
    vector[N]:D_true ~ dnorm(mu, sigma),
    mu <- a + bA * A + bM * M,
    a ~ dnorm(0, 0.2),
    c(bA, bM) ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = dlist, chains = 4, cores = 4
)

precis(m15.1, depth=2)

# 15.1.2 Error on both outcome and predictor

dlist <- list(
  D_obs = standardize(d$Divorce),
  D_sd = d$Divorce.SE / sd(d$Divorce),
  M_obs = standardize(d$Marriage),
  M_sd = d$Marriage.SE / sd(d$Marriage),
  A = standardize(d$MedianAgeMarriage),
  N = nrow(d)
)

m15.2 <- ulam(
  alist(
    D_obs ~ dnorm(D_true, D_sd),
    vector[N]:D_true ~ dnorm(mu, sigma),
    mu <- a + bA * A + bM * M_true[i],
    M_obs ~ dnorm(M_true, M_sd),
    vector[N]:M_true ~ dnorm(0,1),
    a ~ dnorm(0,0.2),
    bA ~ dnorm(0,0.5),
    bM ~ dnorm(0,0.5),
    sigma ~ dexp(1)
    ), data = dlist, chains = 4, cores = 4
)

post <- extract.samples(m15.2)
D_true <- apply(post$D_true, 2 ,mean)
M_true <- apply(post$M_true, 2, mean)
plot(dlist$M_obs, dlist$D_obs, pch=16, col=rangi2,
     xlab='marriage rate (std)', ylab = 'divorce rate (std)')
points(M_true, D_true)
for (i in 1:nrow(d))
  lines(c(dlist$M_obs[i],M_true[i]), c(dlist$D_obs[i],D_true[i]))

# 15.2 Missing Data
# 15.2.1 Bayesian Imputation
library(rethinking)
data(milk)
d <- milk
d$neocortex.prop <- d$neocortex.perc / 100
d$logmass <- log(d$mass)

dat <- list(
  K = standardize(d$kcal.per.g),
  B = standardize(d$neocortex.prop),
  M = standardize(d$logmass)
)

m15.3 <- ulam(
  alist(
    K ~ dnorm(mu, sigma),
    mu <- a + bB * B + bM * M,
    B ~ dnorm(nu, sigma_B),
    c(a, nu) ~ dnorm(0,0.5),
    c(bB, bM) ~ dnorm(0, 0.5),
    sigma_B ~ dexp(1),
    sigma ~ dexp(1)
  ), data = dat, chains = 4, cores = 4
)

precis(m15.3, depth=2)

# without missing value
obs_idx <- which(!is.na(d$neocortex.prop))
dat_obs <- list(
  K = dat$K[obs_idx],
  B = dat$B[obs_idx],
  M = dat$M[obs_idx]
)
m15.4 <- ulam(
  alist(
    K ~ dnorm(mu, sigma),
    mu <- a + bB * B + bM * M,
    B ~ dnorm(nu, sigma_B),
    c(a,nu) ~ dnorm(0, 0.5),
    c(bB, bM) ~ dnorm(0, 0.5),
    sigma_B ~ dexp(1),
    sigma ~ dexp(1)
  ), data = dat_obs, chains = 4, cores = 4
)

plot(coeftab(m15.3, m15.4), pars = c('bB','bM'))


post <- extract.samples(m15.3)
B_impute_mu <- apply(post$B_impute, 2, mean)
B_impute_ci <- apply(post$B_impute , 2, PI)

# B vs K
plot(dat$B, dat$K, pch=16, col=rangi2,
     xlab='neocortex percent (std)', ylab='kcal milk (std)')
miss_idx <- which(is.na(dat$B))
Ki <- dat$K[miss_idx]
points(B_impute_mu, Ki)
for ( i in 1:12 ) lines(B_impute_ci[,i], rep(Ki[i],2))

# M vs B
plot(dat$M, dat$B, pch=16, col=rangi2,
     ylab='neocortex percent (std)', xlab='log body mass (std)')
Mi <- dat$M[miss_idx]
points(Mi, B_impute_mu)
for(i in 1:12) lines(rep(Mi[i],2),B_impute_ci[,i])

# 15.2.2 Improving the imputation model

m15.5 <- ulam(alist(
  # K as a function of B and M
  K ~ dnorm(mu, sigma),
  mu <- a + bB * B_merge + bM * M,
  
  # M and B correlation
  MB ~ multi_normal(c(muM,muB),Rho_BM,Sigma_BM),
  matrix[29,2]:MB <<- append_col(M, B_merge),
  
  # define B_merge as mix of observed and imputed values
  vector[29]:B_merge <- merge_missing(B, B_impute),
  
  # priors
  c(a,muB,muM) ~ dnorm(0, 0.5),
  c(bB, bM) ~ dnorm(0, 0.5),
  sigma ~ dexp(1),
  Rho_BM ~ dlkjcorr(2),
  Sigma_BM ~ dexp(1)
), data = dat, chains = 4, cores = 4)

precis(m15.5, depth=3, pars=c('bM','bB','Rho_BM'))
