# Chapter 12 Monsters and Mixtures
# 12.1 Over-dispersed outcomes
# 12.1.1 Beta-binomial
library(rethinking)
library(loo)
library(rstan)

pbar <- 0.5
theta <- 5
curve(dbeta2(x, pbar, theta), from = 0, to = 1,
      xlab='probability', ylab='Density')

data("UCBadmit")
d <- UCBadmit; rm(UCBadmit)
d$gid <- ifelse(d$applicant.gender == 'male', 1L, 2L)
dat <- list(A = d$admit, N = d$applications, gid = d$gid)
m12.1 <- ulam(alist(
  A ~ dbetabinom(N, pbar, theta),
  logit(pbar) <- a[gid],
  a[gid] ~ dnorm(0,1.5),
  theta ~ dexp(1)
), data = dat, chains =4, cores = 4 )

post <- extract.samples(m12.1)
post$da <- post$a[,1] - post$a[,2]
precis(post, depth = 2)

gid <- 2
# draw posterior mean beta distribution
curve(dbeta2(x, mean(logistic(post$a[,gid])), mean(post$theta)), from=0, to = 1, ylab='density',
      xlab='probability admit', ylim=c(0,3), lwd=2)
# draw 50 beta distributions sampled from the posterior
for (i in 1:50){
  p <- logistic(post$a[i,gid])
  theta <- post$theta[i]
  curve(dbeta2(x,p,theta), add=TRUE, col=col.alpha('black',0.2))
}
mtext('distribution of female admission rates')

postcheck(m12.1)

# 12.1.2 Negative-binomial or gamma-Poisson
data("Kline")
d <- Kline; rm(Kline)
d$P <- standardize(log(d$population))
d$contact_cid <- ifelse(d$contact =='high', 2L, 1L)

dat2 <- list(
  T = d$total_tools,
  P = d$population,
  cid = d$contact_cid
)

m12.3 <- ulam(alist(
  T ~ dgampois(lambda, phi),
  lambda <- exp(a[cid]) * P^b[cid] / g,
  a[cid] ~ dnorm(1,1),
  b[cid] ~ dexp(1),
  g ~ dexp(1),
  phi ~ dexp(1)
), data = dat2, chains = 4, cores= 4, log_lik = T)

# 12.1.3 Over-dispersion, entropy, and information criteria


# 12.2 Zero-inflated Outcomes
# 12.2.1 Example: Zero-inflated Poisson

# Define parameters
prob_drink <- 0.2
rate_work <- 1 #avg manuscript
N <- 365
set.seed(365)
drink <- dbinom(N, 1, prob_drink) # Sim monk drink
y <- (1-drink)*rpois(N, rate_work) # Sim manuscripts completed
simplehist(y, xlab='manuscripts completed', lwd=4)
zeros_drink <- sum(drink)
zeros_work <- sum(y==0 & drink == 0)
zeros_total <- sum(y==0)
lines(c(0,0), c(zeros_work,zeros_total), lwd=4, col=rangi2)

m12.4 <- ulam(alist(
  y ~ dzipois(p, lambda),
  logit(p) <- ap,
  log(lambda) <- al,
  ap ~ dnorm(-1.5,1),
  al ~ dnorm(1, 0.5)
), data = list(y=as.integer(y)), chains = 4, cores = 4)
precis(m12.4)
inv_logit(-2.87)
exp(0.07)

# 12.3 Ordered categorical outcomes
# 12.3.1 Moral intuition
data("Trolley")
d<- Trolley ; rm(Trolley)

# 12.3.2 Describing an ordered distribution with intercepts
simplehist(d$response, xlim=c(1,7), xlab='response')
pr_k <- table(d$response) / nrow(d)
cum_pr_k <- cumsum(pr_k)
plot(1:7, cum_pr_k, type='b', xlab='response', ylab='cumulative proportion', ylim=c(0,1))

logit <- function(x) log(x/(1-x))
( lco <- logit(cum_pr_k))


m12.5 <- ulam(
  alist(
    R ~ dordlogit(0, cutpoints),
    cutpoints ~ dnorm(0, 1.5)
), data = list(R=d$response), chains = 4, cores = 4)
precis(m12.5, depth = 2)
inv_logit(coef(m12.5))
stancode(m12.5)

# 12.3.3 Adding predictior variables
( pk <- dordlogit(1:7, 0, coef(m12.5)))
sum(pk*(1:7))
( pk <- dordlogit(1:7, 0, coef(m12.5)-0.5))
sum(pk*(1:7))

dat <- list(
  R = d$response,
  A = d$action,
  I = d$intention,
  C = d$contact
)

m12.6 <- ulam(alist(
  R ~ dordlogit(phi, cutpoints),
  phi <- bA * A + bC * C + BI * I,
  BI <- bI + bIA * A + bIC * C,
  c(bA, bC, bI, bIA, bIC) ~ dnorm(0, 0.5),
  cutpoints ~ dnorm(0,1.5)
), data = dat, cores = 4, chains = 4)

precis(m12.6)
plot(precis(m12.6))

plot(NULL, type='n', xlab='intention', ylab='probability',
     xlim=c(0,1), ylim=c(0,1), xaxp=c(0,1,1), yaxp=c(0,1,2))
kA <- 0 ; kC <- 0 ; kI <- 0:1
pdat <- data.frame(A=kA, C=kC, I=kI)
phi <- link(m12.6, data=pdat)$phi
for (s in 1:50){
  pk <- pordlogit(1:6, phi[s,], post$cutpoints[s,])
  for ( i in 1:6 ) lines(kI, pk[,i], col=col.alpha('black',0.1))
}

kA <- 0 ; kC <- 1 ; kI <- 0:1
pdat <- data.frame(A=kA, C=kC, I=kI)
s <- sim(m12.6, data=pdat)
simplehist(s, xlab = 'response')


# 12.4 Ordered categorical predictors
levels(d$edu)
edu_levels <- c(6,1,8,4,7,2,5,3)
d$edu_new <- edu_levels[d$edu]

library(gtools)
set.seed(1805)
delta <- rdirichlet(10, alpha = rep(2,7))
str(delta)

h <- 3
plot(NULL, xlim=c(1,7), ylim=c(0,0.4), xlab='index',ylab='probability')
for (i in 1:nrow(delta)) lines(1:7, delta[i,], type='b',
                               pch=ifelse(i==h, 16,1), lwd=ifelse(i==h,4,1.5),
                               col=ifelse(i==h,'black',col.alpha('black',0.7)))

dat <- list(
  R = d$response,
  action = d$action,
  intention = d$intention,
  contact = d$contact,
  E = as.integer(d$edu_new),
  alpha = rep(2,7)
)

m12.5 <- ulam(
  alist(
    R ~ ordered_logistic(phi, kappa),
    phi <- bE*sum(delta_j[1:E]) + bA * action + bI * intention + bC * contact,
    kappa ~ dnorm(0,1.5),
    c(bA,bI,bE,bC) ~ dnorm(0,1),
    vector[8]: delta_j <<- append_row(0, delta),
    simplex[7]: delta ~ dirichlet(alpha)
  ), data= dat, chains = 4, cores = 4)

precis(m12.5, depth = 2, omit='kappa')
traceplot(m12.5, omit='kappa')

delta_labels <- c("Elem","MidSch","SHS","HSG",'SCol','Bach','Mast','Grad')
pairs(m12.5, pars='delta', labels=delta_labels)

dat$edu_norm <- normalize(d$edu_new)

m12.6 <- ulam(alist(
  y ~ ordered_logistic(mu, cutpoints),
  mu <- bE*edu_norm + bA * action + bI * intention + bC * contact,
  c(bA,bE,bI,bC) ~ normal(0,1),
  cutpoints ~ normal(0,1.5)
), data = dat, chains = 4, cores = 4)

