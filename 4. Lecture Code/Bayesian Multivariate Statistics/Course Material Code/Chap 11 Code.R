#Chap 11 God Spiked the Integers
#11.1 Binomial Regression
#11.1.1 Logistic Regression: Prosocial chimpanzees
library(rethinking)
library(loo)
library(rstan)
data("chimpanzees")
d <- chimpanzees; rm(chimpanzees)

d$treatment <- 1 + d$prosoc_left + 2*d$condition
xtabs(~treatment + prosoc_left + condition, d)

m11.1 <- quap(alist(
  pulled_left ~ dbinom(1,p),
  logit(p) <- a,
  a ~ dnorm(0,10)
), data = d)

set.seed(1999)
prior <- extract.prior(m11.1, n=1e4)
p <- inv_logit(prior$a)
dens(p, adj=0.1)

m11.2 <- quap(alist(
  pulled_left ~ dbinom(1,p),
  logit(p) <- a + b[treatment],
  a ~ dnorm(0, 1.5),
  b[treatment] ~ dnorm(0,10)
), data= d)
set.seed(1999)
prior <- extract.prior(m11.2, n=1e4)
p <- sapply(1:4, function(k) inv_logit(prior$a + prior$b[,k]))
dens(abs(p[,1] - p[,2]), adj=0.1)


m11.3 <- quap(alist(
  pulled_left ~ dbinom(1,p),
  logit(p) <- a + b[treatment],
  a ~ dnorm(0, 1.5),
  b[treatment] ~ dnorm(0,0.5)
), data= d)
set.seed(1999)
prior <- extract.prior(m11.3, n=1e4)
p <- sapply(1:4, function(k) inv_logit(prior$a + prior$b[,k]))
dens(abs(p[,1] - p[,2]), adj=0.1)
mean(abs(p[,1] - p[,2]))


dat_list <- list(
  pulled_left = d$pulled_left,
  actor = d$actor,
  treatment = as.integer(d$treatment)
)
m11.4 <- ulam(alist(
  pulled_left ~ dbinom(1,p),
  logit(p) <- a[actor] + b[treatment],
  a[actor] ~ dnorm(0,1.5),
  b[treatment] ~ dnorm(0,0.5)
), data = dat_list, chains=4, log_lik = TRUE)
precis(m11.4, depth=2)

post <- extract.samples(m11.4)
p_left <- inv_logit(post$a)
plot(precis(as.data.frame(p_left)), xlim=c(0,1))

labs <- c("R/N","L/N","R/P","L/P")
plot(precis(m11.4, depth=2, pars='b'), labels = labs)

pl <- by(d$pulled_left, list(d$actor, d$treatment), mean)
pl[1,]

plot(NULL, xlim=c(1,28), ylim=c(0,1), xlab='', ylab='proportion left lever', xaxt='n', yaxt='n')
axis(2, at=c(0,0.5,1), labels = c(0,0.5,1))
abline(h=0.5, lty=2)
for (j in 1:7) abline( v=(j-1)*4+4.5, lwd=0.5)
for (j in 1:7) text((j-1)*4+2.5, 1.1, concat("actor",j), xpd=TRUE)
for (j in (1:7)[-2]) {
  lines((j-1)*4+c(1,3), pl[j,c(1,3)], lwd=2, col=rangi2)
  lines((j-1)*4+c(2,4), pl[j,c(2,4)], lwd=2, col=rangi2)
}
points(1:28, t(pl), pch=16, col='white', cex=1.7)
points(1:28, t(pl), pch=c(1,1,16,16), col=rangi2, lwd=2)
yoff <- 0.01
text(1, pl[1,1]-yoff, "R/N", pos=1, cex=0.8)
text(2, pl[1,2]+yoff, "L/N", pos=3, cex=0.8)
text(3, pl[1,3]-yoff, "R/P", pos=1, cex=0.8)
text(4, pl[1,4]+yoff, "L/P", pos=3, cex=0.8)
mtext("observed proportions\n")

dat <- list(actor=rep(1:7, each=4), treatment = rep(1:4, times=7))
p_post <- link_ulam(m11.4, data=dat)
p_mu <- apply(p_post, 2, mean)
p_ci <- apply(p_post, 2, PI)

d$side <- d$prosoc_left + 1
d$cond <- d$condition +1
dat_list2 <- list(
  pulled_left = d$pulled_left,
  actor = d$actor,
  side = as.integer(d$side),
  cond = as.integer(d$cond)
)
m11.5 <- ulam(alist(
  pulled_left ~ dbinom(1,p),
  logit(p) <- a[actor] + bs[side] + bc[cond],
  a[actor] ~ dnorm(0,1.5),
  bs[side] ~ dnorm(0,0.5),
  bc[cond] ~ dnorm(0,0.5)
), data = dat_list2, chains = 4, log_lik = TRUE)


m11.5b <- ulam(alist(
  pulled_left ~ dbinom(1,p),
  logit(p) <- a[actor] + bs[side] + bc[cond],
  a[actor] ~ dnorm(0,1.5),
  bs[side] ~ dnorm(mu,sigma_bs),
  bc[cond] ~ dnorm(0,sigma_bc),
  mu ~ dnorm(0,1),
  sigma_bs ~ dexp(1),
  sigma_bc ~ dexp(1)
), data = dat_list2, chains = 4, log_lik = TRUE)

rethinking::compare(m11.4, m11.5, m11.5b)



# 11.1.2 Relative shark and absolute penguin
post <- extract.samples(m11.4)
mean(exp(post$b[,4]-post$b[,2]))

# 11.1.3 Aggregated binomial: Chimpanzees again, condensed
data("chimpanzees")
d <- chimpanzees; rm(chimpanzees)
d$treatment <- as.integer(1 + d$prosoc_left + 2*d$condition)
d$side <- as.integer(d$prosoc_left + 1)
d$cond <- as.integer(d$condition + 1)
d_aggregated <- aggregate(d$pulled_left, 
                          list(treatment = d$treatment, actor = d$actor,
                               side = d$side, cond = d$cond),
                          sum)
colnames(d_aggregated)[5] <- "left_pulls"

dat <- with(d_aggregated, list(
  left_pulls = left_pulls,
  treatment = treatment,
  actor = actor, 
  side = side,
  cond = cond
))

m11.6 <- ulam(
  alist(
    left_pulls ~ dbinom(18,p),
    logit(p) <- a[actor] + b[treatment],
    a[actor] ~ dnorm(0,1.5),
    b[treatment] ~ dnorm(0,0.5)
), data = dat, chains = 4, log_lik = TRUE)
compare(m11.6, m11.4, func=LOO)
compare(m11.6, m11.4, func=WAIC)

-2*dbinom(6,9,0.2,log=TRUE)
-2*sum(dbern(c(1,1,1,1,1,1,0,0,0),0.2,log=TRUE))

( k<- LOOPk(m11.6))

# 11.1.4 Aggregated binomial: Graduate school admissions
library(rethinking)
data("UCBadmit")
d <- UCBadmit; rm(UCBadmit)
str(d)

d$gid <- ifelse(d$applicant.gender == 'male', 1, 2)
m11.7 <- quap(alist(
  admit ~ dbinom(applications, p),
  logit(p) <- a[gid],
  a[gid] ~ dnorm(0,1.5)
), data = d)
precis(m11.7, depth = 2)

post <- extract.samples(m11.7)
diff_a <- post$a[,1] - post$a[,2]
diff_p <- inv_logit(post$a[,1]) - inv_logit(post$a[,2])
precis(list(diff_a = diff_a, diff_p = diff_p))

postcheck(m11.7, n=1e4)
d$dept_id <- rep(1:6, each=2)
for (i in 1:6) {
  x <- 1 + 2*(i-1)
  y1 <- d$admit[x]/d$applications[x]
  y2 <- d$admit[x+1] / d$applications[x+1]
  lines(c(x,x+1), c(y1,y2), col=rangi2, lwd=2)
  text(x+0.5, (y1+y2)/2+0.05, d$dept[x], cex=0.8, col=rangi2)
}


# INCLUDING THE DIFFERENCE OF THE DEPARTMENT

d$dept_id <- rep(1:6, each=2)
m11.8 <- quap(alist(
  admit ~ dbinom(applications, p),
  logit(p) ~ a[gid] + delta[dept_id],
  a[gid] ~ dnorm(0,1.5),
  delta[dept_id] ~ dnorm(0,1.5)
), data = d)
precis(m11.8, depth = 2)

post <- extract.samples(m11.8)
diff_a <- post$a[,1] - post$a[,2]
diff_p <- inv_logit(post$a[,1]) - inv_logit(post$a[,2])
precis(list(diff_a = diff_a, diff_p = diff_p))

pg <- sapply(1:6, function(k)
  d$applications[d$dept_id==k] / sum(d$applications[d$dept_id==k]))
rownames(pg) <- c("male",'female')
colnames(pg) <- unique(d$dept)
round(pg, 2)

postcheck(m11.8, n=1e4)
d$dept_id <- rep(1:6, each=2)
for (i in 1:6) {
  x <- 1 + 2*(i-1)
  y1 <- d$admit[x]/d$applications[x]
  y2 <- d$admit[x+1] / d$applications[x+1]
  lines(c(x,x+1), c(y1,y2), col=rangi2, lwd=2)
  text(x+0.5, (y1+y2)/2+0.05, d$dept[x], cex=0.8, col=rangi2)
}

# 11.2 Poisson regression
# 11.2.1 Example: Oceanic tool complexity
data("Kline")
d <- Kline; rm(Kline)
d$P <- scale(log(d$population))
d$contact_id <- ifelse(d$contact == 'high',2,1)
curve(dlnorm(x, 0,10), from=0, to = 100, n = 200)

N <- 100
a <- rnorm(N, 3, 1.5)
b <- rnorm(N, 0, 10)
plot(NULL, xlim=c(-2,2), ylim=c(0,100))
for (i in 1:N) curve(exp(a[i] + b[i]*x), add=T, col=col.alpha("black",0.5))

set.seed(10)
N <- 100
a <- rnorm(N,3,0.5)
b <- rnorm(N,0,0.2)
plot(NULL, xlim=c(-2,2), ylim=c(0,100))
for (i in 1:N) curve(exp(a[i] + b[i]*x), add=T, col=col.alpha("black",0.5))

x_seq <- seq(from = log(100), to = log(200000), length.out = 100)
lambda <- sapply(x_seq, function(x) exp(a+b*x))
plot(NULL, xlim=range(x_seq), ylim=c(0,500), xlab='log population', ylab='total tool')
for (i in 1:N) lines(x_seq, lambda[i,], col=col.alpha("black",0.5), lwd=1.5)

plot(NULL, xlim=range(exp(x_seq)), ylim=c(0,500), xlab='log population', ylab='total tool')
for (i in 1:N) lines(exp(x_seq), lambda[i,], col=col.alpha("black",0.5), lwd=1.5)


dat <- list(
  T = d$total_tools,
  P = d$P,
  cid = as.integer(d$contact_id))
# Intercept only model
m11.9 <- ulam(alist(
  T ~ dpois(lambda),
  log(lambda) <- a,
  a ~ dnorm(3,0.5)
), data = dat, chains = 4, cores= 4, log_lik = T)
# Interaction model
m11.10 <- ulam(alist(
  T ~ dpois(lambda),
  log(lambda) <- a[cid] + b[cid]*P,
  a[cid] ~ dnorm(3, 0.5),
  b[cid] ~ dnorm(0, 0.2)
), data = dat, chains = 4, cores = 4, log_lik = T)

compare(m11.9, m11.10)

k <- LOOPk(m11.10)
plot(dat$P, dat$T, xlab = 'log population (std)', ylab = 'total tools',
     col = rangi2, pch = ifelse(dat$cid == 1, 1 , 16), lwd=2,
     ylim=c(0,75), cex = 1 + normalize(k))
# Set up horizontal axis values to compute predictions 
ns <- 100
P_seq <- seq(from=-1.4, to = 3, length.out = ns)
# Prediction for low contact
lambda <- link(m11.10, data = data.frame(P=P_seq, cid=1))
lmu <- apply(lambda, 2 , mean)
lci <- apply(lambda,2 , PI)
lines(P_seq, lmu, lty=2, lwd=1.5)
shade(lci, P_seq, xpd=F)
# Predictions for high contact
lambda <- link(m11.10, data = data.frame(P=P_seq, cid=2))
lmu <- apply(lambda, 2 , mean)
lci <- apply(lambda,2 , PI)
lines(P_seq, lmu, lty=1, lwd=1.5)
shade(lci, P_seq, xpd=F)

plot(d$population, d$total_tools, xlab = 'log population (std)', ylab = 'total tools',
     col = rangi2, pch = ifelse(dat$cid == 1, 1 , 16), lwd=2,
     ylim=c(0,75), cex = 1 + normalize(k))
# Set up horizontal axis values to compute predictions 
ns <- 100
P_seq <- seq(from=-5, to = 3, length.out = ns)
pop_seq <- exp(P_seq*sd(log(d$population)) + mean(log(d$population)))
# Prediction for low contact
lambda <- link(m11.10, data = data.frame(P=P_seq, cid=1))
lmu <- apply(lambda, 2 , mean)
lci <- apply(lambda,2 , PI)
lines(pop_seq, lmu, lty=2, lwd=1.5)
shade(lci, pop_seq, xpd=F)
# Predictions for high contact
lambda <- link(m11.10, data = data.frame(P=P_seq, cid=2))
lmu <- apply(lambda, 2 , mean)
lci <- apply(lambda,2 , PI)
lines(pop_seq, lmu, lty=1, lwd=1.5)
shade(lci, pop_seq, xpd=F)


# Model based
dat2 <- list(T=d$total_tools, P=d$population, cid=as.integer(d$contact_id))
m11.11 <- ulam(
  alist(
    T ~ dpois(lambda),
    lambda <- exp(a[cid]) * P^b[cid]/g,
    a[cid] ~ dnorm(1,1),
    b[cid] ~ dexp(1),
    g ~ dexp(1)
), data = dat2, chains = 4, cores = 4, log_lik = T)


# 11.2.3 Example: Expousre and the offset.
num_days <- 30
y <- rpois(num_days, 1.5)
num_weeks <- 4
y_new <- rpois(num_weeks, 0.5*7)

y_all <- c(y, y_new)
exposure <- c(rep(1,30), rep(7,4))
monastery <- c(rep(0,30), rep(1,4))
d <- data.frame(y = y_all, days = exposure, monastery = monastery)

d$log_days = log(d$days)
m11.12 <- ulam(alist(
  y ~ dpois(lambda),
  log(lambda) <- log_days + a + b*monastery,
  a ~ dnorm(0,1),
  b ~ dnorm(0,1)
), data = d, chains = 4, cores = 4)

post <- extract.samples(m11.12)
lambda_old <- exp(post$a)
lambda_new <- exp(post$a + post$b)
precis(data.frame(lambda_old, lambda_new))


# 11.3 Censoring and survival
data("AustinCats")
d <- AustinCats; rm(AustinCats)

d$adopt <- ifelse(d$out_event=='Adoption', 1L, 0L)
dat <- list(
  days_to_event = as.numeric(d$days_to_event),
  color_id = ifelse(d$color == 'Black', 1L, 2L),
  adopted = d$adopt
)

m11.14 <- ulam(alist(
  days_to_event|adopted == 1 ~ exponential(lambda),
  days_to_event|adopted == 0 ~ custom(exponential_lccdf(!Y | lambda)),
  lambda <- 1.0 / mu,
  log(mu) <- a[color_id],
  a[color_id] ~ normal(0,1)
), data = dat, chains = 4, cores = 4)

precis(m11.14, 2)

# Average mean time of adoption
post <- extract.samples(m11.14)
post$D <- exp(post$a)
precis(post, 2)
