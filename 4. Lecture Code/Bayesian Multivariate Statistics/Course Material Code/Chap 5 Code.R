#5.1 Spurious association
library(rethinking)
data("WaffleDivorce")
d <- WaffleDivorce

#Standardize variables
d$A <- scale(d$MedianAgeMarriage)
d$D <- scale(d$Divorce)
d$M <- scale(d$Marriage)

sd(d$MedianAgeMarriage)

m5.1 <- map(alist(
  D ~ dnorm(mu, sigma),
  mu <- a + bA * A,
  a ~ dnorm(0, 0.2),
  bA ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = d)

set.seed(10)
prior <- extract.prior(m5.1)
mu <- link(m5.1, post = prior, data = list( A = c(-2,2)))
plot( NULL, xlim = c(-2,2), ylim = c(-2,2))
for (i in 1:50) lines(c(-2,2), mu[i,], col=col.alpha("black",0.4))

A_seq <- seq(from = -3, to =3.2, length.out = 30)
mu <- link(m5.1, data = list(A = A_seq))
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI)
plot(D ~ A, data = d, col = rangi2)
lines(A_seq, mu.mean, lwd=2)
shade(mu.PI, A_seq)
precis(m5.1)

m5.2 <- quap(flist = alist(
  D ~ dnorm(mu, sigma),
  mu <- a + bM*M,
  a ~ dnorm(0, 0.2),
  bM ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = d)


library(dagitty)
dag5.1 <- dagitty("dag {
  A -> D
  A -> M
  M -> D
}")
coordinates(dag5.1) <- list (x=c(A=0, D=1, M=2), y=c(A=0, D=1, M=0))
plot(dag5.1)

#5.1.3 Approximating the posterior
m5.3 <- quap(flist = alist(
  D ~ dnorm(mu, sigma),
  mu <- a + bM*M + bA*A,
  a ~ dnorm(0,0.2),
  bM ~ dnorm(0, 0.5),
  bA ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = d)
precis(m5.3)

plot(coeftab(m5.1, m5.2, m5.3), par = c("bA", 'bM'))

#5.1.4 Plotting multivariate posteriors
#5.1.4.1 Predictor residual plots
m5.4 <- quap(alist(
  M ~ dnorm(mu, sigma),
  mu <- a + bAM*A,
  a ~ dnorm(0,0.2),
  bAM ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = d)

mu <- link(m5.4)
mu_mean <- apply(mu,2, mean)
mu_resid <- d$M - mu_mean

#5.1.4.2 Counterfactual plots
#Prepare new counterfactual data
M_seq <- seq(from = -2, to = 3, length.out = 30)
pred_data <- data.frame(M = M_seq, A = 0)

#Compute counterfactual mean divorce
mu <- link(m5.3, data = pred_data)
mu_mean <- apply(mu, 2, mean)
mu_PI <- apply(mu, 2, PI)

#Simulate counterfactual divorce outcomes
D_sim <- sim(m5.3, data = pred_data, n=1e4)
D_PI <- apply(D_sim, 2, PI)

#Display predictions, hiding row data with type = 'n'
plot(D ~ M, data = d, xlim = c(-2,3), ylim=c(-2,2),type='n')
mtext("Median age marriage (std) = 0")
lines(M_seq, mu_mean)
shade(mu_PI, M_seq)
shade(D_PI, M_seq)

#5.1.4.3 Posterior predictions plots
#Call link without specifying new data
mu <- link(m5.3)

#Summarize samples across cases
mu_mean <- apply(mu,2,mean)
mu_PI <- apply(mu, 2, PI, prob=.89)

#Simulate observations
D_sim <- sim(m5.3, n=1e4)
D_PI <- apply(D_sim, 2, PI)

plot(mu_mean ~ d$D, col=rangi2, ylim=range(mu_PI), xlab='Observed Divorce',
     ylab='Predicted divorce')
abline(a=0, b=1, lty = 2)
for (i in 1:nrow(d)) lines(rep(d$D[i],2), mu_PI[,i], col=rangi2)
identify(x=d$D, y=mu_mean, labels = d$Loc)


#5.2 Masked Relationship
library(rethinking)
data("milk")
d <- milk
str(d)

d$K <- scale(d$kcal.per.g)
d$N <- scale(d$neocortex.perc)
d$M <- scale(log(d$mass))

m5.5_draft <- quap(alist(
  K ~ dnorm(mu, sigma),
  mu <- a + bN * N,
  a ~ dnorm(0,1),
  bN ~ dnorm(0,1),
  sigma ~ dexp(1)
), data = d)

dcc <- d[complete.cases(d$K, d$N, d$M), ]

m5.5_draft <- quap(flist = alist(
  K ~ dnorm(mu, sigma),
  mu<- a + bN * N,
  a ~ dnorm(0,1),
  bN ~ dnorm(0,1),
  sigma ~ dexp(1)
), data = dcc)

#Test whether the priors are reasonable
prior <- extract.prior(m5.5_draft)
xseq <- c(-2,2)
mu <- link(m5.5_draft, post = prior, data = list(N=xseq))
plot(NULL, xlim=xseq, ylim = xseq)
for (i in 1:150) lines(xseq, mu[i, ], col=col.alpha("black", 0.3))

m5.5 <- quap(flist = alist(
  K ~ dnorm(mu, sigma),
  mu<- a + bN * N,
  a ~ dnorm(0,0.2),
  bN ~ dnorm(0,0.5),
  sigma ~ dexp(1)
), data = dcc)

precis(m5.5)

#Plot the table out
xseq <- seq(from = min(dcc$N)-0.15, to=max(dcc$N)+0.15, length.out = 30)
mu <- link(m5.5, data = list(N=xseq))
mu_mean <- apply(mu, 2, mean)
mu_PI <- apply(mu, 2, PI)
plot(K ~ N, data = dcc)
lines(xseq, mu_mean, lwd=2)
shade(mu_PI, xseq)

#Lets go with the body mass
m5.6 <- quap(alist(
  K ~ dnorm(mu, sigma),
  mu <- a + bM * M,
  a ~ dnorm(0, 0.2),
  bM ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = dcc)
precis(m5.6)

xseq <- seq(from = min(dcc$M)-0.15, to=max(dcc$M)+0.15, length.out = 30)
mu <- link(m5.6, data = list(M=xseq))
mu_mean <- apply(mu, 2, mean)
mu_PI <- apply(mu, 2, PI)
plot(K ~ M, data = dcc)
lines(xseq, mu_mean, lwd=2)
shade(mu_PI, xseq)

# Lets put the two variables together
m5.7 <- quap(alist(
  K ~ dnorm(mu, sigma),
  mu <- a + bN *N + bM * M,
  a ~ dnorm(0, 0.2),
  bN ~ dnorm(0, 0.5),
  bM ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = dcc)
precis(m5.7)

plot(coeftab(m5.5, m5.6, m5.7), pars = c("bM", "bN"))
pairs(~K + M +N, dcc)

# Counterfactual plot
xseq <- seq(from = min(dcc$M)-0.15, to = max(dcc$M)+0.15, length.out = 30)
mu <- link(m5.7, data = data.frame(M=xseq, N=0))
mu_mean <- apply(mu, 2, mean)
mu_PI <- apply(mu, 2, PI)
plot(NULL, xlim = range(dcc$M), ylim = range(dcc$K))
lines(xseq, mu_mean, lwd=2)
shade(mu_PI, xseq)


#5.3 Categorical variables
data("Howell1")
d <- Howell1
str(d)

mu_female <- rnorm(1e4, 178, 20)
mu_male <- rnorm(1e4, 178, 20) + rnorm(1e4, 0 ,10)
precis(data.frame(mu_female, mu_male))

#Index variable
d$sex <- ifelse(d$male==1, 2, 1)
str(d$sex)

m5.8 <- quap(alist(
  height ~ dnorm(mu, sigma),
  mu <- a[sex],
  a[sex] ~ dnorm(178,20),
  sigma ~ dunif(0, 50)
), data = d)

precis(m5.8, depth = 2)

post <- extract.samples(m5.8)
post$diff_fm <- post$a[, 1] - post$a[, 2]
precis(post , depth=2)

#5.3.2 Many categories
data(milk)
d <- milk
unique(d$clade)
d$clade_id <- as.integer(d$clade)

d$K <- scale(d$kcal.per.g)
m5.9 <- quap(alist(
  K ~ dnorm(mu, sigma),
  mu <- a[clade_id],
  a[clade_id] ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = d)

labels <- paste("a[" , 1:4, "]:", levels(d$clade), sep="")
plot(precis(m5.9, depth=2, pars = 'a'), labels = labels, xlab = 'expected kcal (std)')


set.seed(63)
d$house <- sample(rep(1:4, each=8), size = nrow(d))

m5.10 <- quap(alist(
  K ~ dnorm(mu, sigma),
  mu <- a[clade_id] + h[house],
  a[clade_id] ~ dnorm(0, 0.5),
  h[house] ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
), data = d)
plot(precis(m5.10, depth=2, pars = c('a','h'), xlab = 'expected kcal (std)'))
     