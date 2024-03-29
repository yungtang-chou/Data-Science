# Chap 14 Adventures in Covariance
library(rethinking)
# 14.1 Varying slopes by construction
a <- 3.5
b <- (-1)
sigma_a <- 1
sigma_b <- 0.5
rho <- (-0.7) # correlation between intercept and slopes

Mu <- c(a,b)

cov_ab <- sigma_a*sigma_b*rho
Sigma <- matrix(c(sigma_a^2, cov_ab, cov_ab, sigma_b^2), ncol = 2)

sigmas <- c(sigma_a, sigma_b)
Rho <- matrix(c(1,rho,rho,1), nrow = 2)
Sigma <- diag(sigmas) %*% Rho %*% diag(sigmas) # Matrix mulitiplication


N_cafes <- 20
library(MASS)
set.seed(5)
vary_effects <- mvrnorm(N_cafes, Mu, Sigma)

a_cafe <- vary_effects[,1]
b_cafe <- vary_effects[,2]
plot(a_cafe, b_cafe, col=rangi2, xlab='intercepts (a_cafe)', ylab='slopes (b_cafe)')
# Overlay population distribution
library(ellipse)
for (l in c(0.1,0.3,0.5,0.8,0.99))
  lines(ellipse(Sigma,centre = Mu, level = l), col = col.alpha("black",0.2))



# 14.1.2 Simulate observations
set.seed(22)
N_visits <- 10
afternoon <- rep(0:1, N_visits*N_cafes/2)
cafe_id <- rep(1:N_cafes, each = N_visits)
mu <- a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
sigma <- 0.5
wait <- rnorm(N_visits*N_cafes, mu, sigma)
d <- data.frame(cafe = cafe_id, afternoon = afternoon, wait = wait)


# 14.1.3 The varying slopes model
R <- rlkjcorr(1e4, K = 2, eta = 2)
dens(R[,1,2], xlab='correlation')

m14.1 <- ulam(
  alist(
    wait ~ normal(mu,sigma),
    mu <- a_cafe[cafe] + b_cafe[cafe] * afternoon,
    c(a_cafe,b_cafe)[cafe] ~ multi_normal(c(a,b),Rho,sigma_cafe),
    a ~ normal(5,2),
    b ~ normal(-1,0.5),
    sigma_cafe ~ dexp(1),
    sigma ~ dexp(1),
    Rho ~ dlkjcorr(2)
  ), data = d, chains = 4, cores = 4
)

precis(m14.1)

post <- extract.samples(m14.1)
dens(post$Rho[,1,2])

# compute unpooled estimates directly from data
a1 <- sapply(1:N_cafes, function(i) mean(wait[cafe_id==i & afternoon==0]))
b1 <- sapply(1:N_cafes, function(i) mean(wait[cafe_id==i & afternoon==1])) - a1

# extract posterior means of partially pooled estimates
post <- extract.samples(m14.1)
a2 <- apply(post$a_cafe, 2, mean)
b2 <- apply(post$b_cafe, 2, mean)

# plot both and connect with lines
plot(a1, b1, xlab='intercept', ylab='slope',
     pch = 16, col=rangi2, ylim=c(min(b1)-0.1, max(b1) + 0.1), 
     xlim=c(min(a1)-0.1,max(a1)+0.1))
points(a2, b2, pch=1)
for (i in 1:N_cafes) lines(c(a1[i],a2[i]), c(b1[i],b2[i]))

# compute posterior mean bivariate Gaussian
Mu_est <- c(mean(post$a), mean(post$b))
rho_est <- mean(post$Rho[,1,2])
sa_est <- mean(post$sigma_cafe[,1])
sb_est <- mean(post$sigma_cafe[,2])
cov_ab <- sa_est * sb_est * rho_est
Sigma_est <- matrix(c(sa_est^2, cov_ab, cov_ab, sb_est^2), ncol=2)

for (l in c(0.1,0.3,0.5,0.8,0.99))
  lines(ellipse(Sigma_est,centre = Mu_est,level = l), col = col.alpha("black",0.2))


# convert varying effects to waiting times
wait_morning_1 <- (a1)
wait_afternoon_1 <- (a1 + b1)
wait_morning_2 <- (a2)
wait_afternoon_2 <- (a2 + b2)

# plot both and connect with lines
plot(wait_morning_1, wait_afternoon_1, xlab='morning wait', ylab='afternoon wait',
     pch = 16, col=rangi2, ylim=c(min(wait_afternoon_1) -0.1, max(wait_afternoon_2) + 0.1),
     xlim=c(min(wait_morning_1)-0.1,max(wait_morning_2)+0.1))
points(wait_morning_2, wait_afternoon_2, pch=1)
for (i in 1:N_cafes)
  lines(c(wait_morning_1[i], wait_morning_2[i]), 
        c(wait_afternoon_1[i],wait_afternoon_2[i]))
abline(a=0, b=1, lty=2)

# now shrinkage distribution by simulation
v <- mvrnorm(1e4, Mu_est, Sigma_est)
v[,2] <- v[,1] + v[,2] # calculate afternoon wait
Sigma_est2 <- cov(v)
Mu_est2 <- Mu_est
Mu_est2[2] <- Mu_est[1] + Mu_est[2]

# draw contours
for (l in c(0.1,0.3,0.5,0.8,0.99))
lines(ellipse(Sigma_est2, centre = Mu_est2, level = l),
      col=col.alpha("black",0.5))

# 14.2 Advanced varying slopes
library(rethinking)
data("chimpanzees")
d <- chimpanzees; rm(chimpanzees)
d$block_id <- d$block
d$treatment <- 1L + d$prosoc_left + 2L*d$condition

dat <- list(
  L = d$pulled_left,
  tid = d$treatment,
  actor = d$actor,
  block_id = as.integer(d$block_id)
)

m14.2 <- ulam(
  alist(
    L ~ dbinom(1,p),
    logit(p) <- g[tid] + a[actor,tid] + b[block_id,tid],
    
    # adaptive priors
    vector[4]:a[actor] ~ multi_normal(0, Rho_actor, sigma_actor),
    vector[4]:b[block_id] ~ multi_normal(0, Rho_block, sigma_block),
    
    # fixed priors
    g[tid] ~ dnorm(0,1),
    sigma_actor ~ dexp(1),
    Rho_actor ~ dlkjcorr(4),
    sigma_block ~ dexp(1),
    Rho_block ~ dlkjcorr(4)
  ), data = dat, chains = 4, cores = 2
)

m14.3 <- ulam(
  alist(
    L ~ dbinom(1,p),
    logit(p) <- g[tid] + a[actor,tid] + b[block_id,tid],
    
    # adaptive priors - non-centered
    transpars> matrix[actor,4]:a <- 
      compose_noncentered(sigma_actor, L_Rho_actor, z_actor),
    transpars> matrix[block_id,4]:b <-
      compose_noncentered(sigma_block, L_Rho_block, z_block),
    matrix[4, actor]:z_actor ~ normal(0,1),
    matrix[4, block_id]:z_block ~ normal(0,1),
    
    # fixed priors
    g[tid] ~ dnorm(0,1),
    vector[4]: sigma_actor ~ dexp(1),
    cholesky_factor_corr[4]:L_Rho_actor ~ lkj_corr_cholesky(2),
    vector[4]: sigma_block ~ dexp(1),
    cholesky_factor_corr[4]:L_Rho_block ~ lkj_corr_cholesky(2),
    
    # compute ordinary correlation matrixes from Cholesky factors
    gq> matrix[4,4]:Rho_actor <<- Chol_to_Corr(L_Rho_actor),
    gq> matrix[4,4]:Rho_block <<- Chol_to_Corr(L_Rho_block)
  ), data = dat, chains = 4, cores = 4, log_lik = TRUE
)




# 14.3 Instrumental variables and front doors
# 14.3.1 Instrumental variables
set.seed(73)
N <- 500
U_sim <- rnorm(N)
Q_sim <- sample(1:4, size=N, replace=T)
E_sim <- rnorm(N, U_sim + Q_sim)
W_sim <- rnorm(N, U_sim + 0*E_sim)

dat_sim <- list(
  W = standardize(W_sim),
  E = standardize(E_sim),
  Q = standardize(Q_sim)
)

m14.4 <- ulam(alist(
  W ~ dnorm(mu, sigma),
  mu<- aW + bEW *E,
  aW ~ dnorm(0,0.2),
  bEW ~ dnorm(0,0.5),
  sigma ~ dexp(1)
), data = dat_sim, chains = 4, cores = 2)

precis(m14.4)

m14.5 <- ulam(alist(
  c(W,E) ~ multi_normal(c(muW, muE), Rho, Sigma),
  muW <- aW + bEW * E,
  muE <- aE + bQE * Q,
  c(aW,aE) ~ normal(0,0.2),
  c(bEW,bQE) ~ normal(0, 0.5),
  Rho ~ dlkjcorr(2),
  Sigma ~ dexp(1)
), data = dat_sim, chains = 4, cores = 2)

precis(m14.5, depth=3)

set.seed(73)
N <- 500
U_sim <- rnorm(N)
Q_sim <- sample(1:4, size=N, replace=TRUE)
E_sim <- rnorm(N, U_sim + Q_sim)
W_sim <- rnorm(N, -U_sim + 0.2*E_sim)
dat_sim <- list(
  W = standardize(W_sim),
  E = standardize(E_sim),
  Q = standardize(Q_sim)
)

m14.4x <- ulam(m14.4, data = dat_sim , chains = 4, cores = 4)
m14.5x <- ulam(m14.5, data = dat_sim , chains = 4, cores = 4)

library(dagitty)
dagIV <- dagitty("dag{
                 E -> W
                 E <- U -> W
                 Q -> E
                 }")
instrumentalVariables(dagIV, exposure = 'E', outcome = 'W')

# 14.3.2 Front-door criterion



# 14,4 Social relations as correlated varying effects
library(rethinking)
data(KosterLeckie)

kl_data <- list(
  N = nrow(kl_dyads),
  N_households = max(kl_dyads$hidB),
  did = kl_dyads$did,
  hidA = kl_dyads$hidA,
  hidB = kl_dyads$hidB,
  giftsAB = kl_dyads$giftsAB,
  giftsBA = kl_dyads$giftsBA
)

m14.6 <- ulam(
  alist(
    giftsAB ~ dpois(lambdaAB),
    giftsBA ~ dpois(lambdaBA),
    log(lambdaAB) <- a + gr[hidA,1] + gr[hidB,2] + d[did,1],
    log(lambdaBA) <- a + gr[hidB,1] + gr[hidA,2] + d[did,2],
    a ~ dnorm(0,1),
    
    # gr matrix of varying effects
    vector[2]:gr[N_households] ~ multi_normal(0, Rho_gr, sigma_gr),
    Rho_gr ~ dlkjcorr(4),
    sigma_gr ~ dexp(1),
    
    # dyad effects
    transpars> matrix[N,2]:d <-
      compose_noncentered(rep_vector(sigma_d,2), L_Rho_d, z),
    matrix[2,N]:z ~ normal(0,1),
    cholesky_factor_corr[2]:L_Rho_d ~ lkj_corr_cholesky(8),
    sigma_d ~ dexp(1),
    
    # compute correlation matrix for dyads
    gq> matrix[2,2]:Rho_d <<- Chol_to_Corr(L_Rho_d)
  ), data = kl_data, chains = 4, cores = 4, iter=3000)

precis(m14.6, depth = 3, pars=c("Rho_gr","sigma_gr"))

post <- extract.samples(m14.6)
g <- sapply(1:25, function(i) post$a + post$gr[,i,1])
r <- sapply(1:25, function(i) post$a + post$gr[,i,2])
Eg_mu <- apply(exp(g), 2 ,mean)
Er_mu <- apply(exp(r), 2, mean)

plot(exp(g[,i]),exp(r[,i]))

plot(NULL, xlim=c(0,8.6), ylim=c(0,8.6), xlab='generalized giving',
     ylab='generalized receiving', lwd=1.5)
abline(a=0, b=1, lty=2)

# ellipses
library(ellipse)
for (i in 1:25){
  Sigma <- cov(cbind(g[,i],r[,i]))
  Mu <- c(mean(g[,i]),mean(r[,i]))
  for (l in c(0.5)){
    el <- ellipse(Sigma, centre = Mu, level = l)
    lines(exp(el), col=col.alpha("black",0.5))
  }
}
points(Eg_mu, Er_mu, pch=21, bg='white', lw=1.5)
precis(m14.6, depth=3, pars=c("Rho_d","sigma_d"))

dy1 <- apply(post$d[,,1], 2, mean)
dy2 <- apply(post$d[,,2], 2, mean)
plot(dy1, dy2)


# 14.5 Continuous categories and the Gaussian process
# 14.5.1 Example: Spatial autocorrelation in Oceanic tools
library(rethinking)
data("islandsDistMatrix")

Dmat <- islandsDistMatrix
colnames(Dmat) <- c('Ml','Ti','SC','Ya','Fi','Tr','Ch','Mn','To','Ha')
round(Dmat,1)

# linear
curve(exp(-1*x), from=0, to=4, lty=2,
      xlab='distance',ylab='correlation')
# squared
curve(exp(-1*x^2), add=TRUE)

data("Kline2")
d <- Kline2
d$society <- 1:10

dat_list <- list(
  T = d$total_tools,
  P = d$population,
  society = d$society,
  Dmat = islandsDistMatrix
)

m14.7 <- ulam(
  alist(
    T ~ dpois(lambda),
    lambda <- (a*P^b/g)*exp(k[society]),
    vector[10]: k ~ multi_normal(0, SIGMA),
    matrix[10,10]: SIGMA <- cov_GPL2(Dmat, etasq, rhosq, 0.01),
    c(a,b,g) ~ dexp(1),
    etasq ~ dexp(2),
    rhosq ~ dexp(0.5)
), data = dat_list, chains = 4, cores = 4)

precis(m14.7, depth=3)

# non-centered version
m14.7nc <- ulam(
  alist(
    T ~ dpois(lambda),
    lambda <- (a*P^b/g)*exp(k[society]),
    
    # non-centered
    transpars> vector[10]: k <<- L_SIGMA * z,
    vector[10]: z ~ normal(0,1),
    transpars> matrix[10,10]:L_SIGMA <<- cholesky_decompose(SIGMA),
    transpars> matrix[10,10]:SIGMA <- cov_GPL2(Dmat,etasq,rhosq,0.01),
    
    c(a,b,g) ~ dexp(1),
    etasq ~ dexp(2),
    rhosq ~ dexp(0.5)
  ), data = dat_list, chains = 4, cores = 4)

precis(m14.7nc, depth=3, pars = 'k')


post <- extract.samples(m14.7)

# plot the posterior median covariance function
plot(NULL, xlab='distance (thousand km)', ylab='covariance',
     xlim=c(0,10), ylim=c(0,2))
# compute posterior mean covariance
x_seq <- seq(from = 0, to = 10, length.out = 100)
pmcov <- sapply(x_seq, function(x) post$etasq*exp(-post$rhosq*x^2))
pmcov_mu <- apply(pmcov,2,mean)
lines(x_seq,pmcov_mu, lwd=2)

# plot 60 functions sampled from the posterior
for (i in 1:60) {
  curve(post$etasq[i] * exp(-post$rhosq[i]*x^2), add=TRUE,
        col=col.alpha("black",0.3))
}


# compute posterior median covariance among societies
K <- matrix(0, nrow = 10, ncol = 10)
for (i in 1:10)
  for (j in 1:10)
    K[i,j] <- median(post$etasq) * exp(-median(post$rhosq) * islandsDistMatrix[i,j]^2)
diag(K) <- median(post$etasq) + 0.01
# convert to a correlation matrix
Rho <- round(cov2cor(K), 2)
# add row/col names for convenience
colnames(Rho) <- c("Ml",'Ti','SC','Ya','Fi','Tr','Ch','Mn','To','Ha')
rownames(Rho) <- colnames(Rho)
Rho


# scale points size to logpop
psize <- d$logpop / max(d$logpop)
psize <- exp(psize*1.5)-2
# plot raw data and labels
plot(d$lon2, d$lat, xlab='longitude',ylab='latitude',
     col=rangi2, cex=psize, pch=16, xlim=c(-50,30))
labels <- as.character(d$culture)
text(d$lon2, d$lat, labels = labels, cex=0.7, pos=c(2,4,3,3,4,1,3,2,4,2))
# overlay lines shaded by Rho
for (i in 1:10)
  for (j in 1:10)
    if (i < j)
      lines(c(d$lon2[i],d$lon2[j]), c(d$lat[i],d$lat[j]), lwd=2,
            col=col.alpha("black",Rho[i,j]^2))



# compute posterior median relationship, ignoring distance
logpop.seq <- seq(from=6, to=14, length.out = 30)
lambda <- sapply(logpop.seq, function(lp) exp(post$a + post$b * lp))
lambda.median <- apply(lambda , 2 , median)
lambda.PI80 <- apply(lambda, 2, PI, prob=.8)
# plot raw data and labels
plot(d$logpop, d$total_tools, col=rangi2, cex=psize, pch=16,
     xlab='log population',ylab='total tools')
text(d$logpop, d$total_tools, labels = labels, cex=0.7,
     pos = c(4,3,4,2,2,1,4,4,4,2))
# display posterior predictions
lines(logpop.seq, lambda.median, lty=2)
lines(logpop.seq, lambda.PI80[1,], lty=2)
lines(logpop.seq, lambda.PI80[2,], lty=2)
# overlay correlations
for (i in 1:10)
  for (j in 1:10)
    if (i < j)
      lines(c(d$logpop[i],d$logpop[j]), c(d$total_tools[i],d$total_tools[j]), lwd=2,
            col=col.alpha("black",Rho[i,j]^2))


# 14.5.2 Example: Phylogenetic distance
data("Primates301")
data("Primates301_nex")

# plot
library(ape)
plot(ladderize(Primates301_nex), type='fan', font=1, no.margin=TRUE,
     label.offset = 1)


d <- Primates301
d$name <- as.character(d$name)
dstan <- d[complete.cases(d$group_size, d$body, d$brain),]
spp_obs <- dstan$name

dat_list <- list(
  N_spp = nrow(dstan),
  M = standardize(log(dstan$body)),
  B = standardize(log(dstan$brain)),
  G = standardize(log(dstan$group_size)),
  Imat = diag(nrow(dstan))
)

m14.8 <- ulam(
  alist(
    G ~ multi_normal(mu, SIGMA),
    mu <- a + bM * M + bB * B,
    matrix[N_spp,N_spp]: SIGMA <- Imat * sigma_sq,
    a ~ dnorm(0,1),
    c(bM, bB) ~ dnorm(0,0.5),
    sigma_sq ~ dexp(1)
  ), data = dat_list, chains = 4, cores = 4
)

precis(m14.8)

tree_trimmed <- keep.tip(Primates301_nex, spp_obs)
Rbm <- corBrownian(phy = tree_trimmed)
V <- vcv(Rbm)
Dmat <- cophenetic(tree_trimmed)
plot(Dmat, V, xlab='phylogenetic distance', ylab='covariance')
