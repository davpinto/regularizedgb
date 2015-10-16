# ============================================================
# regGaussianBayes: Gaussian Bayes Classifier with Regularized 
# Covariance Matrix Estimation
#
# USAGE: 
# gb.model <- rgbc(x.train, y.train);
# gb.resp  <- predict(model, x.test, y.test);
#
# REFERENCES:
# 1. "A well conditioned estimator for large-dimensional 
# covariance matrices" - Ledoit and Wolf (2004)
# 2. "A Shrinkage Approach to Large-Scale Covariance Matrix
# Estimation and Implications for Functional Genomics" - Schafer and Strimmer (2005)
#
# AUTHOR: 
# David Pinto
#
# LAST UPDATE:
# Nov. 16, 2014 at 16:03
# ============================================================

rgbc <- function(x,y)
{
   buildGaussModel <- function(label,x,y)
   {
      # --- Select samples by label ---
      x.class <- x[y==label,];
      
      # --- Class priori probability ---
      prob <- nrow(x.class)/nrow(x);
      
      # --- Centroid ---
      center <- colMeans(x.class);
      
      # --- Shrinkage Estimator ---
      C <- corpcor::cov.shrink(x=x.class, verbose=FALSE);
      # C <- corpcor::make.positive.definite(cov(x.class));
            
      return(list(lab=label,priori=prob,mu=center,sig=C))
   }
   
   # --- Build a model for each class ---
   labels <- as.factor(y);
   model  <- lapply(levels(labels), FUN=buildGaussModel, x=x, y=labels);
      
   return( structure(model, class='rgbc') )
}

predict.rgbc <- function(model,x,y)
{
   computePosteriori <- function(model,x)
   {
      # --- Apply Bayes Rule ---
      x.dens <- fastMVNDensity(x, model$mu, model$sig, logd=FALSE);
      
      return(x.dens*model$priori)
   }
   
   # --- Assign labels ---   
   levels    <- do.call(c, lapply(model, function(list.el) list.el$lab));
   post.prob <- lapply(model,computePosteriori,x=x);
   post.prob <- do.call(cbind, post.prob);
   post.prob <- sweep(post.prob, 1, rowSums(post.prob), '/');
   post.max  <- apply(post.prob,1,which.max);
   y.hat     <- as.factor( levels[post.max] );
   
   # --- Classification Performance ---
   out.resp <- as.numeric(y);
   out.pred <- as.numeric(y.hat);
   out.perf <- computeClassPerformance(out.resp,out.pred);
   
   return(list(out=y.hat,prob=post.prob,acc=out.perf$acc,auc=out.perf$auc))
}

computeClassPerformance <- function(resp, pred)
{
   # --- Classification Accuracy ---
   acc <- sum( as.numeric(resp==pred) )/length(resp);
   
   # --- Classification AUC (Area Under the ROC Curve) ---
   auc <- as.numeric( pROC::multiclass.roc(resp,pred,levels=resp[!duplicated(resp)])$auc );
   
   return(list(acc=acc, auc=auc))
}

mvnDensity <- function(x,mu,sig)
{
   # --- Data Dimension ---
   k <- ncol(x);
   
   # --- Covariance Inverse ---
   rooti <- backsolve(chol(sig),diag(k));
   
   # --- Mahalanobis Distance ---
   quads <- colSums( (crossprod(rooti,(t(x)-mu)))^2 );
   
   # --- Estimate MVN Density ---
   log.dens <- -(k/2)*log(2*pi) + sum(log(diag(rooti))) - .5*quads;
   dens <- exp(log.dens);
   
   return( dens )
}

splitTrainTest <- function(x,y,test.percent)
{
   # --- Inner class training and testing patterns ---
   splitByClass <- function(label,x,y,percent)
   { 
      x <- x[y==label,,drop=FALSE];
      y <- y[y==label];      
      test.qty <- round(percent*nrow(x));
      test.idx <- 1:test.qty;
      x.test  <- x[test.idx,,drop=FALSE];
      y.test  <- y[test.idx];
      x.train <- x[-test.idx,,drop=FALSE];
      y.train <- y[-test.idx];      
      return(list(x.tr=x.train,x.te=x.test,y.tr=y.train,y.te=y.test))
   }
   
   # --- Get classes ---
   y.label <- as.factor(y);
   labels  <- levels(y.label);
   
   # --- Split by class label ---
   split.data <- lapply(labels,splitByClass,x=x,y=y.label,percent=test.percent);
   
   # --- Join Train and Test patterns ---
   x.train <- do.call(rbind, lapply(split.data,function(l) l$x.tr) );
   x.test  <- do.call(rbind, lapply(split.data,function(l) l$x.te) );
   y.train <- do.call(c, lapply(split.data,function(l) as.character(l$y.tr)) );
   y.test  <- do.call(c, lapply(split.data,function(l) as.character(l$y.te)) );
   
   return(list(x.tr=x.train,x.te=x.test,y.tr=as.factor(y.train),y.te=as.factor(y.test)))
}

# --- Compute F-score Measure ---
computeFStatistic <- function(x, y)
{
   computeClassStats <- function(label,x,y)
   {
      return(list(means=colMeans(x[y==label,,drop=FALSE]),
                  vars=apply(x[y==label,,drop=FALSE],2,var)))
   }
   
   # --- Data Dimension ---
   n <- nrow(x);
   m <- ncol(x);
   
   # --- Classes ---
   y <- as.factor(y);
   labels <- levels(y);
   k <- length(labels);
   
   cat('\n\nRanking Features...\n\n')
   pb <- txtProgressBar(min=0, max=m, style=3)   
   
   # --- Get Class Statistics ---   
   overall.means <- colMeans(x);
   class.sizes   <- do.call(c, lapply(labels,function(label,y) sum(as.numeric(y==label)),y=y) );
   class.stat    <- lapply(labels, computeClassStats, x=x, y=y);
   class.means   <- do.call(rbind, lapply(class.stat, function(stats) stats$means));
   class.vars    <- do.call(rbind, lapply(class.stat, function(stats) stats$vars));   
   
   # --- Compute F-scores ---   
   f.score <- rep(0, times=m);
   for(col.idx in 1:m)
   {
      f.score[col.idx] <- ((n-k)/(k-1))*sum( class.sizes*(class.means[,(col.idx),drop=TRUE]
         - overall.means[col.idx])^2 )/sum( (class.sizes-1)*class.vars[,(col.idx),drop=TRUE] );
            
      setTxtProgressBar(pb, col.idx)
   }
   cat('\n\n')
   close(pb)
   
   # --- Normalize F-scores ---
   f.score <- (f.score-min(f.score))/diff(range(f.score));
   
   # --- Build a feature ranking ---
   rank <- sort.int(f.score, decreasing=TRUE, index.return=TRUE)$ix;
   
   return( list(score=f.score, idx=rank) )
}