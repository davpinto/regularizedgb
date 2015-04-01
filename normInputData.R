# ====================================================
# normInputData: Data matrix column-wise normalization
#
# USAGE: 
# x.norm <- normInputData(x,type=c('std','cov','cor','cos'));
#
# REFERENCES:
#
# AUTHOR: 
# David Pinto
#
# LAST UPDATE:
# Nov. 16, 2014 at 15:09
# ====================================================

normInputData <- function(x, type='std')
{
   switch(type,
      
      # --- Center columns to zero mean and scale them to unit standard deviation ---
      'std' = {
         x <- apply(x, 2, function(x.col) (x.col-mean(x.col))/sd(x.col));
         x[!is.finite(x)] <- 0;
      },
      
      # --- Center and scale columns such that x'x is the covariance matrix ---
      'cov' = {
         x <- apply(x, 2, function(x.col,n) (x.col-mean(x.col))/sqrt(n-1), n=nrow(x));
      },
      
      # --- Center and scale columns such that x'x is the correlation matrix ---
      'cor' = {
         x <- apply(x, 2, function(x.col,n) (x.col-mean(x.col))/sd(x.col)/sqrt(n-1), n=nrow(x));
         x[!is.finite(x)] <- 0;
      },
      
      # --- Center and scale columns such that x'x is the cossine similarity matrix ---
      'cos' = {
         x <- apply(x, 2, function(x.col) x.col/norm(x.col,type='2'));
      }      
   );
   
   return(x)
}