/*************************************************************************************
 * The original code can be found in: http://gallery.rcpp.org/articles/dmvnorm_arma/ * 
 *************************************************************************************/

/* PS: recommended for Ubuntu users */

#include "RcppArmadillo.h"
#include "Rcpp.h"
#include <omp.h>

const double log2pi = std::log(2.0 * M_PI);

/*** Calculate Multivariate Normal Density ***/
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::vec parallelMVNDensity(arma::mat x,  
                         arma::rowvec mean,  
                         arma::mat sigma, 
                         bool logd = false,
                         int cores = 2) { 
    omp_set_num_threads(cores);
    int n = x.n_rows;
    int xdim = x.n_cols;
    arma::vec out(n);
    arma::mat rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(xdim/2) * log2pi;
    #pragma omp parallel for schedule(static) 
    for (int i=0; i < n; i++) {
        arma::vec z = rooti * arma::trans( x.row(i) - mean) ;    
        out(i)      = constants - 0.5 * arma::sum(z%z) + rootisum;     
    }  
      
    if (logd==false) {
        out=exp(out);
    }
    return(out);
}