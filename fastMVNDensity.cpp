/*************************************************************************************
 * The original code can be found in: http://gallery.rcpp.org/articles/dmvnorm_arma/ * 
 *************************************************************************************/

#include "RcppArmadillo.h"
#include "Rcpp.h"

const double log2pi = std::log(2.0 * M_PI);

/*** Calculate Multivariate Normal Density ***/
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::vec fastMVNDensity(arma::mat x,  
                      arma::rowvec mean,  
                      arma::mat sigma, 
                      bool logd = false) { 
    int n = x.n_rows;
    int xdim = x.n_cols;
    arma::vec out(n);
    arma::mat rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
    
    for (int i=0; i < n; i++) {
        arma::vec z = rooti * arma::trans( x.row(i) - mean) ;    
        out(i)      = constants - 0.5 * arma::sum(z%z) + rootisum;     
    }  
      
    if (logd == false) {
        out = exp(out);
    }
    return(out);
}