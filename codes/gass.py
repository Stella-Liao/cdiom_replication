import numpy as np
import pandas as pd
from spglm.iwls import _compute_betas, _compute_betas_gwr #_gwr
from mgwr.search import golden_section
import statsmodels.api as sm
import geopandas as gpd
import libpysal.weights as sw
import copy
from copy import deepcopy
import math
import scipy
from spglm import family
from terms import ConstantTerm, LinearTerm, SATerm, DIOTerm

class CDIOM:
    def __init__(self, y, *args, constant = False):
        self.y = y
        self.z = None
        self.args = args # Input model terms
        self.term_mapping = {} # Dictionary to store the mapping of each covariate in 'X' to its corresponding term
        self.constant = constant # Intercept
        self.num_constant_term = 0 # NO. of intercept, it should be one.
        self.num_linear_terms = 0 # NO. of linear terms
        self.initial_sigmas = [] 
        self.sigmas = [] 
        self.initial_X = None
        self.X = None
        self.w = None
        self.final_X = None
        self.fit_function = None
        self.AWCI_sigmas = None
        self.RBCI_sigmas = None
        self.CI_betas = None
        self.fitted_y = None
        self.residuals = None
        self.std_err = None
        self.tvals = None
        self.zvals = None
        self.pvals = None
        self.AIC = None
        self.log_likelihood = None
        self.Deviance = None
        self.R_squared = None  
        self.R_squared_CS = None
        self.R_squared_McFadden = None
        self.percent_deviance = None
        
        self.initialize()

    def initialize(self):
        initial_X_matrices = []
        current_col_index = 0 # Current column index in initial_X
        
        if self.constant == True:
            constant = ConstantTerm(self.y.shape[0])
            initial_X_matrices.append(constant.X)
            current_col_index = 1
            self.num_constant_term = 1
            self.term_mapping [0] = (type(constant).__name__, constant)  
        
        for arg_idx, arg in enumerate(self.args):
            num_columns = 0  # Number of columns this arg will add to initial_X

            if isinstance(arg, LinearTerm):
                initial_X_matrices.append(arg.X)
                num_columns = arg.X.shape[1]  
                self.num_linear_terms += num_columns 
            
            elif isinstance(arg, SATerm):
                initial_X_matrices.append(arg.cal(arg.initial_sigma))
                num_columns = arg.cal(arg.initial_sigma).shape[1]
                self.initial_sigmas.append(arg.initial_sigma)
            
            elif isinstance(arg, DIOTerm):
                initial_X_matrices.append(arg.cal(arg.initial_sigma))
                num_columns = arg.cal(arg.initial_sigma).shape[1]
                self.initial_sigmas.append(arg.initial_sigma)
                
            else:
                raise ValueError(f"Unsupported term type: {type(arg)}")

            # Record the term mapping for the new columns
            for col in range(current_col_index, current_col_index + num_columns):
                self.term_mapping [col] = (type(arg).__name__, arg)   # storing index and type name

            current_col_index += num_columns  # update current column index

        # Concatenate terms
        self.initial_X = np.hstack(initial_X_matrices)

    def backfit(self, y, X, sigs, verbose = False, max_iter = 50, tol = 1e-8, printed = False):
        n,k = X.shape
        w = np.ones(n)

        betas = _compute_betas_gwr(y, X, w.reshape(-1,1))[0] 
        XB = np.multiply(betas.T, X)
        yhat = np.dot(X, betas)
        err = y.reshape((-1, 1)) - yhat
        # n_iter = 0
        scores = []
        delta = 1e6
        tmp_sigs = sigs

        for n_iter in range(1, max_iter + 1):
            new_XB = np.zeros_like(X)
            params = np.zeros_like(betas)

            for j in range(k):

                temp_y = XB[:, j].reshape((-1, 1))
                temp_y = temp_y + err.reshape((-1, 1))
                temp_X = X[:, j].reshape((-1, 1))
                type_name, term_instance = self.term_mapping[j]  
            
                if type_name not in ['LinearTerm', 'ConstantTerm']:
                    #gscr = lambda x: sm.OLS(y, np.hstack((X[:, np.arange(X.shape[1]) != j], term_instance.cal(x) * self.w))).fit().aic
                    gscr = lambda x: sm.GLM(y, np.hstack((X[:, np.arange(X.shape[1]) != j], term_instance.cal(x) * self.w)), family = sm.families.Gaussian()).fit().aic
                    #gscr = lambda x: sm.GLM(self.y, np.hstack((self.X[:, np.arange(X.shape[1]) != j], term_instance.cal(x))), family = sm.families.Poisson()).fit().aic

                    sig, aic, _ = golden_section(term_instance.lower_bound, term_instance.upper_bound, 0.382, gscr, 1e-2, 50, 99999, int_score = term_instance.int_score)
                    tmp_sigs[j-self.num_linear_terms-self.num_constant_term] = sig
                    sv = term_instance.cal(sig) #new smoothed values
                    if printed:
                        print(sig, aic)
                    #X[:, j] = sv.flatten()
                    self.X[:,j] = sv.flatten()
                    temp_X = (sv * self.w).flatten().reshape((-1,1))

                #beta = _compute_betas(temp_y, temp_X)#[0]
                beta = _compute_betas_gwr(temp_y, temp_X, w.reshape((-1,1)))[0]
                yhat = np.dot(temp_X, beta)
                new_XB[:, j] = yhat.flatten()
                err = (temp_y - yhat).reshape((-1, 1))
                params[j, :] = beta[0][0]

            score = np.sum((y-XB)**2)/n
            XB = new_XB

            scores.append(deepcopy(score))
            delta = score

            if verbose:
                print("Current iteration:", n_iter, ",SOC:", np.round(score, 8))
            if delta < tol:
                break
        
        return params, X, tmp_sigs 
    
    def fit_Poisson(self, input_y = None, verbose = False, max_iter = 50, crit_threshold = 1e-8, printed = False):
        self.fit_function = self.fit_Poisson
        X = self.initial_X.copy()
        self.X = self.initial_X.copy()
        
        y = self.y.copy()
        if input_y is not None:
            y = input_y.copy()
        sigmas = self.initial_sigmas.copy()
        
        # initialize betas and weights
        betas = np.zeros(X.shape[1], np.float)
        offset = np.ones(len(y))
        y_off = y / offset
        fam = family.Poisson()
        y_off = fam.starting_mu(y_off) 
        v = fam.predict(y_off).reshape(-1,1)
        mu = fam.starting_mu(y).reshape(-1,1)
        
        crit = 999999999
        n_iter = 0
        
        while crit > crit_threshold and n_iter < max_iter:
            
            # calcuate weights and adjusted y
            w = fam.weights(mu)
            z = v + (fam.link.deriv(mu) * (y.reshape(-1,1) - mu))
            
            w = np.sqrt(w)
            wx = np.multiply(X, w.reshape(-1,1))
            wz = np.multiply(z.reshape(-1,1), w.reshape(-1,1))
            self.w = w.copy()

            n_betas, _, tmp_sigmas = self.backfit(wz, wx, sigmas, verbose = verbose, max_iter = max_iter, tol = crit_threshold, printed = printed)  
            sigmas = deepcopy(tmp_sigmas)
            X = deepcopy(self.X)
            # print(n_betas)
            
            # update v and mu for next iteration
            v = np.dot(X, n_betas)#.reshape(-1,1)
            mu = fam.fitted(v)
            mu = mu * offset.reshape(-1,1)
            
            # criterion 
            num = np.sum((n_betas - betas)**2) / len(y)
            den = np.sum(np.sum(n_betas, axis=1)**2)
            crit = (num / den)**0.5
            betas = n_betas
            
            n_iter += 1 # increment the iteration counter
            #print(n_iter)
            
        self.coefficients = betas
        self.sigmas = sigmas
        self.final_X = X
        self.z = z
        
        self.wz = wz
        self.wx = wx
        self.w = w

        pass
    
    def inference_Poisson(self): 
        # Linear prediction
        eta = np.dot(self.final_X, self.coefficients)

        self.fitted_y = np.exp(eta)

        n = self.final_X.shape[0]  # number of observaions  
        k = self.final_X.shape[1]  # number of parameters

        # Calculate residuals (Pearson residuals for Poisson)
        residuals = (self.y.reshape(-1,1) - self.fitted_y) / np.sqrt(self.fitted_y)
        self.residuals = residuals

        # Calculate standard error of coefficients
        # For Poisson, the variance is the mean. So, V^-1 = diagonal of means
#         V_inv = np.diag(self.fitted_y.flatten())
#         cov_beta = np.linalg.inv(np.dot(self.final_X.T, np.dot(V_inv, self.final_X)))
        V_inv = self.fitted_y.flatten()  # This represents the diagonal entries directly

        # Calculate the weighted dot product for the design matrix
        # Element-wise multiplication of each column of final_X with V_inv, then matrix multiplication with final_X.T
        weighted_X = self.final_X.T * V_inv
        cov_beta = np.linalg.inv(np.dot(weighted_X, self.final_X))
        
        # Calculate standard error of coefficients
        se_beta = np.sqrt(np.diag(cov_beta))
        self.std_err = se_beta

        # Calculate confidence intervals
        critical_value = scipy.stats.norm.ppf(1 - 0.05 / 2)  # Z-value for Poisson
        coefs_lower_bound = self.coefficients.flatten() - critical_value * se_beta
        coefs_upper_bound = self.coefficients.flatten() + critical_value * se_beta
        self.CI_betas = list(zip(coefs_lower_bound, coefs_upper_bound))

        # Calculate Wald statistics (z values for Poisson)
        self.zvals = self.coefficients.flatten() / se_beta

        # Calculate p values
        self.pvals = 2 * (1 - scipy.stats.norm.cdf(np.abs(self.zvals)))
        
        # Calculate Deviance: the deviance of the model with predictors (your fitted model). 
        ratio = self.y.reshape(-1,1) / self.fitted_y
        ratio = np.where(ratio == 0, 1, ratio)
        self.Deviance = 2 * np.sum(self.y.reshape(-1,1) * np.log(ratio) - (self.y.reshape(-1,1) - self.fitted_y))

        # Compute log-likelihood for null model
        lambda_null = np.mean(self.y.reshape(-1,1))  # The mean of observed counts for the null model
        logL0 = np.sum(-lambda_null + self.y.reshape(-1,1) * np.log(lambda_null) - scipy.special.gammaln(self.y.reshape(-1,1) + 1))

        # Compute log-likelihood for full model
        logLm = np.sum(-self.fitted_y + self.y.reshape(-1,1) * np.log(self.fitted_y) - scipy.special.gammaln(self.y.reshape(-1,1) + 1))
        self.log_likelihood = logLm
        
        # Calculate AIC
        self.AIC = 2*k - 2*logLm 
        
        #Calculate percet deviance (analogous to residual sum of squares in OLS)
        ratio2 = self.y.reshape(-1,1) / lambda_null
        ratio2 = np.where(ratio2 == 0, 1, ratio2)
        null_deviance = 2 * (np.sum(self.y.reshape(-1,1) * np.log(ratio2) - (self.y.reshape(-1,1) - lambda_null)))
        model_deviance = 2 * np.sum(self.y.reshape(-1,1) - self.fitted_y + self.y.reshape(-1,1) * np.log(ratio))
        percent_deviance = 1 - (model_deviance / null_deviance)
        self.percent_deviance = percent_deviance

        # Compute Cox & Snell R^2 
        R_squared_CS = 1 - (np.exp(2 * (logL0 - logLm) / len(self.y)))
        self.R_squared_CS = R_squared_CS
        
        # Compute McFadden R^2
        R_squared_McFadden =  1 - logLm/logL0
        self.R_squared_McFadden = R_squared_McFadden
        
        pass
    
    def calculate_AWCI_sigmas(self, level = 0.95):
        
        self.AWCI_sigmas = []
        for tidx, tsig in enumerate(self.sigmas):
            
            tsig_idx = int(tidx + self.num_linear_terms + self.num_constant_term)
            tsig_term_instance = self.term_mapping[tsig_idx][1]
            
            # create an array of candidate sigmas
            tsig_b4 = np.arange(tsig_term_instance.lower_bound, tsig, tsig_term_instance.CI_step)
            tsig_af = np.arange(tsig, tsig_term_instance.upper_bound, tsig_term_instance.CI_step)
            tsig_candidates = np.hstack((tsig_b4, tsig_af)).flatten()
            
            tsig_aics = []
            for sig in tsig_candidates:
                aic = sm.GLM(self.wz, np.hstack((self.wx[:, np.arange(self.wx.shape[1]) != tsig_idx], tsig_term_instance.cal(sig) * self.w)), family = sm.families.Gaussian()).fit().aic
                
                tsig_aics.append((sig, aic))
                
            tsig_awdf = pd.DataFrame(tsig_aics, columns=['Sigma', 'AIC'])  

            minAIC = np.min(tsig_awdf.AIC)
            deltaAICs = tsig_awdf.AIC - minAIC
            awsum = np.sum(np.exp(-0.5 * deltaAICs))
            tsig_awdf = tsig_awdf.assign(AW = np.exp(-0.5 * deltaAICs)/awsum)
            tsig_awdf = tsig_awdf.sort_values(by = 'AW',ascending=False)
            tsig_awdf = tsig_awdf.assign(cumAW = tsig_awdf.AW.cumsum())

            index = len(tsig_awdf[tsig_awdf.cumAW < level]) + 1
            tsig_min = tsig_awdf.iloc[:index,:].Sigma.min()
            tsig_max = tsig_awdf.iloc[:index,:].Sigma.max()
            
            self.AWCI_sigmas.append((round(tsig_min, 4), round(tsig_max,4)))
            
        pass
    
    def calculate_RBCI_sigmas(self, level=0.95, max_iter = 100, crit_threshold = 1e-8, printed = False):
        
        if self.fit_function is None:
            raise ValueError("No fit function has been set. Please call fit_Gaussain or fit_Poisson before calling this method.")

        fitted_y = self.fitted_y.copy()
        residuals = self.y.reshape(-1,1) - self.fitted_y
        
        self.RBCI_sigmas = []
        sigdicts = {}
        lower = (1 - level) * 100 / 2.0
        upper = 100 - lower

        sigdicts = {i: [] for i in range(len(self.sigmas))}

        for i in range(max_iter):
            
            np.random.seed(i)
            bootstrap_residuals = np.random.choice(residuals[:, 0], size=len(residuals), replace=True).reshape(-1, 1)
            bootstrap_y = (fitted_y + bootstrap_residuals).flatten()
            tgass = deepcopy(self)

            tgass.fit_function(input_y = bootstrap_y, crit_threshold = crit_threshold)
            
            for tidx, tsig in enumerate(tgass.sigmas):
                sigdicts[tidx].append(tsig)
                
            if printed:
                print(i)

        for siglist in sigdicts.values():
            sigdf = pd.DataFrame(siglist)
            sigdf.columns = ['Sigma']
            sigdf = sigdf.sort_values(by=['Sigma'])

            minSig = np.percentile(sigdf, lower)
            maxSig = np.percentile(sigdf, upper)

            self.RBCI_sigmas.append((round(minSig, 4), round(maxSig, 4)))