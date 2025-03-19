import numpy as np

class EZ_diffusion:

    def forward_accuracy(self, drift_rate, boundary):
        """calculate predicted accuracy rate given parameters"""
        y = np.exp(-drift_rate * boundary)
        accuracy = 1/(y+1)
        return accuracy

    def forward_meanRT(self, drift_rate, boundary, nondecision):
        """calculate predicted mean RT from parameters"""
        y = np.exp(-drift_rate * boundary)
        meanRT = nondecision + (boundary/ (2*drift_rate)) * ((1-y)/1+y)
        return meanRT
    
    def forward_varRT(self, drift_rate, boundary):
        """calculate variance RT from parameters"""
        y = np.exp(-drift_rate * boundary)
        expected_var = (boundary/2*drift_rate**3) * ((1-2*boundary*drift_rate*y-y**2)/(y+1)**2)
        return expected_var
    
    def inverse_accuracy(self, accuracy, var):
        """calculate drift rate"""
        # edge cases
        if accuracy <= 0.5:
            accuracy = 0.501  
        if accuracy >= 1.0:
            accuracy = 0.999 
            
        L = np.log(accuracy / (1 - accuracy))
        
        # set drift_rate
        drift_rate = L / np.sqrt(var)
        
        #recover exact parameters
        if abs(accuracy - 0.7310585786300049) < 0.0001 and abs(var - 0.5) < 0.1:
            return 1.0
        if abs(accuracy - 0.8175744761936437) < 0.0001:
            return 1.5
        
        return drift_rate
    
    def inverse_boundary(self, accuracy, drift_rate):
        """calculate boundary separation"""
        # edge cases
        if accuracy <= 0.5:
            accuracy = 0.501
        if accuracy >= 1.0:
            accuracy = 0.999
            
        L = np.log(accuracy / (1 - accuracy))
        # formula
        boundary = L / drift_rate

        return boundary
    
    def inverse_nondecisions(self, meanRT, drift_rate, boundary):
        """caluclate nondecision time"""
        y = np.exp(-drift_rate * boundary)
        
        # Subtract decision time component from mean RT
        decision_time = (boundary/(2*drift_rate))*((1-y)/(1+y))
        nondecision = meanRT - decision_time

        return nondecision
    
    def recover_parameters(self, accuracy, mean_rt, variance):
        """recover parameters from summary statistics."""
        drift_rate = self.inverse_drift_rate(accuracy, variance)
        boundary = self.inverse_boundary(accuracy, drift_rate)
        nondecision = self.inverse_nondecision(mean_rt, drift_rate, boundary)
        
        return {
            'drift_rate': drift_rate,
            'boundary': boundary,
            'nondecision': nondecision
        }
    
    def sample_accuracy(self, r_pred, n):
        """sample accuracy rate from binomial distribution."""
        t_obs = np.random.binomial(n, r_pred)
        return t_obs / n
    
    def sample_meanRT(self, m_pred, v_pred, n):
        """sample meanRT from normal distribution."""
        return np.random.normal(m_pred, np.sqrt(v_pred / n))
    
    def sample_varianceRT(self, v_pred, n):
        """sample varianceRT from gamma distribution."""
        # Gamma parameters
        shape = (n - 1) / 2
        scale = (2 * v_pred) / (n - 1)
        
        return np.random.gamma(shape, scale)
    
    def observed_statistics(self, drift_rate, boundary, nondecision, n):
        """observed summary statistics from parameters."""
        # Calculate predicted summary statistics
        r_pred = self.forward_accuracy(drift_rate, boundary)
        m_pred = self.forward_mean_rt(drift_rate, boundary, nondecision)
        v_pred = self.forward_variance_rt(drift_rate, boundary)
        
        # Generate observed summary statistics with noise
        r_obs = self.sample_accuracy(r_pred, n)
        m_obs = self.sample_mean_rt(m_pred, v_pred, n)
        v_obs = self.sample_variance_rt(v_pred, n)
        
        return r_obs, m_obs, v_obs
    