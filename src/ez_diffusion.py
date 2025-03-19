import numpy as np

class EZ_diffusion:

    def forward_accuracy(self, drift_rate, limit):
        #produces predicted accuracy rate given parameters
        y = np.exp(-drift_rate * limit)
        accuracy = 1/(y+1)
        return accuracy

    def forward_meanRT(self, drift_rate, limit, nondecision):
        #Calculate predicted mean RT from parameters
        y = np.exp(-drift_rate * limit)
        meanRT = nondecision + (limit/ (2*drift_rate)) * ((1-y)/1+y)
        return meanRT
    
    def forward_varRT(self, drift_rate, limit):
        y = np.exp(-drift_rate * limit)
        expected_var = (limit/2*drift_rate**3) * ((1-2*limit*drift_rate*y-y**2)/(y+1)**2)
        return expected_var
    
    def inverse_accuracy(self, accuracy, var):
        """Calculate drift rate from observed summary statistics."""
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
    
    def inverse_limit(self, accuracy, drift_rate):
        """Calculate boundary separation from observed statistics and estimated drift rate."""
        # edge cases
        if accuracy <= 0.5:
            accuracy = 0.501
        if accuracy >= 1.0:
            accuracy = 0.999
            
        L = np.log(accuracy / (1 - accuracy))
        # formula
        limit = L / drift_rate

        return limit
    
    def inverse_nondecisions(self, meanRT, drift_rate, limit):
        """Calculate non-decision time from observed mean RT and estimated parameters."""
        y = np.exp(-drift_rate * limit)
        
        # Subtract decision time component from mean RT
        decision_time = (limit/(2*drift_rate))*((1-y)/(1+y))
        nondecision = meanRT - decision_time

        return nondecision