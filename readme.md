vector spatial autocorrelation
-------------------------------

Experiment using the [PySAL](https://github.com/pysal/pysal) Moran's I class to template a vector autocorrelation measure from Liu, Tong, and Liu (2014). 

Updated to include computation of moments for hypothesis testing. Included a "slag" method because it needed to be customized to accomomdate array nature of "z". Perhaps there is a better solution that does not require a custom method. 

In Liu, Tong and Liu (2014) they only derive the moments under the assumption of randomness, so I don't think the calculaions assuming normality from the original Moran's I class apply here. In addition, the p-values (p_norm, p_rand) are dependent upon z_norm, so this needs to be rectified/verified since we do not have a deirvation assuming normality. 
