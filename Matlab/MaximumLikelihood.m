function [class,likelihood] = MaximunLikelihood(x,SNR)
M = [4 8 16 64];
modulationPool = {'4psk' '8psk' '16qam' '64qam'};
likelihood = zeros(1,4);
N0 = 10^(-SNR/20);
sigma = N0/sqrt(2);
for j = 1:4
    data = [0:1:M(j)-1];
    if(j == 1)
        txSig = pskmod(data,M(j));
    elseif(j==2)
        txSig = pskmod(data,M(j),pi/M(j));
    else
        txSig = qammod(data,M(j));
        txSig = txSig*1/sqrt(2/3*(M(j)-1));
    end    
    for i = 1:length(x)
        likelihood(j) = likelihood(j) + log10(sum(1/M(j)/(2*pi*sigma^2).*exp(-(abs(x(i)-txSig)).^2/2/(sigma^2))));
    end
end
[A I] =  max(likelihood);
class = modulationPool{I};
end