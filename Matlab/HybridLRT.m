function [class,likelihood] = HybridLRT(x,SNR)
M = [4 8 16 64];
modulationPool = {'4psk' '8psk' '16qam' '64qam'};
likelihood = zeros(1,4);
N0 = 10^(-SNR/20);
sigma = N0/sqrt(2);
maxlike = zeros(1,90*17);
for j = 1:4
    m = 1;
    data = [0:1:M(j)-1];
    if(j == 1)
        txSig = pskmod(data,M(j));
    elseif(j==2)
        txSig = pskmod(data,M(j),pi/M(j));
    else
        txSig = qammod(data,M(j));
        txSig = txSig*1/sqrt(2/3*(M(j)-1));
    end
    for amplitude = 0.2:0.05:1
        for theta = 1:1:90
            liketemp = 0;
            for t = 1:length(x)
                liketemp = liketemp + log10(sum(1/M(j)/(2*pi*sigma^2).*exp(-(abs(x(t)-amplitude*exp(i*theta*pi/180)*txSig)).^2/2/(sigma^2))));
            end
            maxlike(m) = liketemp;
            m = m+1;
        end
    end
    likelihood(j) = max(maxlike);
end
[A I] =  max(likelihood);
class = modulationPool{I};
end