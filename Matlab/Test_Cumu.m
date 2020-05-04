close all;
M = [4 8 16 64];
modulationPool = {'4psk' '8psk' '16qam' '64qam'};
symbol = 1000;
num = 1000;
SNR = [12];
t = 0;

for k = 1:length(SNR)
    error = zeros(1,4);
    for j = 1:4   
        for i = 1:num
            data = randi([0 M(j)-1],symbol,1);
            if(j == 1)
                txSig = pskmod(data,M(j));
            elseif(j == 2)
                txSig = pskmod(data,M(j),pi/M(j));
            else
                txSig = qammod(data,M(j));            
                txSig = txSig*1/sqrt(2/3*(M(j)-1));
            end
            S = SNR(k);
            rxSig = awgn(txSig,S,'measured');
            tic
            class = Cumulant(rxSig);
            t = t + toc;
            if(~strcmp(class,modulationPool{j}))
                error(j) = error(j) + 1;
            end
        end
    end
    disp('Error for each modulation type:')
    disp(error);
    disp('Average accuracy:')
    disp(1-sum(error)/4/num);
end
disp(t/num/4)



