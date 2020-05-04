close all;
M = [4 8 16 64];

symbol = 1000;
resolution = 36;
num = 1;
SNR = [20];
x = zeros(num*4,symbol);
y = zeros(num*4,1);

for k = 1:length(SNR)
    for j = 1:4
        for i = 1:num
            data = randi([0 M(j)-1],symbol,1);
            if(j == 1)
                txSig = pskmod(data,M(j));
            elseif(j == 2)
                txSig = pskmod(data,M(j));
            else
                txSig = qammod(data,M(j));            
                txSig = txSig*1/sqrt(2/3*(M(j)-1));
            end
            rxSig = awgn(txSig,SNR(k),'measured');
            x(i+num*(j-1),:) = reshape(rxSig,1,symbol);
            y(i+num*(j-1),:) = j-1;
        end
        figure('position', [500, 500, 500, 500]);
        
        % plot gray Cartesian Coordinate figure
%         rxSig = [real(rxSig),imag(rxSig)];
%         pic = sig2pic(rxSig,-1.5,1.5,-1.5,1.5,resolution,resolution);
%         image = imagesc(pic);
%         colormap gray
%         axis off

        % plot gray accumulated Polar Coordinate figure with Gaussian
%         rxSig = [abs(rxSig),atan2(real(rxSig),imag(rxSig))];
%         pic = sig2pic_gaussian(rxSig,0,2,-3.2,3.2,resolution,resolution);
%         pic = pic < 0.8;
%         image = imagesc(pic);
%         colormap gray

        % plot colorful accumulated Cartesian Coordinate figure
%         rxSig = [real(rxSig),imag(rxSig)];
%         pic = sig2pic_accu(rxSig,-1.5,1.5,-1.5,1.5,resolution,resolution);
%         image = imagesc(1-pic/(max(pic(:))));
%         colormap hot
%         axis off

        % plot colorful accumulated Polar Coordinate figure
        rxSig = [abs(rxSig),atan2(real(rxSig),imag(rxSig))];
        pic = sig2pic_accu(rxSig,0,2,-3.2,3.2,resolution,resolution);
        image = imagesc(1-pic/(max(pic(:))));
        colormap hot
        axis off
        
        % plot colorful accumulated Polar Coordinate figure with Gaussian
%         rxSig = [abs(rxSig),atan2(real(rxSig),imag(rxSig))];
%         pic = sig2pic_gaussian(rxSig,0,2,-3.2,3.2,resolution,resolution);
%         image = imagesc(1-pic/(max(pic(:))));
%         colormap hot
%         axis off
    end
    filename = strcat('raw',num2str(num),'_',num2str(symbol),'_',num2str(SNR(k)),'.mat');
    save(filename,'y','x');  
end

