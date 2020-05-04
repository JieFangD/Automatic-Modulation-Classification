function [ out ] = sig2pic_gaussian(sig,x0,x1,y0,y1,r0,r1)
linx = linspace(x0,x1,r0);
liny = linspace(y0,y1,r1);
[Px,Py] = meshgrid(linx(2:end-1),liny(2:end-1));
P = cat(3,Px,Py);
out = zeros(r0,r1);
for i = 1:size(sig,1)
    for j = 1:r0
        for k = 1:r1
            tmp =  sum((sig(i,:)-[linx(j),liny(k)]).^2);
            out(j,k) = out(j,k) + exp(-tmp/2/0.01);
        end
    end
end
