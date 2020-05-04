function [ out ] = sig2pic(sig,x0,x1,y0,y1,r0,r1)
linx = linspace(x0,x1,r0);
liny = linspace(y0,y1,r1);
out = ones(r0,r1);
for i = 1:size(sig,1)/2
    [~,x] = min(abs(linx-sig(i,1)));
    [~,y] = min(abs(liny-sig(i,2)));
    out(x,y) = 0;
end