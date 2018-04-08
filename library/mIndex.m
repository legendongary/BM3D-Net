function M = mIndex(WindowSize, P, Q)
nw = WindowSize;
hw = ( nw - 1 ) / 2;
M = ones(nw * nw, P, Q);
M = reshape(M, nw*nw, []);
[X, Y]=meshgrid(1 : P, 1 : Q);
X = X';
Y = Y';
co = 1;
for k = -hw : hw
    for l = -hw : hw
        st_x = X + l;
        st_y = Y + k;
        idSet = st_x > P | st_x < 1 | st_y > Q | st_y < 1;
        M(co, idSet) = inf;
        co = co + 1;
    end
end
M = reshape(M, nw * nw, P, Q);
end