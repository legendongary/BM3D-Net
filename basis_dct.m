N = 7*7*8;
DCT = dctmtx(7);
DCT = kron(DCT, DCT);
Haar = [1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2));
    1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2)), 1/(2 * sqrt(2)), -1/(2 * sqrt(2)), -1/(2 * sqrt(2)), -1/(2 * sqrt(2)), -1/(2 * sqrt(2));
    1/2, 1/2, -1/2, -1/2, 0, 0, 0, 0;
    0, 0, 0, 0, 1/2, 1/2, -1/2, -1/2;
    1/(sqrt(2)), -1/(sqrt(2)), 0, 0, 0, 0, 0, 0;
    0, 0, 1/(sqrt(2)), -1/(sqrt(2)), 0, 0, 0, 0;
    0, 0, 0, 0, 1/(sqrt(2)), -1/(sqrt(2)), 0, 0;
    0, 0, 0, 0, 0, 0, 1/(sqrt(2)), -1/(sqrt(2))];
basis = zeros(N, N);
for i=1:49
    for j=1:8
        filter = DCT(i, :)'*Haar(j, :);
        filter = reshape(filter, [], 1);
        basis(:, 8*(i-1)+j) = filter;
    end;
end;