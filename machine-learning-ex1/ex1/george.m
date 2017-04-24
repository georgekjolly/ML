function george()

data = load('ex1data2.txt')

mu = [0,0];

for i=1:length(mu)
mu(i) = std( data(:,i));
end

for i=1:length(mu)
fprintf('mu val = %f\n',mu(i));
end

end
