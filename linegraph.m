function [A,degree]=linegraph(n,radius)

xy = [(1:n)', ones(n,1)];
%--> distance matrix of all pairs of nodes
Md=sqrt( (xy(:,1)*ones(1,n)-ones(n,1)*xy(:,1).').^2+(xy(:,2)*ones(1,n)-ones(n,1)*xy(:,2).').^2);
%---> Adjacency matrix
A=((Md+2*radius*eye(n))<radius)*eye(n);

degree=A*ones(n,1);
%---> Laplacian
end
