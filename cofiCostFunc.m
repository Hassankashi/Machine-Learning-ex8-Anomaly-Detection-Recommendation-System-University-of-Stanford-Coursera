function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

sumJ = 0;

 

    
% Y = [1682 x 944] ---- [movies x users]
% R = [1682 x 944] ---- [movies x users]
% X = [1682 x 10]  ---- [movies x 10-features]
% Theta = [944 x  10] - [user x 10-features]
% for one movie, say movie = 1, find all users who have rated it: 
%     idx1 = find (R(i,:)==1)
%     so that is in idx1 [1 x 453]

   

% Find the Theta values for those users
%    ThetaTemp which is [453 * 10] 
 
% ThetaTemp = zeros( 1 ,n);
% YTeamp = zeros( 1 ,n);
% 
% for i=1:num_movies
%        
%         %ThetaTemp=R( find(idx == 1),: );
%          for myJ=1:n
%                  %ThetaTemp = zeros( 1 ,6);
%                 
%                 %       idx1 = find (R(i,myJ)==1);
%                 %       [m2 n2] = size(idx1);
% 
%                      if (R(i,myJ)==1)
%                                               
%                         ThetaTemp(1,myJ) = Theta(find (R(i,myJ)==1));
%                         
%                            YTeamp(1,myJ)  = Y( find(R(i,myJ)==1));
%                            
%                           
%                      end
%         end
%    
% end

%Theta2=sum(sum(R.*Theta));
costfunc=zeros(size(Y));
Xgrad=zeros(size(X));
Thetagrad=zeros(size(Theta));


costfunc=(X*Theta'-Y).*R;

J = (sum(sum((costfunc).^2)))/2;

%%%%%%%%%%%

 [m, n] = size(X);
 [mR nR] = size(R); 
 [mY nY] = size(Y); 
 [mT nT] = size(Theta); 
%%%%%%%%%%
%%%%%%%%%%%%%Xgrad=sum(sum(costfunc))*(R((1:mT),(1:nT)).*Theta);
Xgrad=costfunc*Theta+lambda.*X;
X_grad = Xgrad;
%%%%%%%%%% Xgrad=(costfunc*Theta);
% 
%%%%%%%%%%%%%Thetagrad=(costfunc'*X);
%costfuncX=((X*Theta'-Y)'*X).*R;
Tcostfunc=transpose(costfunc);
%Thetagrad=sum(sum(costfunc));
idx = find(R==1);

Thetagrad=costfunc'*X+lambda.*Theta;
Theta_grad = Thetagrad;


grad = [X_grad(:); Theta_grad(:)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%J%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%REQULARIZATION%%%%%%%%%%%%%%%%%%%%%%%%%5
reg=(lambda/2)*(sum(sum(Theta.^2))+sum(sum(X.^2)))
J=J+reg;
% J = (sum(((X*Theta'-Y).*R).^2))/m;
%%%%%%%%%%%%%5 sumJ =sum((X* ThetaTemp-YTeamp).^2);   
%%%%%%%%%%%J = sumJ/num_movies;
 %%%%%%%%%%%%%%%%%%%ThetaTemp=Theta( find(R(i,:)==1),: );

  %  ThetaTemp = zeros( n2 , n);

%    --- all Theta parameters for the above 453 users
% Find the ratings for that movie from those users - YTeamp = [1 x 453]
%%%%%%%%%%%%%%%%%%%%%YTeamp=Y( find(R(i,:)==1),: );
%    YTeamp = zeros(1,n2);
    

% Now, ratings of that movie by those users predicted will be
%    X [i,:] * ThetaTemp' = [1 x 10] x [10 x 453] = [1 x 453]
% sumJ =sumJ+ ((X(i,:)* ThetaTemp'-YTeamp).^2);
   
   
% Subtract off the actual ratings to get the delta which is 
%    X [i,:] * ThetaTemp' - YTemp - which is [1 x 453] 

% Multiply the above [1 x 453] by ThetaTemp which is [453 x 10] 
%   to get a single row vector that is the gradient for that movie [1 x 10]

% for j=1:num_users
%     
%     sumJ = sumJ+((X(i,:)* Theta(:,j)'-Y(i,j)).^2);
%    
%     
% end


    


%sumJ = sum(((X * ThetaTemp') - YTemp).^2);
%J = (1/2)*sum((X*theta - y).^2);

%sumJ = sum((X* transpose(Theta)-Y).^2);

            
  

%if R(i,j)=1  J = 














% =============================================================



end
