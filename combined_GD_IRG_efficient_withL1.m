clear 
clc
delete(findall(0,'Type','figure'))

%% generate a dataset from the cameraman image

bandw = 20;
[A, B, n] = data_set(bandw);

%% function definition

f = @(x,A,B) (norm(A*x - B))^2 ;
grad_f = @ (x,A,b) transpose(A)*(A*x-b);

% grad_f_block = @ (x,A_block,A,b) transpose(A_block)*(A*x-b); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% below is the gradiet with considering NORM-1
grad_f_block = @ (x,A_block,A,b,x_1) transpose(A_block)*(A*x-b)+sign(x_1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g = @(x) (x'*x)/2;
grad_g = @(x) x;


%% definition of regularization and stepsize parameters

gamma_0 = 1 - 10^(-2); eta_0 = gamma_0;
a = 0.5 + 10^(-3); b = 0.5 - 10^(-2);

Gamma = @(gamma0_vary, k) gamma0_vary*(k + 1)^(-a);
% Gamma_const = gamma_0*(10^5/2 + 1)^(-a);
Eta = @(k) eta_0*(k + 1)^(-b);

%% implementation of REGULARIZED GRADIENT DESCENT
for gamma_vary = [1]
     for Eta_vary = [0 0.001 0.01  0.1 1]
%      for Eta_vary = [0 0.022]

        iter1 = 10^5; % number of iterations
%         iter1 = 10^2;
        x = repmat(1000, n, 1);

        % Algorithm defined below:
        for k = 1:iter1
            h = waitbar(k/iter1);
            
%             x = x - gamma_vary*(grad_f(x,A,B) + Eta_vary*grad_g(x)); % x1 updated to x_k using Gradient descent (CONSTANT gamma)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % below is the gradiet with considering NORM-1
            x = x - gamma_vary*(grad_f(x,A,B) + Eta_vary*(grad_g(x)+sign(x))); % x1 updated to x_k using Gradient descent (CONSTANT gamma)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end

        X = reshape(x,[sqrt(n) sqrt(n)]); %B is sparse matrix
        X = full(X); %B is double
        X = uint8(X); % B is uint8 for imshow()
        F1 = figure; 
    %     title1 = ['Gradient descent: ' num2str(iter1) ' iteration & Eta = ', num2str(Eta_vary)];
        imshow(X), 
    % title(title1); 
        FileName=['GradientDescent Gamma',num2str(gamma_vary),' AND Eta = ',num2str(Eta_vary),'.png'];
        set(gcf,'PaperPositionMode','auto')
        saveas(F1, FileName);

     end
end

%% implementation of RANDOMIZED ITERATIVE REGULARIZED GRADIENT

clear x X

x = repmat(1000, n, 1);

% for gamma0_vary = [0.1, 1]
for gamma0_vary = [1]

% for i = 2:3
    for i = 1:6

        total_iterations = 10^i; % number of iterations

        % X = zeros(iter,n);
        number_of_blocks = 8;
        x = zeros(n,1);

        % Algorithm defined below:
        I=zeros(1,n/number_of_blocks);
        for itr = 1:total_iterations
            h = waitbar(itr/total_iterations);

        %     x = x- Gamma(itr)*(grad_f_block(x, A, A, B) + Eta(itr)*grad_g(x)); % x1 updated (without block)

            i_k = randi([1 number_of_blocks]); % generating a random number (within [0, 20]) following uniform distribution
            I = (n/number_of_blocks*(i_k-1)+1):n/number_of_blocks*i_k; % set of dimensions which get updated at this iteration
            A1 = A(:,min(I):max(I)); % selecting corresponding columns of A for update (n*I)
            x1 = x(min(I):max(I)); % selecting corresponding rows of x for update (I*1)
            B1 = B; % selecting whole B (n*1)

            x_k1 = x1- Gamma(gamma0_vary, itr)*(grad_f_block(x, A1, A, B1, x1) + Eta(itr)*grad_g(x1)); % x1 updated to x_k1
            %             x_p1 = point_projection(x_k1, A1); % projecting x_k1 onto
        %     x_p1 = x_k1; % without using projection

            if i_k ~= 1 
                x21 = x(1:min(I)-1); %tracking the unupdated dimensions of x, BEFORE I, and keeping them in x21 
            end

            if i_k ~= number_of_blocks
                x22 = x(max(I)+1:n); %tracking the unupdated dimensions of x, AFTER I, and keeping them in x21
            end

            if i_k ~= 1 && i_k ~= number_of_blocks
                x = [x21; x_k1; x22]; % stacking all the updated and non-updated values of x together.
            elseif i_k == 1 && number_of_blocks ~= 1
                x = [x_k1; x22]; % stacking all the updated and non-updated values of x together.
            elseif i_k == number_of_blocks && number_of_blocks ~= 1
                x = [x21; x_k1]; % stacking all the updated and non-updated values of x together.
            elseif number_of_blocks == 1
                x = x_k1;
            end

        end

        X_f=reshape(x,[sqrt(n) sqrt(n)]); %B is sparse matrix
        X_f=full(X_f); %B is double
        X_f=uint8(X_f); % B is uint8 for imshow()
        F = figure; 
        % title1 = ['IRG for 10^', num2str(i), ' iterations'];
        imshow(X_f), 
        % title(title1); 
        FileName=['IRG_10^',num2str(i),'at Gamma',num2str(gamma0_vary),'.png'];
        set(gcf,'PaperPositionMode','auto')
        saveas(F, FileName);

    end
end