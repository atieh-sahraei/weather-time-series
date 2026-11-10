%% MTK337 Assignment 3: LGA weather time series


%% Clear workspace
clear; clc; close all;
folder = 'Weather_Assignment3_results';
if ~exist(folder,'dir'), mkdir(folder); end
diary(fullfile(folder,'Run_log.txt'));
fprintf('Run date: %s\nMATLAB: %s\n', datestr(now), version);

%% Read the data and select LaGuardia
T = readtable('weather.csv','TreatAsEmpty',{'NA'});
W = T(strcmp(string(T.origin),'LGA'),:);
variables = {'temp','dewp','humid','wind_speed','precip'};
units = {'deg F','deg F','percent','mph','inches'};
data = W{:,variables};
% Build local clock times from the numeric columns; keep original row order.
time = datetime(W.year,W.month,W.day,W.hour,0,0);
n = height(W);
nTrain = 8500;
if n <= nTrain, error('Not enough LGA observations for the requested split.'); end
if any(~isfinite(data(:)))
    error('Unexpected missing values. Inspect the file before continuing.');
end
if any(diff(time)<seconds(0))
    error('Rows are not in chronological order. Check timestamps first.');
end
nTest = n-nTrain;
fprintf('\nLGA observations: %d\nTraining: %d\nTest: %d\n',n,nTrain,nTest);
fprintf('Last training observation: %s\n',char(time(nTrain)));
fprintf('Test period: %s to %s\n',char(time(nTrain+1)),char(time(end)));

%% Record timestamp problems rather than hide them
gapHours = hours(diff(time));
bad = find(gapHours ~= 1);
timeAudit = table(bad,bad+1,time(bad),time(bad+1),gapHours(bad), ...
    'VariableNames',{'PreviousObservation','NextObservation','PreviousTime', ...
    'NextTime','ClockDifferenceHours'});
disp(timeAudit);
writetable(timeAudit,fullfile(folder,'Timestamp_audit.csv'));
fprintf('Model lags refer to recorded observation steps, not exact hours.\n');
fprintf('The repeated November clock hour is retained (daylight-saving change).\n');
W.Observation = (1:n)';
W.Set = repmat("Training",n,1);
W.Set(nTrain+1:end) = "Test";
writetable(W,fullfile(folder,'LGA_data_and_split.csv'));

%% Descriptive statistics on the training set
train = data(1:nTrain,:);
statistics = table(string(variables)',string(units)',sum(isnan(train))', ...
    mean(train)',median(train)',std(train)',min(train)',max(train)', ...
    'VariableNames',{'Variable','Unit','Missing','Mean','Median','SD','Min','Max'});
disp(statistics);
writetable(statistics,fullfile(folder,'Training_statistics.csv'));


%% Variables ready for import into Econometric Modeler
% Use Import > workspace data in the app. Select each variable and its ACF.
temp = train(:,1);
dewp = train(:,2);
humid = train(:,3);
wind_speed = train(:,4);
precip = train(:,5);

%% Time-series plots: training data for model exploration
figure('Color','w','Position',[100 100 1200 900]);
for j=1:5
    subplot(5,1,j);
    plot(time(1:nTrain),train(:,j)); grid on;
    ylabel([variables{j} ' (' units{j} ')']);
end
sgtitle('LGA weather: first 8500 observations');
saveas(gcf,fullfile(folder,'01_Training_time_series.png'));

%% ACF: similarity with earlier observations
figure('Color','w','Position',[100 100 1200 900]);
for j=1:5
    subplot(3,2,j);
    autocorr(train(:,j),'NumLags',168);
    title(variables{j}); xlabel('Lag (observation steps)');
end
sgtitle('Training ACF: up to 168 observation steps');
saveas(gcf,fullfile(folder,'02_Training_ACF.png'));

figure('Color','w','Position',[100 100 1200 700]);
for j=1:5
    subplot(2,3,j); histogram(train(:,j));
    xlabel([variables{j} ' (' units{j} ')']); ylabel('Count'); grid on;
end
sgtitle('Training distributions');
saveas(gcf,fullfile(folder,'03_Training_histograms.png'));

% A unit-root test is supplementary; it does not rule out seasonality.
% H=1 rejects a unit root under this particular test specification.
ADF_H = zeros(5,1); ADF_p = zeros(5,1);
for j=1:5
    [ADF_H(j),ADF_p(j)] = adftest(train(:,j),'Model','ARD','Lags',24);
end
stationarity = table(string(variables)',ADF_H,ADF_p, ...
    'VariableNames',{'Variable','RejectUnitRoot','ADF_pValue'});
disp(stationarity);
writetable(stationarity,fullfile(folder,'Training_ADF_tests.csv'));

%% regARMA: one secondary predictor at a time, never dew point
% temp(t) = intercept + beta*x(t) + u(t), where u(t) follows ARMA(p,q).
% contemporaneous x is suitable for association/conditional modelling.
% Future measured x is NOT a legitimate input to an unconditional forecast.
% These regressions are compared in training; the test forecast below is VAR.
predictors = {'humid','wind_speed','precip'};
columns = [3 4 5];
AR_orders = [0 1 2 3 6 24];
MA_orders = [0 1 2];
% Only q=0 is fitted for p=24 to keep the larger model manageable.
% Total: 48 candidates. The p=24 model includes ALL lags 1 through 24.
modelList = {};
resultRows = {};
modelNumber = 0;
y = temp;
for j=1:3
    x = train(:,columns(j));
    for p=AR_orders
        for q=MA_orders
            if p==24 && q>0, continue; end
            modelNumber = modelNumber+1;
            fprintf('regARMA %d: %s, AR=%d, MA=%d\n',modelNumber,predictors{j},p,q);
            try
                model = regARIMA(p,0,q);
                [fitted,~,logL,info] = estimate(model,y,'X',x,'Display','off');
                e = infer(fitted,y,'X',x);
             
                k = p+q+3;
                aic = -2*logL+2*k;
                bic = -2*logL+k*log(nTrain);
                % Ignore the same first 24 innovations in every diagnostic.
                e = e(25:end);
                [~,pv] = lbqtest(e,'Lags',[48 96],'DoF',[48 96]-p-q);
                ok = info.exitflag>0 && isfinite(aic);
                pass = all(pv>0.05);
                modelList{modelNumber} = fitted;
                resultRows(end+1,:) = {modelNumber,string(predictors{j}),p,q, ...
                    k,logL,aic,bic,pv(1),pv(2),sqrt(mean(e.^2)), ...
                    info.exitflag,ok,pass,""};
            catch ME
                modelList{modelNumber} = [];
                resultRows(end+1,:) = {modelNumber,string(predictors{j}),p,q, ...
                    p+q+3,NaN,Inf,Inf,NaN,NaN,NaN,NaN,false,false,string(ME.message)};
                fprintf('Candidate failed: %s\n',ME.message);
            end
        end
    end
end
regResults = cell2table(resultRows,'VariableNames',{'ModelID','Predictor', ...
    'ARorder','MAorder','Parameters','LogLikelihood','AIC','BIC', ...
    'LB_p48','LB_p96','InnovationRMSE','ExitFlag','Converged', ...
    'PassLjungBox','ErrorMessage'});
regResults = sortrows(regResults,'AIC');
disp(regResults);
writetable(regResults,fullfile(folder,'regARMA_all_candidates.csv'));

%% Best candidate for each predictor

bestRows = zeros(3,1);
for j=1:3
    rows = find(regResults.Predictor==string(predictors{j}) & regResults.Converged);
    if isempty(rows)
        error('No converged regARMA candidate for %s. See Run_log.txt.',predictors{j});
    end
    good = rows(regResults.PassLjungBox(rows));
    if ~isempty(good)
        bestRows(j) = good(1);
    else
        bestRows(j) = rows(1);
        fprintf('WARNING: no %s candidate passes both residual tests.\n',predictors{j});
    end
end
bestReg = regResults(bestRows,:);
disp(bestReg);
writetable(bestReg,fullfile(folder,'regARMA_best_by_predictor.csv'));
choices = find(regResults.Converged & regResults.PassLjungBox);
if isempty(choices)
    choices = find(regResults.Converged);
    fprintf('No candidate passes both tests. Lowest AIC model is provisional.\n');
end
chosenReg = regResults(choices(1),:);
disp('Overall regARMA choice (inspect its diagnostics before accepting):');
disp(chosenReg);
writetable(chosenReg,fullfile(folder,'regARMA_selected.csv'));

%% Residual plots for the three selected regARMA models
figure('Color','w','Position',[100 100 1300 850]);
for j=1:3
    fitted = modelList{bestReg.ModelID(j)};
    e = infer(fitted,y,'X',train(:,columns(j)));
    e = e(25:end);
    subplot(3,3,3*j-2); plot(25:nTrain,e); grid on;
    title(sprintf('%s: AR %d, MA %d',predictors{j},bestReg.ARorder(j),bestReg.MAorder(j)));
    xlabel('Observation'); ylabel('Innovation (deg F)');
    subplot(3,3,3*j-1); autocorr(e,'NumLags',96); title('Innovation ACF');
    xlabel('Lag (observation steps)');
    subplot(3,3,3*j); histogram(e); title('Innovation distribution');
    xlabel('Innovation (deg F)'); ylabel('Count');
end
saveas(gcf,fullfile(folder,'04_regARMA_diagnostics.png'));

%% VAR: temperature, humidity and wind speed together
Y = data(:,[1 3 4]);
Ytrain = Y(1:nTrain,:);
Ytest = Y(nTrain+1:end,:);
% Training-only scaling helps with the different variable units.
mu = mean(Ytrain);
sigma = std(Ytrain);
Ztrain = (Ytrain-mu)./sigma;
scaling = table(["temp";"humid";"wind_speed"],mu',sigma', ...
    'VariableNames',{'Variable','TrainingMean','TrainingSD'});
writetable(scaling,fullfile(folder,'VAR_scaling.csv'));

%% Compare VAR orders using the SAME likelihood sample
VAR_orders = [1 2 3 6 12 24];
presample = max(VAR_orders);
% All models use rows 25:8500 as responses. Rows 1:24 initialize their lags.
% Thus all information used for estimation is within the first 8500 rows.
VARmodels = cell(length(VAR_orders),1);
VARerrors = cell(length(VAR_orders),1);
varRows = {};
for j=1:length(VAR_orders)
    p = VAR_orders(j);
    fprintf('Estimating VAR(%d)\n',p);
    try
        model = varm(3,p);
        [fitted,~,logL,e] = estimate(model,Ztrain(presample+1:end,:), ...
            'Y0',Ztrain(1:presample,:));
        % 3 constants, 9*p lag coefficients, 6 covariance parameters.
        k = 3+9*p+6;
        aic = -2*logL+2*k;
        bic = -2*logL+k*log(nTrain-presample);
        % Stationarity: all companion-matrix eigenvalues must be below 1.
        A = [fitted.AR{:}];
        if p>1
            A = [A; eye(3*(p-1)) zeros(3*(p-1),3)];
        end
        radius = max(abs(eig(A)));
        VARmodels{j} = fitted;
        VARerrors{j} = e;
        varRows(end+1,:) = {j,p,k,logL,aic,bic,radius,radius<1,""};
    catch ME
        varRows(end+1,:) = {j,p,3+9*p+6,NaN,Inf,Inf,NaN,false,string(ME.message)};
        fprintf('VAR candidate failed: %s\n',ME.message);
    end
end
varResults = cell2table(varRows,'VariableNames',{'ModelID','Order','Parameters', ...
    'LogLikelihood','AIC','BIC','MaxEigenvalue','Stable','ErrorMessage'});
varResults = sortrows(varResults,'AIC');
disp(varResults);
writetable(varResults,fullfile(folder,'VAR_order_comparison.csv'));
rows = find(varResults.Stable & isfinite(varResults.AIC));
if isempty(rows), error('No stable VAR candidate. Inspect the model results.'); end
chosenVAR = varResults(rows(1),:);
VARmodel = VARmodels{chosenVAR.ModelID};
eVAR = VARerrors{chosenVAR.ModelID};
disp('Selected stable VAR with lowest AIC:'); disp(chosenVAR);
writetable(chosenVAR,fullfile(folder,'VAR_selected.csv'));
% AIC values of VAR and regARMA cannot be compared with each other:
% different response dimensions/scaling/likelihood samples are involved.

%% VAR residual diagnostics for all three responses
% Univariate ACFs are useful but are not a full multivariate whiteness test.
figure('Color','w','Position',[100 100 1200 750]);
varNames = {'temp','humid','wind_speed'};
for j=1:3
    subplot(3,2,2*j-1);
    plot(presample+1:nTrain,eVAR(:,j)*sigma(j)); grid on;
    title([varNames{j} ' innovations']); xlabel('Observation');
    subplot(3,2,2*j);
    autocorr(eVAR(:,j),'NumLags',96); title([varNames{j} ' innovation ACF']);
    xlabel('Lag (observation steps)');
end
saveas(gcf,fullfile(folder,'05_VAR_diagnostics.png'));

%% Forecast the ENTIRE test period from the end of the training data
% Future humidity, wind and temperature are all unknown here.

[Zforecast,forecastMSE] = forecast(VARmodel,nTest,Ztrain);
Yforecast = Zforecast.*sigma+mu;
predicted = Yforecast(:,1);
actual = Ytest(:,1);
% Approximate 95% intervals conditional on estimated model parameters.
% They do not account for parameter uncertainty or model misspecification.
forecastSE = zeros(nTest,1);
for h=1:nTest
    forecastSE(h) = sqrt(max(0,forecastMSE{h}(1,1)))*sigma(1);
end
lower = predicted-1.96*forecastSE;
upper = predicted+1.96*forecastSE;

%% Baselines using only training information
persistence = repmat(Ytrain(end,1),nTest,1);
trainingMean = repmat(mu(1),nTest,1);
% Repeats the final 24 recorded temperatures, NOT exactly the previous day.
pattern24 = Ytrain(end-23:end,1);
repeat24 = pattern24(mod((0:nTest-1)',24)+1);
forecasts = [predicted persistence trainingMean repeat24];
modelNames = ["VAR";"Last observation";"Training mean";"Repeat last 24 observations"];
RMSE = zeros(4,1); MAE = zeros(4,1); R2 = zeros(4,1); Bias = zeros(4,1);
for j=1:4
    errorValues = forecasts(:,j)-actual;
    RMSE(j) = sqrt(mean(errorValues.^2));
    MAE(j) = mean(abs(errorValues));
    R2(j) = 1-sum(errorValues.^2)/sum((actual-mean(actual)).^2);
    Bias(j) = mean(errorValues);
end
metrics = table(modelNames,RMSE,MAE,R2,Bias, ...
    'VariableNames',{'Model','RMSE_degF','MAE_degF','R2','Bias_degF'});
disp(metrics);
writetable(metrics,fullfile(folder,'Test_performance.csv'));
coverage = mean(actual>=lower & actual<=upper)*100;
fprintf('VAR approximate 95%% interval coverage on test: %.1f%%\n',coverage);

predictionTable = table((nTrain+1:n)',time(nTrain+1:end),actual,predicted, ...
    lower,upper,actual-predicted,Yforecast(:,2),Yforecast(:,3), ...
    persistence,trainingMean,repeat24,'VariableNames',{'Observation','LocalTime', ...
    'ActualTemp','VARtemp','Lower95','Upper95','Residual', ...
    'VARhumid','VARwindSpeed','Persistence','TrainingMean','Repeat24'});
writetable(predictionTable,fullfile(folder,'Test_predictions.csv'));

%% Plots for the report
steps = (1:nTest)';
figure('Color','w','Position',[100 100 1200 550]);
fill([steps;flipud(steps)],[lower;flipud(upper)],[.88 .92 1], ...
    'EdgeColor','none'); hold on;
plot(steps,actual,'k','LineWidth',1.3);
plot(steps,predicted,'b','LineWidth',1.5);
plot(steps,persistence,'--','Color',[.7 .3 .1]);
legend('Approximate 95% interval','Measured','VAR forecast','Persistence', ...
    'Location','best'); grid on;
xlabel('Forecast horizon (recorded observations after row 8500)');
ylabel('Temperature (deg F)');
title(sprintf('LGA: fixed-origin VAR(%d) forecast',chosenVAR.Order));
saveas(gcf,fullfile(folder,'06_VAR_test_forecast.png'));

figure('Color','w','Position',[100 100 1100 700]);
subplot(2,2,1); scatter(actual,predicted,15,'filled'); hold on;
limits = [min([actual;predicted]) max([actual;predicted])];
plot(limits,limits,'k--'); grid on;
xlabel('Measured (deg F)'); ylabel('Forecast (deg F)'); title('Test predictions');
subplot(2,2,2); plot(steps,actual-predicted); yline(0,'k--'); grid on;
xlabel('Forecast horizon'); ylabel('Measured - forecast (deg F)');
subplot(2,2,3); histogram(actual-predicted);
xlabel('Residual (deg F)'); ylabel('Count');
subplot(2,2,4); bar(RMSE); xticks(1:4);
xticklabels({'VAR','Last value','Mean','Repeat 24'});
ylabel('Test RMSE (deg F)'); grid on;
saveas(gcf,fullfile(folder,'07_Test_errors_and_baselines.png'));

%% Save the models and finish
save(fullfile(folder,'Weather_models.mat'),'modelList','bestReg','chosenReg', ...
    'VARmodel','chosenVAR','mu','sigma','nTrain','presample','variables', ...
    'regResults','varResults','metrics','coverage');
fprintf('\nResults saved in %s\n',folder);
fprintf('Next: examine residual plots, then write the report.\n');
fprintf('Low AIC alone does not establish an adequate weather model.\n');
fprintf('Discuss seasonality, timestamp gaps and the long forecast horizon.\n');
fprintf('Do not claim classification was performed: this is regression.\n');
diary off;
