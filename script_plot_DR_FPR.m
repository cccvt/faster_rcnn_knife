function script_plot_DR_FPR()

load('DR_conf.mat');
load('FPR_conf.mat');
plot(DR_conf(:,1),DR_conf(:,2),'r-',FPR_conf(:,1),FPR_conf(:,2),'b-');
xlabel 'confidence'
ylabel 'DR/FPR'
legend('DR-conf','FPR-conf','Location','southwest')
axis([0.8 1 0 1])
hold on;
grid;
plot(FPR_conf(:,1),FPR_conf(:,2),'b-');

end