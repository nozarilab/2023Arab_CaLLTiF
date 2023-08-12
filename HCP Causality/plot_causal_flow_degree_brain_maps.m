clc
close all

load '../Results/hcp_rest_average_cortical_causal_flow_and_degree.mat'

plot_Schaefer100(causal_flow')

saveas(gcf,'../Results/average_causal_flow.png')

close all

addpath('../External Packages and Files/plot_Schaefer100')

plot_Schaefer100(degree')

saveas(gcf,'../Results/average_degree.png')
