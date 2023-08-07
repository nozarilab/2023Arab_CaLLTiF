clc
close all

load 'hcp_rest_average_cortical_causal_flow_and_degree.mat'

plot_Schaefer100(causal_flow')

saveas(gcf,'average_causal_flow.png')

close all

plot_Schaefer100(degree')

saveas(gcf,'average_degree.png')
