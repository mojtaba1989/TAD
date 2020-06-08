%% U V UV
clear all
clc
rng(61)
l = 1.23;d = 1.07;
f=2.1e-3; mu = 3.75e-6;cx = 640;cy=480;K = [-f/mu 0 cx;0 -f/mu cy;0 0 1];gamma = -6;h_z = .7;h_y =-.3;h_x =  0;
%u
for i = 1:10000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    uerror = normrnd(0,2,2,2);
    Puv = Puv + [uerror(1,:);0 0];
    U(i,1) =mean(uerror(1,:));
%     UV(i,1) =mean(uerror(1,:));
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f ; normrnd(0,f*.03);
    cxhat(i,1) = cx ; normrnd(0,cx*.03);
    cyhat(i,1) = cy ; normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma ; normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end
y = U;em = max(abs(error));
[Tq,yq] = meshgrid(-40:40, min(y):range(y)/100:max(y));
eq = griddata(T,y,error,Tq,yq,'nearest');
figure(1)
contourf(Tq,yq,eq,linspace(-em,em,100),'LineStyle','none');hold on
plot([-40,40],[0, 0],'k','LineWidth',1.5);hold off
colorbar;
set(gca, 'FontSize',14)
title('Error (degree)','FontSize',12);
ylabel('$$\Delta u~\textrm{(pixel)}$$','Interpreter','latex','FontSize',16);
xlabel({'$$\textrm{True Hitch-Angle (degree)}$$'},'Interpreter','latex','FontSize',16);
set(figure(1), 'Position', [100, 100, 600, 300])
saveas(figure(1), 'Vmodel','png')
%v
for i = 1:10000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    uerror = normrnd(0,2,2,2);
    Puv = Puv + [0 0;uerror(1,:)];
    V(i,1) = mean(uerror(1,:));
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f ; normrnd(0,f*.03);
    cxhat(i,1) = cx ; normrnd(0,cx*.03);
    cyhat(i,1) = cy ; normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma ; normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end
y = V;em = max(abs(error));
[Tq,yq] = meshgrid(-40:40, min(y):range(y)/100:max(y));
eq = griddata(T,y,error,Tq,yq,'nearest');
figure(1)
contourf(Tq,yq,eq,linspace(-em,em,100),'LineStyle','none');hold on
plot([-40,40],[0, 0],'k','LineWidth',1.5);hold off
colorbar;
set(gca, 'FontSize',14)
title('Error (degree)','FontSize',12);
xlabel({'$$\textrm{True Hitch-Angle (degree)}$$'},'Interpreter','latex','FontSize',16);
ylabel('$$\Delta v~\textrm{(pixel)}$$','Interpreter','latex','FontSize',16);
set(figure(1), 'Position', [100, 100, 600, 300])
saveas(figure(1), 'Umodel','png')
%UV
for i = 1:10000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    uerror = normrnd(0,2,2,2);
    Puv = Puv + uerror;
    UV(i,1) =mean(sqrt(sum(uerror.^2)));
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f ; normrnd(0,f*.03);
    cxhat(i,1) = cx ; normrnd(0,cx*.03);
    cyhat(i,1) = cy ; normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma ; normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end
el =.1:.1:5;
clear se
for i = 1:length(el)
    se(i) = var(error(UV<el(i)));
end
figure(1)
plot(el,se, 'LineWidth',1.5)
set(gca, 'FontSize',14)
xlabel({'$$\max(\sqrt{\Delta u^2+\Delta v^2})$$'},'Interpreter','latex','FontSize',16)
ylabel('$$\textrm{var}(\hat{\theta}_h -\theta_h)$$','Interpreter','latex','FontSize',16)
grid on
set(figure(1), 'Position', [100, 100, 600, 300])
saveas(figure(1), 'UVmodel','png')

%% f cx cy theta_c
clear all
clc
rng(61)
l = 1.23;d = 1.07;
f=2.1e-3; mu = 3.75e-6;cx = 640;cy=480;K = [-f/mu 0 cx;0 -f/mu cy;0 0 1];gamma = -6;h_z = .7;h_y =-.3;h_x =  0;
%cx
for i = 1:10000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f ; normrnd(0,f*.03);
    cxhat(i,1) = cx + normrnd(0,cx*.03);
    cyhat(i,1) = cy ; normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma ; normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end
y = cxhat;em = max(abs(error));
[Tq,yq] = meshgrid(-40:40, min(y):range(y)/100:max(y));
eq = griddata(T,y,error,Tq,yq,'v4');
figure(1)
contourf(Tq,yq-640,eq,linspace(-em,em,100),'LineStyle','none');hold on
plot([-40,40],[0, 0],'k','LineWidth',1.5);hold off
colorbar;
set(gca, 'FontSize',14)
title('Error (degree)','FontSize',12);
ylabel('$$\Delta c_x~\textrm{(pixel)}$$','Interpreter','latex','FontSize',16);
xlabel('True Hitch-Angle (degree)','FontSize',14);
set(figure(1), 'Position', [100, 100, 600, 300])
saveas(figure(1), 'Cxmodel','png')

%%
%cy
for i = 1:10000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f ; normrnd(0,f*.03);
    cxhat(i,1) = cx ; normrnd(0,cx*.03);
    cyhat(i,1) = cy + normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma ; normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end
y = cyhat;em = max(abs(error));
[Tq,yq] = meshgrid(-40:40, min(y):range(y)/100:max(y));
eq = griddata(T,y,error,Tq,yq,'v4');
figure(2)
contourf(Tq,yq-480,eq,linspace(-em,em,100),'LineStyle','none');hold on
plot([-40,40],[0, 0],'k','LineWidth',1.5);hold off
colorbar;
set(gca, 'FontSize',14)
title('Error (degree)','FontSize',12);
ylabel('$$\Delta c_y~\textrm{(pixel)}$$','Interpreter','latex','FontSize',16);
xlabel('True Hitch-Angle (degree)','FontSize',14);
set(figure(2), 'Position', [100, 100, 600, 300])
saveas(figure(2), 'Cymodel','png')
%f
for i = 1:10000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f + normrnd(0,f*.03);
    cxhat(i,1) = cx ; normrnd(0,cx*.03);
    cyhat(i,1) = cy ; normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma ; normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end
y = fhat;em = max(abs(error));
[Tq,yq] = meshgrid(-40:40, min(y):range(y)/100:max(y));
eq = griddata(T,y,error,Tq,yq,'v4');
figure(3)
contourf(Tq,yq*1000-02.1,eq,linspace(-em,em,100),'LineStyle','none');hold on
plot([-40,40],[0, 0],'k','LineWidth',1.5);hold off
colorbar;
set(gca, 'FontSize',14)
title('Error (degree)','FontSize',12);
ylabel('$$\Delta f~\textrm{(mm)}$$','Interpreter','latex','FontSize',16);
xlabel('True Hitch-Angle (degree)','FontSize',14);
set(figure(3), 'Position', [100, 100, 600, 300])
saveas(figure(3), 'Fmodel','png')
% theta_c
for i = 1:10000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f ; normrnd(0,f*.03);
    cxhat(i,1) = cx ; normrnd(0,cx*.03);
    cyhat(i,1) = cy ; normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma + normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
    error(i,1) = That(i,1)-T(i,1);
end
y = gammahat;em = max(abs(error));
[Tq,yq] = meshgrid(-40:40, min(y):range(y)/100:max(y));
eq = griddata(T,y,error,Tq,yq,'v4');
figure(4)
contourf(Tq,yq+6,eq,linspace(-em,em,100),'LineStyle','none');hold on
plot([-40,40],[0, 0],'k','LineWidth',1.5);hold off
colorbar;
set(gca, 'FontSize',14)
title('Error (degree)','FontSize',12);
ylabel('$$\Delta \theta_c~\textrm{(degree)}$$','Interpreter','latex','FontSize',16);
xlabel('True Hitch-Angle (degree)','FontSize',14);
set(figure(4), 'Position', [100, 100, 600, 300])
saveas(figure(4), 'Gammamodel','png')

%% f cx cy theta_c
clear all
close all
clc
rng(61)
l = 1.23;d = 1.07;
f=2.1e-3; mu = 3.75e-6;cx = 640;cy=480;K = [-f/mu 0 cx;0 -f/mu cy;0 0 1];gamma = -6;h_z = .7;h_y =-.3;h_x =  0;
for i = 1:100000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f + normrnd(0,f*.03);
    cxhat(i,1) = cx + normrnd(0,cx*.03);
    cyhat(i,1) = cy + normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma ; normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end

%f
el =[1.9:.1:2.3]*.001;
clear se
for i = 1:length(el)
    datax = T(abs(fhat-el(i))<= 5e-5);
    datay = That(abs(fhat-el(i))<= 5e-5);
    dc = fitlm(datax,datay);
    se(i) = dc.Coefficients.Estimate(2)-1;
end
figure(1);
plot(el*1000-2.1,se, 'LineWidth',1.5)
set(gca, 'FontSize',12)
xlabel('$$\Delta f~(mm)$$','Interpreter','latex','FontSize',16)
ylabel('$$\textrm{mean}(\hat{\theta}_h/\theta_h-1)$$','Interpreter','latex','FontSize',16)
grid on
set(figure(1), 'Position', [100, 100, 600, 300])
saveas(figure(1), 'Fcorr','png')

%cx
el =590:10:690;
clear se
for i = 1:length(el)
    se(i) = mean(error(abs(cxhat-el(i))<= 5));
end
figure(2);
plot(el-640,se, 'LineWidth',1.5)
set(gca, 'FontSize',12)
xlabel('$$\Delta c_x~\textrm{(pixel)}$$','Interpreter','latex','FontSize',16)
ylabel('$$\textrm{mean}(\hat{\theta}_h-\theta_h)$$','Interpreter','latex','FontSize',16)
grid on
set(figure(2), 'Position', [100, 100, 600, 300])
saveas(figure(2), 'Cxcorr','png')

%cy
el =450:10:510;
clear se
for i = 1:length(el)
    datax = T(abs(cyhat-el(i))<= 5);
    datay = That(abs(cyhat-el(i))<= 5);
    dc = fitlm(datax,datay);
    se(i) = dc.Coefficients.Estimate(2)-1;
end
figure(3);
plot(el-480,se, 'LineWidth',1.5)
set(gca, 'FontSize',12)
xlabel('$$\Delta c_y~\textrm{(pixel)}$$','Interpreter','latex','FontSize',16)
ylabel('$$\textrm{mean}(\hat{\theta}_h/\theta_h-1)$$','Interpreter','latex','FontSize',16)
grid on
set(figure(3), 'Position', [100, 100, 600, 300])
saveas(figure(3), 'Cycorr','png')

% theta_c
l = 1.23;d = 1.07;
f=2.1e-3; mu = 3.75e-6;cx = 640;cy=480;K = [-f/mu 0 cx;0 -f/mu cy;0 0 1];gamma = -6;h_z = .7;h_y =-.3;h_x =  0;
for i = 1:100000
    T(i,1) = unifrnd(-40,40);
    P = corner(T(i,1),l,d);
    Puv = cam_uv(P,h_x,h_y,h_z,gamma,K); 
    hx(i,1) = h_x ; normrnd(0,abs(h_x)*.03);
    hy(i,1) = h_y ; normrnd(0,abs(h_y)*.03);
    hz(i,1) = h_z ; normrnd(0,abs(h_z)*.03);
    lhat (i,1) = l ; normrnd(0,l*.03);
    dhat (i,1) = d ; normrnd(0,d*.03);
    fhat(i,1) = f ; normrnd(0,f*.03);
    cxhat(i,1) = cx ; normrnd(0,cx*.03);
    cyhat(i,1) = cy ; normrnd(0,cy*.03);
    Khat = [-fhat(i,1)/mu 0 cxhat(i,1);0 -fhat(i,1)/mu cyhat(i,1);0 0 1];
    gammahat(i,1) = gamma + normrnd(0,abs(gamma)*.03);
    That(i,1) = hitch_angle(Puv,hx(i,1),hy(i,1),hz(i,1),lhat(i,1),dhat(i,1),gammahat(i,1),Khat);
%     That(i,1) = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K);
    error(i,1) = That(i,1)-T(i,1);
end

el =-6.6:.1:-5.4;
clear se
for i = 1:length(el)
    datax = T(abs(gammahat-el(i))<= 0.05);
    datay = That(abs(gammahat-el(i))<= 0.05);
    dc = fitlm(datax,datay);
    se(i) = dc.Coefficients.Estimate(2)-1;
end
figure(4);
plot(el+6,se, 'LineWidth',1.5)
set(gca, 'FontSize',12)
xlabel('$$\Delta \theta_c~\textrm{(degree)}$$','Interpreter','latex','FontSize',16)
ylabel('$$\textrm{mean}(\hat{\theta}_h/\theta_h-1)$$','Interpreter','latex','FontSize',16)
grid on
set(figure(4), 'Position', [100, 100, 600, 300])
saveas(figure(4), 'Gammacorr','png')




%%
el =.1:.1:5;
clear se
for i = 1:length(el)
    se(i) = var(error(UV<el(i)));
%     datax = T(abs(cyhat-el(i))<= 5);
%     datay = That(abs(cyhat-el(i))<= 5);
%     dc = fitlm(datax,datay);
%     se(i) = dc.Coefficients.Estimate(2)-1;
end
figure(1);
plot(el,se, 'LineWidth',1.5)
xlabel('$$\hat{c}_y-c_y~(pixel)$$','Interpreter','latex','FontSize',16)
xlabel('$$\max(\sqrt{\Delta u^2+\Delta v^2})$$','Interpreter','latex','FontSize',16)
% ylabel('$$var(\hat{\theta}_h /\theta_h-1)$$','Interpreter','latex','FontSize',14)
ylabel('$$var(\hat{\theta}_h -\theta_h)$$','Interpreter','latex','FontSize',14)

set(gca, 'FontSize',12)
xlim()
grid on
set(gcf, 'Position', [100, 100, 600, 300])
saveas(figure(1), 'UVsimulationError','png')



%%
myq(5.2705)
global en
en = normrnd(0,1,1e8,1);


function Puv = cam_uv(P,h_x,h_y,h_z,gamma,K)
    %P=[x,y]'
    C = cosd(gamma);
    S = sind(gamma);
    Tr = [1 0 0;0 C S;0 -S C];
    Tc = [-1 0 0;0 0 1;0 1 0];
    Td = [h_x h_y h_z]';
    P = [P;[0 0]];
    P = (Tr*Tc)\(P-Td);
    P = P./P(3,:);
    Puv = round(K*P);Puv = Puv(1:2,:);
end

function P = corner(T,l,d)
    r=sqrt(l^2+d^2/4);
    w = atand(d/2/l);
    P = [r*sind(T-w) r*sind(T+w); r*cosd(T-w) r*cosd(T+w)];
end

function theta = hitch_angle(Puv,h_x,h_y,h_z,l,d,gamma,K)
    %P=[x,y]'
    C = cosd(gamma);
    S = sind(gamma);
    Tr = [1 0 0;0 C S;0 -S C];
    Tc = [-1 0 0;0 0 1;0 1 0];
    Td = [h_x h_y h_z]';
    Z = h_z./(S-C*(Puv(2,:)-K(2,3))/K(2,2));
    Puv(Puv<1) = NaN;
    Puv(1,Puv(1,:)>=1280) = NaN;
    Puv(2,Puv(2,:)>=960)  = NaN;
    Puv = [Puv;[1 1]];
    P = K\Puv.*Z;
    P = (Tr*Tc)*P+Td;
    P = P(1:2,:);
    Pp = P(:,1);
    if isnan(Pp(1))
        Pp = NaN;
    end
    Pd = P(:,2);
    if isnan(Pd(1))
        Pd = NaN;
    end
    omega = atand(d/2/l);
    
    if ~isnan(Pp(1))&&~isnan(Pd(1))
        Ppr = rotate(Pp,-omega);
        Pdr = rotate(Pd,omega);
        m = (Ppr+Pdr)/2;
        theta = atand(m(1)/m(2));
    elseif ~isnan(Pp(1))
        theta = atand(Pp(1)/Pp(2))+omega;
    elseif ~isnan(Pd(1))
        theta = atand(Pd(1)/Pd(2))-omega;
    end
end

function out = rotate(P,omega)
    C = cosd(omega);
    S = sind(omega);
    T = [C -S;S C];
    out = T*P;
end

function out = myq(x)
    global en
    out = mean(en > x);
end
