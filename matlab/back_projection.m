function b_normal = back_projection(filename)
load('C:\Users\86193\Desktop\fu512.mat');

% load('.\set\xueguan512_200BW.mat');
% load('./希尔伯特变换/100BW.mat');
% load('C:\Users\123\Desktop\1883.mat');
% folder =  ('C:\Users\HP\Desktop\test_80BW\');
% folder1 = ('C:\Users\HP\Desktop\test_80BW\');
% for N= 11:30
%     N
% filename = strcat(folder,num2str(N),'.mat');
% load(filename);
% sensor_data = imread('C:\Users\123\Desktop\光声\ret.png');
sensor_data = imread(filename);
% sensor_data = imread('C:\Users\HP\Desktop\开题\60tr.png');
% sensor_data = sensor_100BW;
sensor_data = double(sensor_data);
    t_array = kgrid.t_array;
    dt = kgrid.dt;
    fs = 1/dt;
    np = length(t_array);
    NFFT = 2^nextpow2(np);
    f_vec = fs*linspace(0,1,NFFT);
    w_vec = 2*pi*f_vec;
    k_vec = ifftshift(2*pi*f_vec/(sound_speed));
    weighting_omega = 0;
    p0_recon = zeros(Nx, Ny);

    for ii = 1:num_sensor_points
        x_ii = cart_sensor_mask(1,ii);
        y_ii = cart_sensor_mask(2,ii);
        distance_x_square = (kgrid.x - x_ii).^2;
        distance_y_square = (kgrid.y - y_ii).^2;
        distance_xy = sqrt(distance_x_square + distance_y_square);
        distance_xy_index = round(distance_xy/(sound_speed*dt));
        distance_xy_index(~distance_xy_index)=1;
        p_ii = sensor_data(ii,:);
        p_ii_derivative0 = ifft((1i*k_vec).*fft(p_ii,NFFT));
        p_ii_derivative = real((p_ii_derivative0(1:np))) *0;
        bp_ii = (p_ii - sound_speed.*t_array.*p_ii_derivative);
        out_of_range = find(distance_xy_index > length(bp_ii));  %%为了防止超出索引范围
        distance_xy_index(out_of_range) = length(bp_ii);
        p0_ii = bp_ii(distance_xy_index);

%         figure(1)
%         imagesc((p0_ii))

        % weighting angle
        weighting_arc_ii = (2*pi*sensor_radius)/num_sensor_points;
        weighting_phy_ii = 2*pi/num_sensor_points;
        weighting_omega_ii = weighting_arc_ii*(1./((distance_x_square + distance_y_square).*sensor_radius).*(-x_ii.*kgrid.x - y_ii.*kgrid.y + sensor_radius.^2));
        weighting_omega_ii(isnan(weighting_omega_ii)) = weighting_phy_ii;
        weighting_omega_ii(isinf(weighting_omega_ii)) = weighting_phy_ii;
        p0_ii = weighting_omega_ii.*p0_ii;
        weighting_omega = weighting_omega + weighting_omega_ii;
        p0_recon = p0_recon + p0_ii;
    end

    p0_recon0 = p0_recon./(2*pi);
    b=p0_recon0(151:406,151:406); 
    
    b_max=max(max(b));
    b_min=min(min(b));
    b_normal=(b-b_min)/(b_max-b_min);
%     imshow(b_normal,[]);
%     imwrite(b_normal,'C:\Users\HP\Desktop\开题\p60tr.png');
%   filename1 = strcat(folder1,num2str(N),'bp','.png');
%   imwrite(b_normal,filename1);
    %s = imread('C:\Users\86193\Desktop\2.png');
    h = b_normal;  
    %[psnr,ssim] = psnr_comp(s,h); 
    %filename2 = strcat('C:\Users\86193\Desktop\1\','15checkpoint_1629bp','_','.png');
    %imwrite(b_normal,filename2);
% end
end
