clc; clear; close all;

global p % define a global variable for the plot handle
dx = 0.01;
dy = 0.01;
beta = 0.5;

x = -1:dx:1;
y = -1:dy:1;
[X,Y] = meshgrid(x,y);

% initialize some initial values for w1, w2, and T
w1 = 0.5;
w2 = 0.5;
T = 0.5;

f_act = 1./(1 + exp(-1*beta.*(w1*X+w2*Y-T))); % calculate the initial f_act values

% create a figure with three sliders and a plot
f = figure;
ax = axes('Parent',f,'position',[0.13 0.39  0.77 0.54]);

% Slide bar for w1
s_w1 = uicontrol('Parent',f,'Style','slider','Position',[81,110,419,23],...
              'value',w1, 'min',-5, 'max',5);
bgcolor = f.Color;
sl1 = uicontrol('Parent',f,'Style','text','Position',[50,110,23,23],...
                'String','-5','BackgroundColor',bgcolor);
sl2 = uicontrol('Parent',f,'Style','text','Position',[500,110,23,23],...
                'String','5','BackgroundColor',bgcolor);
sl3 = uicontrol('Parent',f,'Style','text','Position',[240,85,100,23],...
                'String','w1','BackgroundColor',bgcolor);
            
% Slide bar for w2
s_w2 = uicontrol('Parent',f,'Style','slider','Position',[81,70,419,23],...
              'value',w2, 'min',-5, 'max',5);
bgcolor = f.Color;
s21 = uicontrol('Parent',f,'Style','text','Position',[50,70,23,23],...
                'String','-5','BackgroundColor',bgcolor);
s22 = uicontrol('Parent',f,'Style','text','Position',[500,70,23,23],...
                'String','5','BackgroundColor',bgcolor);
s23 = uicontrol('Parent',f,'Style','text','Position',[240,45,100,23],...
                'String','w2','BackgroundColor',bgcolor);
            
% Slide bar for T
s_T = uicontrol('Parent',f,'Style','slider','Position',[81,30,419,23],...
              'value',T, 'min',-5, 'max',5);
bgcolor = f.Color;
s31 = uicontrol('Parent',f,'Style','text','Position',[50,30,23,23],...
                'String','-5','BackgroundColor',bgcolor);
s32 = uicontrol('Parent',f,'Style','text','Position',[500,30,23,23],...
                'String','5','BackgroundColor',bgcolor);
s33 = uicontrol('Parent',f,'Style','text','Position',[240,5,100,23],...
                'String','T','BackgroundColor',bgcolor);
            
p = mesh(ax, X, Y, f_act); % create a plot and store its handle in p
xlabel('X');
ylabel('Y');
colorbar;

% assign the same callback function to all sliders, and pass b_w1, b_w2, and b_T as additional arguments
s_w1.Callback = {@update, s_w1, s_w2, s_T};
s_w2.Callback = {@update, s_w1, s_w2, s_T};
s_T.Callback = {@update, s_w1, s_w2, s_T};

function update (src, event, b_w1, b_w2, b_T)
    global p % access the global variable p
    dx = 0.01;
    dy = 0.01;
    beta = 0.5;

    x = -1:dx:1;
    y = -1:dy:1;
    [X,Y] = meshgrid(x,y);

    w1 = get(b_w1, 'Value'); % get the current slider value for w1 using the get function
    w2 = get(b_w2, 'Value'); % get the current slider value for w2 using the get function
    T = get(b_T, 'Value'); % get the current slider value for T using the get function
    f_act = 1./(1 + exp(-1*beta.*(w1*X+w2*Y-T))); % calculate the new f_act values
    p.ZData = f_act; % update the plot 
end