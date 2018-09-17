DATADIR = "lp_data";
if ~exist(fullfile(DATADIR, "mat"), 'dir')
    mkdir(fullfile(DATADIR, "mat"));
end
files = dir(fullfile(DATADIR, "mps", "*.mps"));
for i = 1:length(files)
    fprintf("Reading %s\n", files(i).name);
    m = mpsread(fullfile(DATADIR, "mps", files(i).name));
    n = length(m.f);
    c = m.f;
    A = sparse([m.Aineq; m.Aeq; speye(n)]);
    u = [m.bineq; m.beq; m.ub];
    l = [-Inf; m.beq; m.lb];
    int_idx = m.intcon;
    save(fullfile(DATADIR, "mat", files(i).name(1:end-4) + ".mat"), ...
        'c','l', 'A', 'u', 'int_idx');
end
