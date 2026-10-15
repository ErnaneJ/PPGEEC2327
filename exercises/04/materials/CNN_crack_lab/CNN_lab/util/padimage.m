function v=padimage(v0,nbcz,nbcx)
v=[repmat(v0(:,1),1,nbcx), v0, repmat(v0(:,end),1,nbcx)];
v=[repmat(v(1,:),nbcz,1); v; repmat(v(end,:),nbcz,1)];
end