binvox.exe -d 32 -t nrrd .\ShapeNetCore.v1\02691156\10155655850468db78d106ce0a280f87\model.obj 
move .\ShapeNetCore.v1\02691156\10155655850468db78d106ce0a280f87\model.nrrd .\MatureData\1
SurfacePointsSamplingPCL.exe .\ShapeNetCore.v1\02691156\10155655850468db78d106ce0a280f87\model.obj  .\MatureData\1\model.pcd -n_samples 1000