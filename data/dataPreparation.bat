setlocal enabledelayedexpansion
set /a count = 0
for /d %%v in (./ShapeNetCore.v1/02691156/*) do (
set /a count += 1
md .\MatureData\!count!
binvox.exe -d 32 -t nrrd .\ShapeNetCore.v1\02691156\%%v\model.obj 
move .\ShapeNetCore.v1\02691156\%%v\model.nrrd .\MatureData\!count!
SurfacePointsSamplingPCL.exe .\ShapeNetCore.v1\02691156\%%v\model.obj  .\MatureData\!count!\model.pcd -n_samples 1000 -no_vis_result
)
ENDLOCAL
