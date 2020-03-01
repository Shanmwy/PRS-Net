for /d %%v in (./ShapeNetCore.v1/02691156/*) do (
cd ./ShapeNetCore.v1/02691156/%%v 
rename model.nrrd %%v.nrrd
cd ../../..)
for /r %%v in (*.nrrd) do (move %%v NrrdData)
cd ./NrrdData
for /f "delims=" %%a in ('dir /a-d/b') do (
set /a N+=1
call ren "%%~sa" "%%N%%.nrrd"
)