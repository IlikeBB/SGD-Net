TemplateDir=${1}
# TemplateDir=./MNIs/
cd ${2}
STARTTIME=$(date +"%s")
# reorient image 
/usr/local/fsl/bin/fslreorient2std ${3} diff_reoriented
/usr/local/fsl/bin/fslreorient2std ${4} t1_reorient.nii.gz
/usr/local/fsl/bin/fslreorient2std ${5} lesion_reorient.nii.gz

# brain extraction for nonlinear registration to MNI152
/usr/local/fsl/bin/bet t1_reorient.nii.gz t1_bet.nii.gz -R -f 0.15 -g 0 -m

#  non bet registration is better than bet 
/usr/local/fsl/bin/flirt -in t1_reorient.nii.gz -ref diff_reoriented.nii.gz -omat str2epi.mat -out str2epi.nii.gz
/usr/local/fsl/bin/convert_xfm -inverse str2epi.mat -omat epi2str.mat

# T1 register to MNI
/usr/local/fsl/bin/flirt -ref ${TemplateDir}/MNI152_T1_2mm_brain.nii.gz -in t1_bet.nii.gz -omat str2mni_affine_transf.mat
/usr/local/fsl/bin/fnirt --in=t1_reorient.nii.gz --ref=${TemplateDir}/MNI152_T1_2mm.nii.gz --aff=str2mni_affine_transf.mat --cout=str2mni_nonlinear_transf --config=T1_2_MNI152_2mm

# for registration checking
/usr/local/fsl/bin/applywarp --in=t1_reorient.nii.gz  --ref=${TemplateDir}/MNI152_T1_2mm.nii.gz --warp=str2mni_nonlinear_transf.nii.gz --out=t12MNI

# Combine matrix that register diffusion space to MNI (premat*nonlineartrans)
/usr/local/fsl/bin/convertwarp --ref=${TemplateDir}/MNI152_T1_2mm_brain.nii.gz --premat=epi2str.mat --warp1=str2mni_nonlinear_transf --out=comprehensive_warps.nii.gz --relout

# apply warp (diff to MNI)

/usr/local/fsl/bin/applywarp --in=diff_reoriented.nii.gz --ref=${TemplateDir}/MNI152_T1_2mm_brain.nii.gz --warp=comprehensive_warps --rel --out=diff2MNI.nii.gz
/usr/local/fsl/bin/applywarp --in=lesion_reorient.nii.gz --ref=${TemplateDir}/MNI152_T1_2mm_brain.nii.gz --warp=comprehensive_warps --rel --out=lesion2MNI.nii.gz --interp=nn
# ------------command ends here---------------- 


# information of lesion location output into csv file
#echo 'Lesion Percentage: The percenatag of lesion across all brain regions' > LesionMapping_${6}.csv
#echo 'Region Percnetage: the percentage of lesion in a specific brain region' >> LesionMapping_${6}.csv

# load BA(48)/AAL3(170)/JHU_WM(48) info
for name in BA AAL3 JHU JUELICH_25; do    
    if [[ ${name} == 'BA' ]]; then
        tmp=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -k lesion2MNI.nii.gz -H 49 0 48)
        atlas=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -H 49 0 48)
        echo 'BA regions, Region Names,Lesion voxels, Lesion Percentage,Region Percnetage' >> LesionMapping_${6}.csv
    elif [[ ${name} == 'AAL3' ]]; then
        tmp=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -k lesion2MNI.nii.gz -H 171 0 170)
        atlas=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -H 171 0 170)
        echo 'AAL3 regions,Region Names,Lesion voxels, Lesion Percentage,Region Percnetage' >> LesionMapping_${6}.csv
    elif [[ ${name} == 'JHU' ]]; then
        tmp=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -k lesion2MNI.nii.gz -H 49 0 48)
        atlas=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -H 49 0 48)      
        echo 'JHU WM labels,Region Names,Lesion voxels, Lesion Percentage,Region Percnetage' >> LesionMapping_${6}.csv  
    elif [[ ${name} == 'JUELICH_25' ]]; then
        tmp=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -k lesion2MNI.nii.gz -H 122 0 121)
        atlas=$(/usr/local/fsl/bin/fslstats ${TemplateDir}/${name}_MNI.nii.gz -H 122 0 121)      
        echo 'JUELICH labels,Region Names,Lesion voxels, Lesion Percentage,Region Percnetage' >> LesionMapping_${6}.csv 
    fi
     
    t=0
    sum=0
    PA=''
    AT=''
    region=''
    vox=''
    for i in ${tmp}; do
        if [[ $(echo "${i} > 0"|bc) -eq 1 ]]; then
            sum=$(echo ${sum}+${i}|bc)
            region=${region}' '${t}
            vox=${vox}' '$(echo "${i}/1"|bc)
        fi
        t=$((${t}+1))
    done
    t=1
    for i in ${vox}; do
        # tmp=$(echo "scale=4;${i}/${sum}"|bc)
        PA=${PA}' '$(echo "scale=4;100*${i}/${sum}"|bc)
        r=$(echo ${region} |cut -d ' ' -f ${t})
        r=$((${r}+1))
        rv=$(echo ${atlas} |cut -d ' ' -f ${r})
        AT=${AT}' '$(echo "scale=4;100*${i}/${rv}"|bc)
        t=$((${t}+1))
    done

    t=1
    for i in ${region}; do
        # if [[ ${i} -ge 0 ]]; then
        A=$(cat ${TemplateDir}/${name}.txt | awk 'NR=='${i}+1' {print $1}')
        B=$(cat ${TemplateDir}/${name}.txt | awk 'NR=='${i}+1' {print $2}')
        C=$(echo ${vox} |cut -d ' ' -f ${t})
        D=$(echo ${PA} |cut -d ' ' -f ${t})
        E=$(echo ${AT} |cut -d ' ' -f ${t})
        echo ${name} ${A}','${B}','${C}','${D}','${E}>> LesionMapping_${6}.csv
        t=$((${t}+1))
    done

    echo '' >> LesionMapping_${6}.csv
done

ENDTIME=$(date +"%s")
duration=$((${ENDTIME} - ${STARTTIME}))
echo "-------------$(date +"%Y-%m-%d %T") ## $((duration / 60)):$((duration % 60))"
