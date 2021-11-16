if [ $# -eq 1 ]
then
    cd "$1"
fi
echo 'ctr,numcent,localisation,id_treatment,filename_dose_matrix'
#for dir in `ls -d */`
for dir in "Curie/" "GR/" "Nice/" "Reims/" "Toulouse/"
do
    ls $dir | awk -v dir=$dir -F"_" '{print $2","substr($3,1,length($3)-1)","substr(dir,1,length(dir)-1)","substr($3,length($3),length($3))","$0}'
done

