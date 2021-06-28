#!/bin/zsh

regex=$1
out_name=$2

results_dir="/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results"

outpath="$results_dir/combined/$out_name"
outpath2="$results_dir/combined/r_$out_name"
tmp_name=$(openssl rand -hex 6)
tmp="$results_dir/combined/$tmp_name"

files=$(ls $results_dir | grep "$regex")
echo "Merging: $files"
header_file=$(ls $results_dir | grep "$regex" | head -n1)
# Get header
echo "Header file: $header_file"
head -n1 "$results_dir/$header_file" >"$outpath"
# Remove hash from start of chrm header
sed '1s/^.//' "$outpath" >"$tmp" && mv "$tmp" "$outpath"
# Merge all files without header
for fname in $files; do
  tail -n+2 "$results_dir/$fname" >>"$tmp"
done
# Sort by chr and pos
sort -k1 -k2 -n $tmp >>$outpath
rm $tmp

#cp "$outpath" "$outpath2"
#echo "Saved and sorted to: $outpath2"

transfer_path="/home/vauvelle/pycharm-sftp/pheprob/results/transfer/."
gzip "$outpath"
#mv "${outpath}.gz" transfer_path

#echo "Also gziped and copied to: $transfer_path"
