PROBLEM=score2perf_vip_perf_conditional_aug_10x
BUCKET=vip-bucket
PROJECT=vip-music-331820

PIPELINE_OPTIONS=\
"--region=us-east1,"\
"--runner=DataflowRunner,"\
"--project=${PROJECT},"\
"--temp_location=gs://${BUCKET}/tmp,"\
"--setup_file=/home/david/Code/vip/magenta/setup.py"

t2t_datagen \
  --data_dir=gs://${BUCKET}/datagen \
  --problem=${PROBLEM} \
  --pipeline_options="${PIPELINE_OPTIONS}" \
  --alsologtostderr
