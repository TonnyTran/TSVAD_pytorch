#!/usr/bin/env bash

set -e -u -o pipefail

################################################################################
# Configuration
################################################################################
# Use a no scoring collar of +/ "collar" seconds around each boundary.
collar=0.0

# Step size in seconds to use in computation of JER.
step=0.010

# If provided, output full scoring logs to this directory. It will contain the
# following files:
# - metrics_full.stdout  --  per-file and overall metrics for full test set; see
#   dscore documentation for explanation
# - metrics_full.stderr  --  warnings/errors produced by dscore for full test set
# - metrics_core.stdout  --  per-file and overall metrics for core test set
# - metrics_core.stderr  --  warnings/errors produced by dscore for full test set
scores_dir="./individual_scores"


################################################################################
# Parse options, etc.
################################################################################
if [ -f path.sh ]; then
    . ./path.sh;
fi

if [ -f cmd.sh ]; then
    . ./cmd.sh;
fi
. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "usage: $0 <release-dir> <rttm-dir>"
  echo "e.g.: $0 /data/corpora/LDC2020E12 exp/diarization_dev/rttms"
  exit 1;
fi

# Root of official LDC release; e.g., /data/corpora/LDC2020E12.
release_dir=$1

refer_rttm=$2

# Directory containing RTTMs to be scored.
sys_rttm=$3


################################################################################
# Score.
################################################################################
# Create temp directory for dscore outputs.
tmpdir=$(mktemp -d -t dh3-dscore-XXXXXXXX)

echo "-----------------------------"
echo "collar: $collar"
echo "Step size: $step"
echo "tmpdir: $tmpdir"
echo "release_dir: $release_dir"
echo "refer_rttm: $refer_rttm"
echo "sys_rttm: $sys_rttm"
echo "-----------------------------"


echo "usage: $0 score CORE set"
echo "$0: ***** RESULTS *****   DER          MISS       FA          ERR"

# Score CORE test set.
for domain in audiobooks broadcast_interview clinical court cts maptask meeting restaurant socio_field socio_lab webvideo; do
    
    ./utils/rttm_from_uem.py $refer_rttm $release_dir/data/uem_scoring/core/$domain.uem $tmpdir ref
    ./utils/rttm_from_uem.py $sys_rttm $release_dir/data/uem_scoring/core/$domain.uem $tmpdir sys

    perl ./utils/md-eval.pl -c $collar\
      -r $tmpdir/ref_$domain.rttm \
      -s $tmpdir/sys_$domain.rttm \
      >  $tmpdir/$domain.der \
      2> /dev/null
    
    der=$(grep OVERALL $tmpdir/$domain.der | awk '{print $6}')
    miss=$(grep 'MISSED SPEAKER' $tmpdir/$domain.der | cut -c 39-43)
    fa=$(grep 'FALARM SPEAKER' $tmpdir/$domain.der | cut -c 39-43)
    err=$(grep 'SPEAKER ERROR TIME' $tmpdir/$domain.der | cut -c 39-43)
    printf "*** DER (core) - %-25s : %-10s  %-10s  %-10s  %-10s \n"  ${domain} ${der} $miss $fa $err
    
done

echo "---------------------------------------------------------------------------"

for domain in one_spk two_spk three_spk four_spk five_to_ten; do
    ./utils/rttm_from_uem.py $refer_rttm $release_dir/data/uem_scoring/core/num_spk/$domain.uem $tmpdir ref
    ./utils/rttm_from_uem.py $sys_rttm $release_dir/data/uem_scoring/core/num_spk/$domain.uem $tmpdir sys

    perl ./utils/md-eval.pl -c $collar\
      -r $tmpdir/ref_$domain.rttm \
      -s $tmpdir/sys_$domain.rttm \
      >  $tmpdir/$domain.der \
      2> /dev/null
    der=$(grep OVERALL $tmpdir/$domain.der | awk '{print $6}')
    miss=$(grep 'MISSED SPEAKER' $tmpdir/$domain.der | cut -c 39-43)
    fa=$(grep 'FALARM SPEAKER' $tmpdir/$domain.der | cut -c 39-43)
    err=$(grep 'SPEAKER ERROR TIME' $tmpdir/$domain.der | cut -c 39-43)
    printf "*** DER (core) - %-25s : %-10s  %-10s  %-10s  %-10s \n"  ${domain} ${der} $miss $fa $err
done

echo "----------------------------------------------------------------------------"

for domain in all; do
    ./utils/rttm_from_uem.py $refer_rttm $release_dir/data/uem_scoring/core/$domain.uem $tmpdir ref
    ./utils/rttm_from_uem.py $sys_rttm $release_dir/data/uem_scoring/core/$domain.uem $tmpdir sys

    perl ./utils/md-eval.pl -c $collar\
      -r $tmpdir/ref_$domain.rttm \
      -s $tmpdir/sys_$domain.rttm \
      >  $tmpdir/$domain.der \
      2> /dev/null
    der=$(grep OVERALL $tmpdir/$domain.der | awk '{print $6}')
    miss=$(grep 'MISSED SPEAKER' $tmpdir/$domain.der | cut -c 39-43)
    fa=$(grep 'FALARM SPEAKER' $tmpdir/$domain.der | cut -c 39-43)
    err=$(grep 'SPEAKER ERROR TIME' $tmpdir/$domain.der | cut -c 39-43)
    printf "*** DER (core) - %-25s : %-10s  %-10s  %-10s  %-10s \n"  ${domain} ${der} $miss $fa $err
done

if [ ! -z "scores_dir" ]; then
 echo "$0: ***"
 echo "$0: *** Full results are located at: ${scores_dir}"
fi

# Clean up.
if [ ! -z "scores_dir" ]; then
  mkdir -p $scores_dir
  cp $tmpdir/* $scores_dir
fi
rm -fr $tmpdir
