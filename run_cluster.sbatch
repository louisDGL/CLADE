#!/bin/bash
#SBATCH --job-name=d22_cluster
#SBATCH --partition=SallesInfo
#SBATCH --array=1-51%51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
# One sample per node: --exclusive means a node is never shared with any other job
# (including sibling array tasks), so the 51 tasks land on 51 distinct nodes.
# "Most powerful nodes": this cluster has no schedulable features (ActiveFeatures=null),
# so we cannot --constraint. The strongest class is the 28-core nodes (vs 16/24/26-core);
# we exclude every non-28-core node, leaving SLURM to place all 51 tasks on 28-core nodes.
#SBATCH --exclude=ablette,aerides,ain,albatros,allemagne,anchois,angleterre,anguille,autriche,autruche,barbeau,barbue,barlia,baudroie,belgique,bengali,brochet,calanthe,carrelet,coucou,dindon,diuris,encyclia,epervier,epipactis,espagne,faisan,finlande,france,gardon,gelinotte,gennaria,groenland,gymnote,habenaria,harpie,hibou,hollande,hongrie,ipsea,irlande,islande,isotria,jabiru,kamiche,labre,lieu,linotte,liparis,lituanie,loriol,lotte,lycaste,malaxis,malte,monaco,mouette,mulet,murene,nandou,neotinea,ombrette,oncidium,ophrys,orchis,perdrix,piranha,pleione,pogonia,pologne,portugal,quetzal,quiscale,raie,requin,rouget,rouloul,roumanie,roussette,saumon,serapias,silure,sitelle,sole,suede,telipogon,test-252,thon,traquet,truite,urabu,vanda,vanilla,verdier,xylobium,zeuxine
# NOTE: submit from the repo root so the relative --output path resolves (logs/ must exist).
#SBATCH --output=logs/cluster_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
python3 steps/01_cluster.py
