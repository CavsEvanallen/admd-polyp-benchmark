#!/usr/bin/env bash
# 所有 GPU 脚本共用。先改这几行,再 `source env.sh`。目录结构沿用 ADMD提交4:
#   manifests/loso/holdout_Ck/seedS/{train,val,test_*}.csv
#   runs[_后缀]/loso/holdout_Ck/<method>/seedS[_pat30]/best.pt
#   results[_后缀]/loso/holdout_Ck/<method>/<TestSet>/<prompt>/seedS/{metrics.csv,gating.csv,masks/}
export WORK=${WORK:-./work}          # 你们的工作目录(含 manifests/ prompts/ cache/ ckpt/ scripts/ admd_plugin/)
export PLUGIN=${PLUGIN:-$WORK/admd_plugin}           # 含 scripts/train.py infer.py
export SCRIPTS=${SCRIPTS:-$WORK/scripts}             # 含打过补丁的 04_metrics.py 03_prompts.py 06_dedup_check.py(dataset__stem 版)
export PROMPTS=${PROMPTS:-$WORK/prompts/loso}        # 提示文件;若你们的目录名不同,只改这一行。预期: $PROMPTS/holdout_Ck/seedS/test_<TestSet>.json
export CENTRES=${CENTRES:-"C1 C2 C3 C4 C5"}
export SEEDS=${SEEDS:-"0 1 2"}
export PUBLIC=${PUBLIC:-1}                            # 1=同时评估 CVC-ColonDB / ETIS-LaribPolypDB / CVC-300
export DRY=${DRY:-0}                                  # DRY=1 只打印命令不执行(先用它检查路径)

enc_setup () {   # $1 = mobilesam | sam_b
  ENC=$1
  if [ "$ENC" = mobilesam ]; then CKPT=$WORK/ckpt/mobile_sam.pt; CACHE=$WORK/cache; SFX=""
  else CKPT=$WORK/ckpt/sam_vit_b_01ec64.pth; CACHE=$WORK/cache_sam_b; SFX="_sam_b"; fi
}

# 方法 -> train.py 参数(与 ADMD提交4/runs/*/config.json 一致)
method_args () {
  case $1 in
    sam_ft)       echo "--decoder sam" ;;
    wide_sam)     echo "--decoder sam --wide" ;;          # 需要你们现用的 train.py 支持 --wide(提交4 的 config.json 里有 wide 字段)
    admd)         echo "--decoder admd" ;;
    uniform_ens)  echo "--decoder admd --admd_query off --admd_align off --fusion uniform --domain_loss_w 0" ;;
    noquery)      echo "--decoder admd --admd_query off" ;;       # B2 消融
    noalign)      echo "--decoder admd --admd_align off" ;;
    nodomsup)     echo "--decoder admd --domain_loss_w 0" ;;
    *) echo "unknown method $1" >&2; return 1 ;;
  esac
}

run () { echo "+ $*"; if [ "$DRY" != 1 ]; then "$@"; fi; }

test_sets () {  # $1 = centre
  echo "Source_$1"; if [ "$PUBLIC" = 1 ]; then echo CVC-ColonDB ETIS-LaribPolypDB CVC-300; fi
}

# train_one ENC METHOD CENTRE SEED PATIENCE RUNSFX(如 _pat30 或空) RUNROOT
train_one () {
  local enc=$1 m=$2 c=$3 s=$4 pat=$5 tag=$6 runroot=$7; enc_setup $enc
  local out=$WORK/$runroot/loso/holdout_$c/$m/seed$s$tag
  if [ -f $out/best.pt ]; then echo "skip train (exists) $out"; return; fi
  run python $PLUGIN/scripts/train.py --manifest_dir $WORK/manifests/loso/holdout_$c/seed$s --encoder $enc \
      --ckpt $CKPT --cache_dir $CACHE $(method_args $m) --patience $pat --seed $s --out $out
}

# infer_one ENC METHOD CENTRE SEED RUNSFX RUNROOT RESROOT [FUSION] [KEEPMASKS=0]
infer_one () {
  local enc=$1 m=$2 c=$3 s=$4 tag=$5 runroot=$6 resroot=$7 fus=$8 keep=${9:-0}; enc_setup $enc
  local run_dir=$WORK/$runroot/loso/holdout_$c/$m/seed$s$tag
  for t in $(test_sets $c); do for p in 1pt bbox; do
    local o=$WORK/$resroot/loso/holdout_$c/$m/$t/$p/seed$s
    if [ -f $o/metrics.csv ]; then echo "skip infer (exists) $o"; continue; fi
    run python $PLUGIN/scripts/infer.py --run $run_dir --manifest $WORK/manifests/loso/holdout_$c/seed$s/test_$t.csv \
        --prompts $PROMPTS/holdout_$c/seed$s/test_$t.json --prompt $p --cache_dir $CACHE --ckpt $CKPT --out $o ${fus:+--fusion $fus}
    run python $SCRIPTS/04_metrics.py --manifest $WORK/manifests/loso/holdout_$c/seed$s/test_$t.csv --pred_dir $o/masks --out $o/metrics.csv
    if [ "$keep" != 1 ] && [ "$DRY" != 1 ]; then rm -rf $o/masks; fi     # 默认删掩码以节省空间(metrics 已算完)
  done; done
}
