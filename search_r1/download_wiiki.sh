#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/search_r1"
save_path="${SAVE_PATH:-${repo_root}/local_dense_retriever}"
hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${save_path}"
cd "${repo_root}"

export HF_ENDPOINT="${hf_endpoint}"

download_tool="${HFD_TOOL:-}"
if [[ -z "${download_tool}" ]]; then
  if command -v aria2c >/dev/null 2>&1; then
    download_tool="aria2c"
  elif command -v wget >/dev/null 2>&1; then
    download_tool="wget"
  else
    echo "Neither aria2c nor wget is installed. Please install one of them first." >&2
    exit 1
  fi
fi

bash "${repo_root}/hfd.sh" PeterJinGo/wiki-18-e5-index \
  --dataset \
  --include part_aa part_ab \
  --local-dir "${save_path}" \
  --tool "${download_tool}" \
  -x "${HFD_THREADS:-4}" \
  -j "${HFD_JOBS:-5}"

# hfd caches repo metadata under the target directory. Clear it before downloading
# another repo into the same directory so the second file list is generated from
# the correct dataset metadata.
rm -rf "${save_path}/.hfd"

bash "${repo_root}/hfd.sh" PeterJinGo/wiki-18-corpus \
  --dataset \
  --include wiki-18.jsonl.gz \
  --local-dir "${save_path}" \
  --tool "${download_tool}" \
  -x "${HFD_THREADS:-4}" \
  -j "${HFD_JOBS:-5}"

cat "${save_path}"/part_* > "${save_path}/e5_Flat.index"
if [[ -f "${save_path}/wiki-18.jsonl.gz" ]]; then
  gzip -df "${save_path}/wiki-18.jsonl.gz"
fi
