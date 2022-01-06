#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
TMP_FILE=${SCRIPT_DIR}/.tmp

find ${SCRIPT_DIR}/content/galleries -name _sources.txt -type f > ${TMP_FILE}
while IFS= read -r file
do (
    DIR=$(dirname $(realpath $file))
    echo "Processing ${DIR}";
    cd "${DIR}" &&
    wget --input-file=_sources.txt --continue --show-progress
)
done < ${TMP_FILE}
rm ${TMP_FILE}
