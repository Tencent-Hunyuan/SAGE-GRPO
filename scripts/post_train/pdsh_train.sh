#!/bin/bash
set -x

if [ $# -ne 1 ]; then
  echo "Usage: $0 <script>"
  exit 1
fi

script=$1
export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')

if [ ! -d $PROJECT_BASE ]; then
    echo "PROJECT_BASE not exist: $PROJECT_BASE"
    exit 1
fi

CURRENT_DIR=$(pwd)
worker_num=$(( $NODE_NUM /8 ))

pdsh -w $node_ip -f $worker_num "cd ${CURRENT_DIR}; bash $script"
