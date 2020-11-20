#!/usr/bin/env bash

CMD="$@"

az container create \
  -g lumonitor-eastus2 \
  --gitrepo-url https://github.com/csp-inc/lumonitor.git \
  --gitrepo-dir \lumonitor \
  -l eastus2 \
  --subscription "Microsoft Azure Sponsorship AI for Earth 40K" \
#  --subscription 57fdfbf2-6e1d-4d46-99e6-607c2f5dcbdb \
  --image blindjesse/lumonitor:0.1 \
  --azure-file-volume-account-key M8UkwC4ZnbUuLPT+/7oZVMm1oqCVAfpc5rJVOc42+kzUr6wUo3yo7pYXS8KNpCiXeVbjgK5DyUKFTKuw/J8NgA== \
  --azure-file-volume-account-name lumonitoreastus2 \
  --azure-file-volume-mount-path \data \
  --azure-file-volume-share-name data



  
