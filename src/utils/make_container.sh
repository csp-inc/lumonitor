#!/usr/bin/env bash

GROUP=lumonitor-eastus2
NAME=lumonitor-eastus2

az container create \
  -g $GROUP \
  -n $NAME \
  -l eastus2 \
  --subscription "Microsoft Azure Sponsorship AI for Earth 40K" \
  --image blindjesse/lumonitor:0.1 \
  --gitrepo-url https://github.com/csp-inc/lumonitor.git \
  --gitrepo-mount-path /lumonitor \
  --azure-file-volume-account-key M8UkwC4ZnbUuLPT+/7oZVMm1oqCVAfpc5rJVOc42+kzUr6wUo3yo7pYXS8KNpCiXeVbjgK5DyUKFTKuw/J8NgA== \
  --azure-file-volume-account-name lumonitoreastus2 \
  --azure-file-volume-share-name data \
  --azure-file-volume-mount-path /lumonitor/data/lumonitor-eastus2 \
  --command-line "/bin/sh -c '$@'" \
  --restart-policy Never
