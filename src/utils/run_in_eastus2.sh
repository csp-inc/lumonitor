#!/usr/bin/env bash

GROUP=lumonitor-eastus2
NAME=lumonitor-eastus2

az container create \
  -g $GROUP \
  -l eastus2 \
  -f src/utils/lumonitor-eastus2.yaml \
  --subscription "Microsoft Azure Sponsorship AI for Earth 40K" \
  --image blindjesse/lumonitor:0.1 \
  --gitrepo-url https://github.com/csp-inc/lumonitor.git \
  --gitrepo-mount-path /lumonitor \
  --azure-file-volume-account-key $AZURE_EASTUS2_VOLUME_ACCOUNT_KEY \
  --azure-file-volume-account-name $AZURE_EASTUS2_VOLUME_ACCOUNT \
  --azure-file-volume-share-name data \
  --azure-file-volume-mount-path /lumonitor/data/lumonitor-eastus2 \
  --command-line "/bin/sh -c 'cd lumonitor;$@'" \
  --restart-policy Never
