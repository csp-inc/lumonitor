#!/usr/bin/env bash

# Needs AZURE_SUBSCRIPTION_ID var set or specified using --azure-subscription-id

docker-machine create -d azure \
  --azure-location eastus2 \
  --azure-resource-group lumonitor-eastus2 \
  --azure-ssh-user jesse \
  --azure-size Standard_E8s_v3 \
  --azure-open-port 80 \
  azurerocks
