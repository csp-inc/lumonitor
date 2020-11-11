#!/usr/bin/env bash

az group create --name lumonitor-eastus2 --location eastus2

# Create file share
ACI_PERS_RESOURCE_GROUP=lumonitor-eastus2
ACI_PERS_STORAGE_ACCOUNT_NAME=lumonitoreastus2
ACI_PERS_LOCATION=eastus2
ACI_PERS_SHARE_NAME=data

# Create the storage account with the parameters
az storage account create \
    --resource-group $ACI_PERS_RESOURCE_GROUP \
    --name $ACI_PERS_STORAGE_ACCOUNT_NAME \
    --location $ACI_PERS_LOCATION \
    --sku Standard_LRS

STORAGE_KEY=$(az storage account keys list --resource-group $ACI_PERS_RESOURCE_GROUP --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)

# Create the file share
az storage share create \
  --name $ACI_PERS_SHARE_NAME \
  --account-name $ACI_PERS_STORAGE_ACCOUNT_NAME \
  --account-key $STORAGE_KEY
