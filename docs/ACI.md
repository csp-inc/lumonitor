## Container-based Azure Workflow

This document describes best practices for using Azure
Container Instances (ACI) to run project code. It is currently needed to
needed to access HLS data on eastus2 shares, which MS suggests will be
faster than accessing locally. In future, it may be needed for model training
and prediction at large scale.

### Workflow goals

-   Spend as little time as possible creating infrastructure and keep scripts
    out of project code as much as possible
-   Integrate with existing knowledge base and code as much as possible
-   Ensure unused VMs and containers don't continue to run (to save costs)
-   In general, minimize costs as much as possible

### Solution summary

Basically, we will use the Docker-Azure integration in beta versions of docker
to create and manage container instances through ACI. This meets the goals in
several ways:

-   This workflow will largely keep us out of the Azure portal and Azure CLI,
    which are not part of legacy processes and require learning overhead.
-   By creating and setting the context using docker, we can move processes
    from local docker containers to Azure-hosted container with a single
    command line option.
-   Container instances incur costs only while they are running. By nature
    containers only run for the lifetime of the process (if started with the
    `docker run` command and not `docker exec`, of course).
-   By using docker instead of the Azure command line tool, we bypass the Azure
    container registry, which incurs costs for container storage.

### Solution steps

1. Install docker with ACI integration, following instructions
   [here](https://docs.docker.com/engine/context/aci-integration/#install-the-docker-compose-cli-on-linux).
   Login using `docker login azure`. 
2. Create an Azure resource group to store resources if needed.
3. Create an ACI context using docker. 
   `docker context create aci --resource group <resource group> --location <etc> --subscription_id <id> <name>`
   Note that not all options need to be set if environmental variables are
   previously set.
4. If a share is desired, create a storage account (and share?) using [Azure
   CLI](https://docs.microsoft.com/en-us/cli/azure/storage/account?view=azure-cli-latest). 
5. Run the docker image.
   Either the context can be set in the run command, or via `docker context use
   <context_name>`.
   Full command example:
   `docker run -p 80:80 -v lumonitoreastus2/data:/data nginx`


### Issues and questions

- Do images need to be on docker hub to be usable?
- How to monitor CPU / memory usage? In general, how to monitor costs?
- How to set resource availability / requirements? Via `docker run` options?
- How to get code we need to run in container? Clone on create? Copy to share? There's no `-w` argument to the ACI enabled docker client, but maybe we could still mount and somehow ??change dirs?? or ??prefix all directories?? on startup? Use blobfuse? 
- How to monitor command output?
