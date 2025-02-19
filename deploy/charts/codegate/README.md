# Codegate

![Version: 0.0.1](https://img.shields.io/badge/Version-0.0.1-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: v0.1.22](https://img.shields.io/badge/AppVersion-2.112.0-informational?style=flat-square)

CodeGate is a local gateway that makes AI agents and coding assistants safer.

## TL;DR

```console
helm repo add codegate []

helm install codegate/codegate
```

## Usage

The Codegate Chart is available in the following formats:
- [Chart Repository](https://helm.sh/docs/topics/chart_repository/)
- [OCI Artifacts](https://helm.sh/docs/topics/registries/)

### Installing from Chart Repository

The following command can be used to add the chart repository:

```console
helm repo add codegate []
```

Once the chart has been added, install one of the available charts:

```console
helm install codegate/codegate
```

### Installing from an OCI Registry

Charts are also available in OCI format. The list of available charts can be found [here](https://github.com/stacklok/codegate/deploy/charts).
Install one of the available charts:

```shell
helm upgrade -i <release_name> oci://ghcr.io/stacklok/codegate/codegate --version=<version>
```

## Source Code

* <https://github.com/stacklok/codegate>

## Values

<!-- TODO: Auto generate these -->
