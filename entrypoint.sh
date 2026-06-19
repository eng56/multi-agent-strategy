#!/bin/sh
set -eu
if [ -f /var/secrets/runtime-env ]; then
  set -a
  # Google Secret Manager stores this as a dotenv-formatted secret payload.
  . /var/secrets/runtime-env
  set +a
fi
exec "$@"
