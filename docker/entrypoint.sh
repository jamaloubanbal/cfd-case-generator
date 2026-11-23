#!/bin/bash
# Sources OpenFOAM, then starts the web app ("web", the default) or runs any other command
# in the OpenFOAM environment, e.g.:
#   docker run --rm cfd-case-generator python -m app.cli check examples/dam_break.json
source /usr/share/openfoam/etc/bashrc >/dev/null 2>&1 || true

# Hugging Face Spaces shows the app in an iframe, where Streamlit's XSRF check breaks uploads
XSRF=true
[ -n "${SPACE_ID:-}" ] && XSRF=false

if [ "${1:-web}" = "web" ]; then
    exec streamlit run app/main.py \
        --server.address=0.0.0.0 --server.port="${PORT:-8501}" --server.headless=true \
        --server.enableXsrfProtection=$XSRF --browser.gatherUsageStats=false
fi
exec "$@"
