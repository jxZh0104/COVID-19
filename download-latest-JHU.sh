#!/usr/bin/env bash
# source: https://github.com/mongodb-developer/open-data-covid-19/blob/master/data-import/0-download-latest-JHU.sh
rm -rf jhu
git clone --depth=1 https://github.com/CSSEGISandData/COVID-19.git jhu
rm -rf jhu/.git