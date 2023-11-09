#!/bin/bash

pkill -9 -f client.py
pkill -9 -f server.py

if test -d results; then
	rm -rf results/
fi

if test -d models; then
	rm -rf models/
fi

