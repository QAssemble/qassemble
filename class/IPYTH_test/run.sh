#!/bin/sh
rm ctqmc.*
rm evalsim.*
rm *.json
python3 -u GW_graphene.py $> test.log
