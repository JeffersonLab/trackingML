#!/bin/sh
#*********************************************************
#---------------------------------------------------------
# JHEP math CLI interface.
#---------------------------------------------------------
SCRIPT_DIR=`dirname $0`
#---------------------------------------------------------
# The MALLOC_ARENA_MAX is GLIB flag that controls
# how much VIRTUAL memory will be claimed by JVM
#---------------------------------------------------------
MALLOC_ARENA_MAX=1; export MALLOC_ARENA_MAX
#---------------------------------------------------------
# SET UP JAVA_OPTIONS With the max memory and starting
# memory
#---------------------------------------------------------
JAVA_OPTIONS="-Xmx12024m -Xms12024m"
java $JAVA_OPTIONS -cp "target/tracking-1.1-SNAPSHOT-jar-with-dependencies.jar" org.jlab.ml.tracking.chambers.RunTraining $*
