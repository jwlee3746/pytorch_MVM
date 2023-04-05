#/bin/bash

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

pid=`ps -ef | grep "/home/dgxa100/anaconda3/bin/python /home/dgxa100/jaewon/MVM/train.py" | grep -v 'grep' | awk '{print $2}'`

if [ -z $pid ]; then
   echo $(date) >> batch_log.txt
   ./run_python_engine.sh >> batch_log.txt 2>&1
   echo "" >> batch_log.txt
fi

