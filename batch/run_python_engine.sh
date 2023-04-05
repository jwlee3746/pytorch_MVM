#!/bin/bash

PYTHON_MAIN_DIR='/home/dgxa100/jaewon'
DATA_DIR="${PYTHON_MAIN_DIR}/Data"
ENGINE_DIR="${PYTHON_MAIN_DIR}/MVM/train.py"

RUN_ARG_TXT='/home/dgxa100/batch/run_arg.txt'
ARG_HIS_LOG='/home/dgxa100/batch/arg_history_log.txt'

: <<'END'
USE_DATA_TXT='/home/dgxa100/batch/data_history.txt'
DATA_LIST_TXT='/home/dgxa100/batch/to_run_data.txt'

to_run_data_arr=(`cat $DATA_LIST_TXT $USE_DATA_TXT $USE_DATA_TXT | sort | uniq -u`)
to_run_data_cnt=${#to_run_data_arr[*]}

if [ $to_run_data_cnt -gt 0 ]; then
	for data in ${to_run_data_arr[*]}
	do
		to_run_data_dir="${DATA_DIR}/${data}"
		if [ -e $to_run_data_dir ]; then
			echo "${to_run_data_dir} 데이터에 대한 MVM+(${ENGINE_DIR})를 실행합니다."
			echo ""
		else
			echo "${to_run_data_dir}가 존재하지 않음. 확인필요."
		fi
	done
fi

END

readarray run_arg_arr < $RUN_ARG_TXT
run_cnt=${#run_arg_arr[*]}

if [ $run_cnt -gt 0 ]; then
	arg=${run_arg_arr[0]}
	echo "${arg} 옵션으로 MVM+(${ENGINE_DIR})를 실행합니다."
	echo -e "----------------------------------------\n${arg}\n$(date) 시작" >> $ARG_HIS_LOG
	/home/dgxa100/anaconda3/bin/python ${ENGINE_DIR} ${arg} &&
		echo -e "$(date) 종료\n----------------------------------------\n" >> $ARG_HIS_LOG &&
		readarray aft_run_arg_arr < $RUN_ARG_TXT &&
		aft_run_cnt=${#aft_run_arg_arr[*]} &&
		cat /dev/null > $RUN_ARG_TXT &&
		if [ $aft_run_cnt -gt 1 ]; then
			for ((idx=1; idx < ${aft_run_cnt} ; idx++));
			do
				echo ${aft_run_arg_arr[idx]} >> $RUN_ARG_TXT
			done
		fi
fi
