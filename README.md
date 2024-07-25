# DDP_IPC

## Description

이 레포는 분산 병렬 학습 시의 사용되는 데이터의 전달과정인 Collective 패턴을 구현하는 레포입니다.

분산 병렬 학습 과정에서 필수적으로 사용되는 IPC도 함꼐 구현되어 있습니다. 

Shared Memory나 Shared File System 방식이 아닌 소켓 통신으로 구현되어 있습니다.

필요에 따라 BroadCast나 Sactter가 구현되어 있지만, 주 된 구현의 목적은 AllReduce와 Ring_Allreduce 입니다.

## IPC


## All Reduce


## Ring Allreduce

