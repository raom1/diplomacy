from time import time
import logging

logging.basicConfig(filename="logs/dip_logs.log", filemode="w", level=logging.INFO, format=" %(name)s :: %(levelname)-s :: %(message)s")
logger = logging.getLogger("app")



def calc_time_diff(begin, stage_name):
    # try:
    #     step_timing_dict[stage_name].append(time()-begin)
    # except:
    #     step_timing_dict[stage_name] = [time()-begin]
    # print(f"step: {stage_name}\nduration: {time()-begin}")
    logger.info(f"step: {stage_name}\tduration: {time()-begin}")