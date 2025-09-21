import os
import logging
import jittor as jt
'''
Utility functions to save and restore training checkpoints in Jittor.
'''
def restore_checkpoint(ckpt_path, state):
    '''
    Load model, optimizer, EMA, and step from a checkpoint file.
    '''
    # 检查文件是否存在
    if not os.path.exists(ckpt_path):
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_path}. "
                        f"Returned the same state as input")
        return state
    else:
        # 加载 checkpoint
        loaded_state = jt.load(ckpt_path)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_path, state):
    '''
    Save model, optimizer, EMA, and step to a checkpoint file.
    '''
    # 保存状态字典
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    # 确保目录存在
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    jt.save(saved_state, ckpt_path)
