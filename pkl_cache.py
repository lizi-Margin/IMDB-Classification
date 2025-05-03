import pickle
import os
from functools import wraps
import hashlib

def pkl_cache(filepath):
    """
    缓存函数结果到pickle文件的装饰器
    
    参数:
        filepath: 缓存文件路径，可以包含格式化字符串（如{param}）
    
    返回:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成唯一的缓存文件名
            final_path = filepath.format(**kwargs) if '{' in filepath else filepath
            
            # 如果文件存在，则尝试加载缓存
            if os.path.exists(final_path):
                try:
                    with open(final_path, 'rb') as f:
                        print(f"从缓存加载: {final_path}")
                        return pickle.load(f)
                except (pickle.PickleError, EOFError) as e:
                    print(f"缓存加载失败，重新计算: {e}")
            
            # 计算函数结果
            result = func(*args, **kwargs)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(final_path) or '.', exist_ok=True)
            
            # 保存结果到缓存
            try:
                with open(final_path, 'wb') as f:
                    pickle.dump(result, f)
                print(f"结果已缓存到: {final_path}")
            except Exception as e:
                print(f"缓存保存失败: {e}")
            
            return result
        return wrapper
    return decorator