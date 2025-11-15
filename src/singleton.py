"""
修复版本的单例装饰器

修复了原实现中的关键问题，同时保持向后兼容性
"""

import threading
import weakref
import logging
import atexit
from functools import wraps
from typing import Dict, Any, Optional

# 尝试导入项目日志记录器，如果失败则使用标准日志
try:
    from util.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)
    # 配置基本日志记录
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 全局单例存储，用于调试和监控
_GLOBAL_SINGLETONS = {}
_GLOBAL_SINGLETONS_LOCK = threading.Lock()


def singleton(cls):
    """
    修复版本的单例装饰器

    主要修复：
    1. 修复弱引用解引用问题
    2. 改进清理机制
    3. 添加调试支持
    4. 增强错误处理
    """
    _instances = {}
    _locks = {}
    _cleanup_registered = False

    @wraps(cls)
    def wrapper(*args, **kwargs):
        nonlocal _cleanup_registered

        # 每个类使用独立锁
        if cls not in _locks:
            _locks[cls] = threading.RLock()

        # 获取或创建实例
        with _locks[cls]:
            # 检查是否已有实例
            weak_ref = _instances.get(cls)

            if weak_ref is not None:
                # 尝试获取强引用
                strong_ref = weak_ref()
                if strong_ref is not None:
                    return strong_ref
                else:
                    # 弱引用已失效，清理
                    logger.debug(f"Cleaning up stale weak reference for {cls.__name__}")
                    del _instances[cls]

            # 创建新实例
            logger.debug(f"Creating new instance of {cls.__name__}")
            try:
                new_instance = cls(*args, **kwargs)

                # 创建弱引用并注册清理回调
                def cleanup_callback(weak_ref_obj):
                    with _locks.get(cls, threading.RLock()):
                        if _instances.get(cls) is weak_ref_obj:
                            logger.debug(f"Cleaning up instance of {cls.__name__}")
                            del _instances[cls]

                    # 从全局存储中也清理
                    with _GLOBAL_SINGLETONS_LOCK:
                        global_key = cls.__name__
                        if global_key in _GLOBAL_SINGLETONS:
                            del _GLOBAL_SINGLETONS[global_key]

                # 更新全局单例存储
                with _GLOBAL_SINGLETONS_LOCK:
                    _GLOBAL_SINGLETONS[cls.__name__] = {
                        'instance': new_instance,
                        'class_name': cls.__name__,
                        'created_at': __import__('time').time(),
                        'module': cls.__module__
                    }

                _instances[cls] = weakref.ref(new_instance, cleanup_callback)
                instance = new_instance

                # 注册清理函数（如果尚未注册）
                if not _cleanup_registered:
                    _cleanup_registered = True
                    _register_global_cleanup()

            except Exception as e:
                logger.error(f"Failed to create instance of {cls.__name__}: {e}")
                raise

        return instance

    # 添加调试方法到装饰器
    def get_debug_info() -> Dict[str, Any]:
        """获取调试信息"""
        with _locks.get(cls, threading.RLock()):
            weak_ref = _instances.get(cls)
            return {
                'class_name': cls.__name__,
                'has_instance': weak_ref is not None,
                'is_weak_ref': True,
                'ref_alive': weak_ref() is not None if weak_ref else False,
                'instance_id': id(weak_ref()) if weak_ref and weak_ref() else None
            }

    def get_instance(*args, **kwargs):
        """获取实例的便捷方法"""
        return wrapper(*args, **kwargs)

    def is_instance_created() -> bool:
        """检查实例是否已创建"""
        with _locks.get(cls, threading.RLock()):
            weak_ref = _instances.get(cls)
            return weak_ref is not None and weak_ref() is not None

    def reset_instance() -> None:
        """重置实例（主要用于测试）"""
        with _locks.get(cls, threading.RLock()):
            weak_ref = _instances.get(cls)
            if weak_ref:
                strong_ref = weak_ref()
                if strong_ref and hasattr(strong_ref, '__del__'):
                    # 调用清理方法
                    try:
                        strong_ref.__del__()
                    except:
                        pass
            _instances[cls] = None
            if cls in _instances:
                del _instances[cls]

    # 将方法添加到装饰器
    wrapper.get_debug_info = get_debug_info
    wrapper.get_instance = get_instance
    wrapper.is_instance_created = is_instance_created
    wrapper.reset_instance = reset_instance

    return wrapper


def _register_global_cleanup():
    """注册全局清理函数"""
    def global_cleanup():
        logger.info("Singleton cleanup: starting...")
        with _GLOBAL_SINGLETONS_LOCK:
            for cls_name, instance_info in list(_GLOBAL_SINGLETONS.items()):
                try:
                    instance = instance_info.get('instance')
                    if instance:
                        logger.debug(f"Cleaning up {cls_name}")
                        # 调用实例的清理方法（如果存在）
                        if hasattr(instance, '__del__'):
                            try:
                                instance.__del__()
                            except:
                                pass
                except Exception as e:
                    logger.error(f"Error cleaning up {cls_name}: {e}")
            _GLOBAL_SINGLETONS.clear()
        logger.info("Singleton cleanup: completed")

    atexit.register(global_cleanup)


def get_all_singletons_info() -> Dict[str, Any]:
    """获取所有单例的调试信息"""
    info = {}
    with _GLOBAL_SINGLETONS_LOCK:
        for cls_name, instance_info in _GLOBAL_SINGLETONS.items():
            try:
                instance = instance_info.get('instance')
                info[cls_name] = {
                    'has_instance': instance is not None,
                    'instance_id': id(instance) if instance else None,
                    'is_alive': instance is not None,
                    'class_name': instance_info.get('class_name'),
                    'created_at': instance_info.get('created_at')
                }
            except Exception as e:
                info[cls_name] = {'error': str(e)}
    return info


# 向后兼容的全局变量访问
def get_singleton_debug_info(cls) -> Dict[str, Any]:
    """获取指定类的单例调试信息（向后兼容）"""
    if hasattr(cls, 'get_debug_info'):
        return cls.get_debug_info()
    return {'error': 'Class does not have debug info method'}


# 添加日志配置
def configure_singleton_logging(level: int = logging.WARNING):
    """配置单例日志记录"""
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# 示例用法和测试
if __name__ == "__main__":
    # 配置日志

    configure_singleton_logging(logging.DEBUG)

    @singleton
    class TestService:
        def __init__(self, name="default"):
            self.name = name
            self.created_at = __import__('time').time()

        def get_info(self):
            return {
                'name': self.name,
                'created_at': self.created_at,
                'id': id(self)
            }

    print("测试修复版本的单例装饰器")

    # 测试基本功能
    service1 = TestService("test1")
    service2 = TestService("test2")  # 参数被忽略，返回同一实例

    print(f"Service1: {service1.get_info()}")
    print(f"Service2: {service2.get_info()}")
    print(f"是同一个实例: {service1 is service2}")
    print(f"调试信息: {TestService.get_debug_info()}")

    # 测试实例状态
    print(f"实例已创建: {TestService.is_instance_created()}")

    # 测试所有单例信息
    print(f"所有单例信息: {get_all_singletons_info()}")