# 项目结构测试指南

## 快速测试

### 1️⃣ 运行结构验证测试

```bash
cd c:\Users\qq152\Desktop\visualmodel
python test_structure.py
```

这个脚本会验证：
- ✅ 所有关键模块的导入是否正常
- ✅ 配置加载是否成功
- ✅ 模型注册表功能
- ✅ 响应工具是否可用
- ✅ 数据模型 (Pydantic Schemas)
- ✅ 文件结构完整性
- ✅ Celery 配置
- ✅ FastAPI 应用启动


### 2️⃣ 逐项手动测试

#### 测试导入
```python
python -c "from app.config.env import AppConfig; print(f'App: {AppConfig.app_name}')"
python -c "from app.models.registry import ModelRegistry; print('Registry OK')"
python -c "from app.core.celery_app import celery_app; print('Celery OK')"
```

#### 测试配置
```python
python -c "
from app.config.env import AppConfig, ModelConfig
print(f'App Name: {AppConfig.app_name}')
print(f'Model Device: {ModelConfig.model_device}')
print(f'Batch Size: {ModelConfig.batch_size}')
"
```

#### 测试模型注册表
```python
python -c "
from app.models.registry import ModelRegistry
ModelRegistry.register('test', {'data': 123})
result = ModelRegistry.get('test')
print(f'Register Test: {result}')
"
```

#### 测试响应生成
```python
python -c "
from app.utils.response_util import ResponseUtil
resp = ResponseUtil.success(msg='测试', data={'id': 1})
print('Response generated successfully')
"
```


## 深度测试场景

### 场景 1: 测试模型工厂（仅检查导入，不加载权重）

```python
# test_factory.py
from app.models.model_factory import ModelFactory
from app.models.registry import ModelRegistry

print("✅ ModelFactory 导入成功")
print("✅ 可以调用 ModelFactory.create_model()")
```

运行: `python test_factory.py`


### 场景 2: 测试 API 路由注册

```python
# test_routes.py
from server import app

print("所有 API 路由:")
for route in app.routes:
    print(f"  - {route.path} ({route.methods if hasattr(route, 'methods') else 'N/A'})")
```

运行: `python test_routes.py`


### 场景 3: 测试 Celery 任务注册

```python
# test_celery_tasks.py
from app.core.celery_app import celery_app

print("已注册的 Celery 任务:")
for task_name in sorted(celery_app.tasks.keys()):
    if 'emotion' in task_name.lower() or 'feat' in task_name.lower():
        print(f"  - {task_name}")
```

运行: `python test_celery_tasks.py`


### 场景 4: 完整的应用启动测试（需要依赖）

```bash
# 需要先安装依赖
# pip install fastapi uvicorn pydantic redis celery

# 启动 FastAPI 服务器
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

然后访问: http://localhost:8000/docs


## 常见问题排查

### 问题: ImportError - app.config 找不到

**解决:**
1. 检查是否在项目根目录中运行
2. 确保 `app/__init__.py` 存在
3. 检查 Python 路径: `python -c "import sys; print(sys.path)"`

### 问题: Redis 连接失败

**解决:**
1. 确保 Redis 已安装并运行: `redis-cli ping`
2. 检查 Redis 地址配置: `app/config/env.py`

### 问题: 模型权重文件缺失

**解决:**
- 这是预期的（如果你没有下载权重）
- 阶段 2 的完全测试需要实际的模型文件

### 问题: ModuleNotFoundError - xxx

**解决:**
1. 检查依赖是否安装: `pip list`
2. 必要的依赖:
   - fastapi >= 0.70.0
   - pydantic >= 2.0
   - pandas
   - numpy
   - torch (如果使用 GPU)


## 测试检查清单

- [ ] test_structure.py 所有测试都通过
- [ ] 能够 import 所有 app.* 模块
- [ ] Redis 配置正确
- [ ] FastAPI 应用能够正常导入
- [ ] 数据模型能正常创建实例
- [ ] 没有循环导入错误


## 预期输出示例

```
╔══════════════════════════════════════════════════════════════╗
║                    项目结构验证测试                           ║
╚══════════════════════════════════════════════════════════════╝

============================================================
测试 1: 导入测试
============================================================
✅ app.config.env
✅ app.config.constant
✅ app.models.base
✅ app.models.registry
✅ app.utils.response_util
... (更多输出)

============================================================
测试总结
============================================================
✅ 通过: 导入测试
✅ 通过: 配置加载
✅ 通过: 模型注册表
... (更多结果)

============================================================
最终结果: 8/8 个测试通过
============================================================

🎉 所有测试都通过了！项目结构正常。
```

## 下一步

- 如果所有测试都通过，可以开始开发
- 检查 `server.py` 中的导入是否正确
- 验证数据库/Redis 连接
- 测试异步任务和 WebSocket
