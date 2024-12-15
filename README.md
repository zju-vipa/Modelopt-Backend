# ModelOpt的后端代码

ModelOpt包含[前端](https://github.com/zju-vipa/modelopt-frontend.git)和[后端](https://github.com/zju-vipa/modelopt-backend.git)，目录结构为：
```
├── ModelOPT
│   ├── modelopt-backend
│   ├── modelopt-frontend
```

## Build Setup


- 安装依赖库：
    - pip install -r requirements.txt

- 建立数据库：
    - 运行modelopt-backend/sql/下的sql命令来构建表
    - 配置 modelopt-backend/config
- 运行程序
    - cd modelopt-backend/
    - python app.py
    - 在浏览器输入提示的网址即可进入系统