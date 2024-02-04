
## Standard
- 提交规范
  - clone 到本地的时候，无比再GitHub repo 先创建`dev-<username>`分支，然后再clone到本地,本地签出`dev-<username>`分支,本地开发完成后，提交到`dev-<username>`分支，然后再提交PR到`main`分支
  
- 测试规范
  - 使用`pytest`进行测试，测试文件放在`tests`目录下，测试文件名以`test_`开头，测试函数名以`test_`开头, 测试类名以`Test`开头
  - 运行[poetry_scripts.py](poetry_scripts.py)文件来执行测试
**注意：[.pre-commit-config.yaml](.pre-commit-config.yaml)会有其他要求,它会被`poetry_scripts.py`自动触发，如果格式或者类型检查不符合规范会自动格式化文件并且给出警告然后显示执行失败，重新执行即可，若仍显示失败，检查警告信息，大概率是不符合`mypy`的要求，出现了潜在的`bug`或者没有`类型注释`**





## Roadmap









## Latest Changes
