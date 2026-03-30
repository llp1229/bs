# 在PyCharm中运行山西古建筑健康监测系统

## 步骤1：配置Python解释器

1. **打开PyCharm**，选择项目 `ancient_building_system`
2. 点击 `File` → `Settings` → `Project: ancient_building_system` → `Python Interpreter`
3. 点击齿轮图标 → `Add...`
4. 选择 `New environment` → `Virtualenv`
5. 选择合适的Python版本（推荐3.8+）
6. 点击 `OK` 创建虚拟环境

## 步骤2：安装依赖

1. 在PyCharm的终端中（`View` → `Tool Windows` → `Terminal`）
2. 运行命令：
   ```bash
   pip install -r requirements.txt
   ```
3. 等待依赖安装完成

## 步骤3：创建运行配置

1. 点击 `Run` → `Edit Configurations...`
2. 点击 `+` → `Python`
3. 配置参数：
   - **Name**: `Streamlit App`
   - **Script path**: 选择 `main.py` 文件
   - **Parameters**: `run main.py`
   - **Python interpreter**: 选择刚才创建的虚拟环境
   - **Working directory**: 选择项目根目录
4. 点击 `OK` 保存配置

## 步骤4：运行应用

1. 点击 `Run` → `Run 'Streamlit App'`
2. 或者点击工具栏上的绿色运行按钮
3. 应用会在浏览器中打开，默认地址：`http://localhost:8501`

## 步骤5：验证功能

1. **背景图**：应该显示 `D:\sy\背景.jpg`
2. **视频**：应该播放 `D:\sy\古建筑.mp4`
3. **病害检测**：上传图片测试识别功能
4. **环境数据**：查看温湿度趋势图
5. **养护咨询**：生成养护建议

## 常见问题及解决方案

### 问题1：Streamlit未找到
- **解决方案**：在虚拟环境中安装Streamlit
  ```bash
  pip install streamlit
  ```

### 问题2：背景图或视频不显示
- **解决方案**：检查文件路径是否正确
  - 背景图：`D:\sy\背景.jpg`
  - 视频：`D:\sy\古建筑.mp4`

### 问题3：模型加载失败
- **解决方案**：检查模型文件路径
  - 模型路径：`runs/detect/train_combined/weights/best.pt`

### 问题4：依赖安装失败
- **解决方案**：使用国内镜像源
  ```bash
  pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
  ```

## 注意事项

1. 确保PyCharm以管理员权限运行（避免权限问题）
2. 首次运行可能需要较长时间加载依赖
3. 如果遇到内存不足问题，可关闭其他应用程序
4. 视频文件较大，可能需要一些时间加载

---

**提示**：如果仍然遇到问题，请检查项目结构是否完整，确保所有必要的文件都存在。