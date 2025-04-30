## 指南
### 1.请在`python3.9`环境中使用项目的`requirements.txt`进行环境配置：
`pip install -r requirements.txt`

### 2.本项目涉及深度学习相关内容，要体验完整页面功能请安装`gpu-pytorch == 2.5.0`（可以直接去官网安装）：
`pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121`

示例CUDA=12.1，请根据自己设备的CUDA版本安装。

### 3.通过如下命令行运行脚本进行实验：
`streamlit run 🏠主页 `

此时会将🏠主页作为主页面打开系统，剩余的分页面来自pages文件下的页面。

### 4.注意事项：
异常检测页面的异常检测功能是通过MyModel模型实现的，因此在对应的py文件中包含了MyModel模型的代码。你可以将这部分内容替换为你想要展示的深度学习模型，并对streamlit实现部分进行相应修改。关于MyModel模型的详细内容请参见
`https://github.com/2022ljz/Anomaly-detection-model-based-on-frequency-domain-localization`

## 页面功能

---

### 🏠 **主页**
  ◦ 研究背景介绍              ◦ 页面导航  

---

### 📊 **模型介绍页**  
  ◦ 模型研究背景介绍            ◦ 模型架构展示              ◦ 模型模块介绍  
  ◦ 多模型异常检测性能对比  

---

### 📈 **数据集可视化页**  
  ◦ 支持用户配置参数            ◦ 自动数据信息统计            ◦ 滑动窗口展示  
  ◦ 数据下载  

---

### 🔍 **异常检测页**  
  ◦ 支持用户配置参数            ◦ 异常检测指标展示            ◦ 滑动窗口展示  
  ◦ 支持仅展示单一类型数据         ◦ 数据下载  

