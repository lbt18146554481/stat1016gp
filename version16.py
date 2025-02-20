import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import os
from openai import OpenAI

# 初始化会话状态
if 'current_step' not in st.session_state:
    st.session_state.update({
        'current_step': 1,
        'raw_data': None,
        'processed_data1': None,
        'processed_data2': None,
        'model': None,
        'model_type': None,
        'X_test': None,
        'y_test': None,
        'y_pred': None,
        'metrics': {'MAE': 0.0, 'MSE': 0.0, 'R2': 0.0},
        'trained': False,
        'show_prediction': False,
        'analysis_result': None,
        'y_test' : None,
        'y_pred' : None
    })

# 固定随机种子
SEED = 42
np.random.seed(SEED)

# 导航组件
def show_navigation():
    st.markdown("---")
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col2:
        if st.session_state.current_step > 1:
            if st.button("上一步", key=f"back_{st.session_state.current_step}"):
                st.session_state.current_step -= 1
                st.session_state.trained = False
                st.session_state.show_prediction = False
                st.rerun()
    
    with col3:
        if st.session_state.current_step < 6:
            btn_disabled = False
            if st.session_state.current_step == 1:
                btn_disabled = (st.session_state.processed_data1 is None)
            elif st.session_state.current_step == 2:
                btn_disabled = (st.session_state.processed_data2 is None)
            elif st.session_state.current_step == 3:
                btn_disabled = not st.session_state.trained
            elif st.session_state.current_step == 4:
                btn_disabled = not st.session_state.show_prediction
            
            if st.button("下一步", 
                        key=f"next_{st.session_state.current_step}",
                        disabled=btn_disabled):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button("从头开始", key=f"back_{st.session_state.current_step + 1}"):
                st.session_state.current_step = 1
                st.rerun()
           
# 页面配置
if st.session_state.current_step == 1:
    st.markdown('<h1 style="font-size:50px; color:white;">人口增长率预测</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size:20px; color:white;">组员：高路绍徐蔡</h3>', unsafe_allow_html=True)

#---------------------------------- 步骤1：数据上传与缺失处理----------------------------------
#------------------------------------------------------------------------------------------

if st.session_state.current_step == 1:
    st.header("第一步：数据上传与缺失处理")
    
    uploaded_file = st.file_uploader("上传数据集（CSV/Excel）", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            st.session_state.raw_data = raw_df
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")
            st.stop()
        
        st.subheader("缺失值填充配置")
        col1, col2 = st.columns([3, 2])
        with col1:
            fill_method = st.selectbox(
                "选择填充方法",
                options=["均值填充（数值列）", "中位数填充（数值列）", 
                        "众数填充（分类列）", "固定值填充","删除包含缺失值的行"],
                index=0
            )
        with col2:
            if fill_method == "固定值填充":
                fill_value = st.number_input("输入填充值")
            else:
                fill_value = None
        
        if st.button("执行缺失值填充"):
            with st.spinner("正在处理缺失值..."):
                processed_df = st.session_state.raw_data.copy()
                if fill_method == "删除包含缺失值的行":
                    original_count = len(processed_df)
                    processed_df = processed_df.dropna()
                    new_count = len(processed_df)
                    drop_count = original_count - new_count
                else:
                    for col in processed_df.columns:
                        if processed_df[col].dtype in ['int64', 'float64']:
                            if fill_method == "均值填充（数值列）":
                                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                            elif fill_method == "中位数填充（数值列）":
                                processed_df[col].fillna(processed_df[col].median(), inplace=True)
                            elif fill_method == "固定值填充":
                                processed_df[col].fillna(fill_value, inplace=True)
                        else:
                            if fill_method == "众数填充（分类列）":
                                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                            elif fill_method == "固定值填充":
                                processed_df[col].fillna(fill_value, inplace=True)
                    st.success("缺失值填充完成！")
                st.session_state.processed_data1 = processed_df

                if fill_method == "删除包含缺失值的行":
                    st.success(f"已删除包含缺失值的{drop_count}行，剩余{new_count}行数据！")
        
        if st.session_state.processed_data1 is not None:
            st.subheader("处理效果")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("填充前缺失值总数", st.session_state.raw_data.isnull().sum().sum())
            with col2:
                st.metric("填充后缺失值总数", st.session_state.processed_data1.isnull().sum().sum())
            st.dataframe(st.session_state.processed_data1.head(3), height=150)
    
    show_navigation()

#-------------------------------------步骤二 数据编码和标准化---------------------------------
#------------------------------------------------------------------------------------------

elif st.session_state.current_step == 2:
    st.header("第二步：数据编码与标准化")
    
    df = st.session_state.processed_data1
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    st.subheader("分类特征编码")
    encoding_method = st.radio("编码方式", ["标签编码（推荐）", "独热编码"], horizontal=True)
    
    st.subheader("数值特征缩放") 
    scaling_method = st.radio("缩放方法", ["不缩放", "标准化", "归一化"], horizontal=True)
    
    if st.button("应用转换"):
        with st.spinner("处理中..."):
            processed_df = df.copy()
            if categorical_cols:
                if encoding_method == "独热编码":
                    encoder = OneHotEncoder(sparse=False)
                    encoded = encoder.fit_transform(processed_df[categorical_cols])
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
                    processed_df = pd.concat([processed_df.drop(categorical_cols, axis=1), encoded_df], axis=1)
                else:
                    for col in categorical_cols:
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col])
            if scaling_method != "不缩放" and numeric_cols:
                scaler = StandardScaler() if scaling_method == "标准化" else MinMaxScaler()
                processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
            st.session_state.processed_data2 = processed_df
            st.success("数据处理完成！")
            st.rerun()
    
    if st.session_state.processed_data2 is not None:
        st.dataframe(st.session_state.processed_data2.head(3), height=150)
    
    show_navigation()

#--------------------------------------第三步：特征选择与模型训练---------------------------------
#--------------------------------------------------------------------------------------------

elif st.session_state.current_step == 3:
    st.header("第三步：特征选择与模型训练")
    
    df = st.session_state.processed_data2
    if df is None:
        st.error("数据未加载，请返回上一步")
        st.stop()
    
    # 特征选择列
    col1, col2 = st.columns([2, 3])
    with col1:
        target = st.selectbox(
            "选择目标列",
            options=df.columns,
            index=df.columns.get_loc('人口增长率') if '人口增长率' in df.columns else len(df.columns)-1
        )
        
        available_features = [col for col in df.columns if col != target]
        default_features = [col for col in available_features if col not in ['国家', '年份', '地区']]
        
        features = st.multiselect(
            "选择特征列", 
            options=available_features,
            default=default_features
        )
        st.session_state.features = features
        st.session_state.target = target

    # 模型训练列
    with col2:
        st.subheader("模型训练配置")
        model_type = st.selectbox(
            "选择模型",
            options=["线性回归", "决策树回归", "随机森林回归", "KNN回归", "SVM回归"],
            index=0
        )
        train_size = st.slider("训练集比例", 0.5, 0.9, 0.8, step=0.05)
        
        if st.button("开始训练", use_container_width=True):
            with st.spinner("训练中..."):
                try:
                    X = df[st.session_state.features]
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        train_size=train_size,
                        random_state=SEED
                    )
                    st.session_state.y_test = y_test
                    model_map = {
                        "线性回归": LinearRegression(),
                        "决策树回归": DecisionTreeRegressor(random_state=SEED),
                        "随机森林回归": RandomForestRegressor(random_state=SEED),
                        "KNN回归": KNeighborsRegressor(),
                        "SVM回归": SVR()
                    }
                    model = model_map[model_type]
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    st.session_state.update({
                        'model': model,
                        'model_type': model_type,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'metrics': {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'MSE': mean_squared_error(y_test, y_pred),
                            'R2': r2_score(y_test, y_pred)
                        },
                        'trained': True
                    })
                    if y_pred:
                       st.success("模型训练成功！")
                       st.session_state.y_pred = y_pred
                    else:
                       st.warning('模型训练失败') 
                except Exception as e:
                    st.error(f"训练失败：{str(e)}")
    show_navigation()


#----------------------------------------第四步：可视化分析------------------------------------
#-------------------------------------------------------------------------------------------

elif st.session_state.current_step == 4:
    st.header("第四步：可视化分析")
    
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第三步")
        st.stop()
    
    #模型性能
    st.subheader("模型性能")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{st.session_state.metrics['MAE']:.2f}")
    col2.metric("MSE", f"{st.session_state.metrics['MSE']:.2f}")
    col3.metric("R²", f"{st.session_state.metrics['R2']:.2f}")
    st.write("")

    #预测数据简要
    st.subheader("预测值 vs 真实值对比简要") 
    y_pred_sample = st.session_state.y_pred[:5]
    y_test_sample = st.session_state.y_test[:5]
    # 创建对比表格
    comparison_df = pd.DataFrame({
    "预测值": y_pred_sample,
    "真实值": y_test_sample
    })
    st.dataframe(
    comparison_df.style.format("{:.2f}"),  # 保留两位小数
     height=200
    )
    st.write("")
    
    #画图
    plt.style.use('seaborn')
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        model_type = st.session_state.model_type
        
        if model_type == "线性回归":
            if len(st.session_state.features) == 1:
                st.subheader("回归图")
                ax1.scatter(st.session_state.X_test, st.session_state.y_test, color='#1f77b4', label='实际值')
                ax1.plot(st.session_state.X_test, st.session_state.y_pred, color='#ff7f0e', linewidth=2, label='预测值')
                ax1.set_xlabel(st.session_state.features[0])
                ax1.set_ylabel(st.session_state.target)
                ax1.set_title("线性回归拟合曲线")
                st.pyplot(fig1)
                st.markdown("""
                **图表说明**  
                - X轴：选择的特征值  
                - Y轴：目标值的实际值与预测值  
                - 红线：线性回归模型的预测拟合线  
                """)
                st.write("")
            else:
                pca = PCA(n_components=1)
                X_pca = pca.fit_transform(st.session_state.X_test)
                st.subheader("散点图")
                ax1.scatter(X_pca, st.session_state.y_test, color='#1f77b4', label='实际值')
                ax1.scatter(X_pca, st.session_state.y_pred, color='#ff7f0e', label='预测值')
                ax1.set_title("散点图")
                ax1.set_xlabel("主成分1 (PCA降维)")
                ax1.set_ylabel(st.session_state.target)
                ax1.set_title(f"{model_type}预测结果", pad=20)
                ax1.legend()
                st.pyplot(fig1)
                st.markdown("""
                **图表说明**  
                - X轴：使用PCA降维后的主成分  
                - Y轴：目标值的实际值与预测值  
                - 蓝点代表实际
                - 黄点代表预测
                """)
                st.write("")
        
        elif model_type in ["决策树回归", "随机森林回归"]:
            importance = pd.DataFrame({
                '特征': st.session_state.features,
                '重要性': st.session_state.model.feature_importances_
            }).sort_values('重要性', ascending=True)
            ax1.barh(importance['特征'], importance['重要性'], color='#2ca02c')
            ax1.set_title("特征重要性分析", pad=20)
            ax1.set_xlabel("重要性得分", labelpad=10)
        
        elif model_type == "KNN回归":
            ax1.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.6, color='#9467bd')
            ax1.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], 
                    [st.session_state.y_test.min(), st.session_state.y_test.max()], 'k--')
            ax1.set_xlabel("实际值", labelpad=10)
            ax1.set_ylabel("预测值", labelpad=10)
            ax1.set_title("实际值 vs 预测值", pad=20)
        
        elif model_type == "SVM回归":
            ax1.scatter(st.session_state.X_test.iloc[:, 0], st.session_state.y_test, color='#17becf', label='实际值')
            ax1.scatter(st.session_state.X_test.iloc[:, 0], st.session_state.y_pred, color='#e377c2', label='预测值')
            ax1.set_xlabel(st.session_state.features[0], labelpad=10)
            ax1.set_ylabel(st.session_state.target, labelpad=10)
            ax1.set_title("SVM回归预测分布", pad=20)
            ax1.legend()
        
        st.subheader("残差图")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.residplot(
            x=st.session_state.y_pred, 
            y=st.session_state.y_test - st.session_state.y_pred,
            lowess=True,
            line_kws={'color': '#d62728', 'lw': 1.5},
            scatter_kws={'alpha': 0.4, 'color': '#2ca02c'}
        )
        ax2.axhline(0, color='#7f7f7f', linestyle='--', linewidth=1)
        ax2.set_title("残差分布图", pad=15)
        ax2.set_xlabel("预测值", labelpad=10)
        ax2.set_ylabel("残差 (实际值 - 预测值)", labelpad=10)
        st.pyplot(fig2)
        st.markdown("""
        **图表说明**  
        - X轴：模型预测值  
        - Y轴：预测残差（实际值 - 预测值）  
        - 红线：残差的趋势线  
        - 灰色虚线： 残差为 0 的基准线，用于对比误差的正负分布
        """)
        
    except Exception as e:
        st.error(f"可视化错误：{str(e)}")
    st.session_state.show_prediction = True
    
    show_navigation()

#----------------------------------------第五步：预测新数据------------------------------------
#-------------------------------------------------------------------------------------------

elif st.session_state.current_step == 5:
    st.header("第五步：新数据预测")
    
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第三步")
        st.stop()
    
    # 创建输入表单
    with st.form("prediction_form"):
        st.subheader("输入预测特征值")
        input_data = {}
        selected_features = st.session_state.features
        
        # 动态生成输入框
        cols = st.columns(2)
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                input_data[feature] = st.number_input(
                    f"{feature} 值",
                    key=f"input_{feature}"
                )
        left, right = st.columns(2)
        if '生育政策' in st.session_state.features:
           with right:
              st.write('有生育政策输入1，无则输入0')
        # 提交按钮
        submitted = st.form_submit_button("开始预测")
        
        if submitted:
            try:
                # 转换输入数据
                input_df = pd.DataFrame([input_data])
                
                # 进行预测
                y_pred = st.session_state.model.predict(input_df)
                
                # 显示结果
                st.success("预测完成！")
                st.metric(
                    label=f"预测 {st.session_state.target}",
                    value=f"{y_pred[0]:.2f}"
                )
                st.session_state.show_prediction = True
            except Exception as e:
                st.error(f"预测失败：{str(e)}")
    show_navigation()

#----------------------------------------第六步：数据智能分析------------------------------------
#--------------------------------------------------------------------------------------------

elif st.session_state.current_step == 6:
    st.header("第六步：数据智能分析")
    st.subheader('调用模型为豆包1.5')
    # 检查必要状态
    if not st.session_state.get('model'):
        st.error("模型未训练，请返回第三步")
        st.stop()
    if st.button("生成智能分析报告"):
       with st.spinner("AI正在生成分析报告..."):
        try:
                 # 构建数据摘要
                data_summary = {
                    "数据概况": {
                    "样本数量": len(st.session_state.processed_data),
                    "特征数量": len(st.session_state.features),
                    "特征变量": st.session_state.features,
                    "目标变量": st.session_state.target
                    },
                    "模型信息": {
                    "模型类型": st.session_state.model_type,
                    "MAE": round(st.session_state.metrics['MAE'], 3),
                    "R2": round(st.session_state.metrics['R2'], 3)
                    },
                   "特征重要性": (
                    dict(zip(
                    st.session_state.features,
                    np.squeeze(
                   # 树模型使用 feature_importances_，线性模型使用 coef_
                   st.session_state.model.feature_importances_ if hasattr(st.session_state.model, 'feature_importances_') 
                   else st.session_state.model.coef_
                   ).round(3)
                   )) 
                   if (hasattr(st.session_state.model, 'feature_importances_') or hasattr(st.session_state.model, 'coef_'))
                   and len(st.session_state.features) == len(np.squeeze(
                   st.session_state.model.feature_importances_ if hasattr(st.session_state.model, 'feature_importances_') 
                   else st.session_state.model.coef_
                   ))
                   else "无法计算特征重要性"
                   )
                   }
    
                   #构建提示词
                message = f"""
                   你是一个专业的数据科学家，你收集了可能影响一个国家人口增长率的数据，并用这些数据训练了模型，然后给模型输入新的数据来预测人口增长率
                   请根据以下分析请求和提供的数据摘要，用中文给出专业分析报告,字数大概300多字：

                   分析要求：
                   1. 解读数据特征与目标变量({st.session_state.target})的关系
                   2. 评估当前模型性能并提出改进建议
                   3. 分析可能影响预测结果准确性的潜在因素
                   4. 给出可操作的政策建议（如存在政策相关特征）
    
                   数据摘要：
                   {data_summary}
                   """
                client = OpenAI(
                   api_key = 'de226819-d548-4b72-aa45-470adb3bd551',
                   base_url = "https://ark.cn-beijing.volces.com/api/v3",
                   )

                   #调用火山引擎-Doubao-1.5...ion-pro-32k
                response = client.chat.completions.create(
                   model="ep-20250123135734-mwd8w",
                   messages=[
                   {
                   "role": "user",
                   "content": [
                   {"type": "text", "text": message},
                   {
                     "type": "image_url",
                     "image_url": {
                         "url": "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg"
                     }
                 },
             ],
         }
     ],
 )
                Response = response.choices[0].message.content  
                st.session_state.analysis_result = Response
        except Exception as e:
                st.error(f"分析失败：{str(e)}")
                st.session_state.analysis_result = None

    if st.session_state.get('analysis_result'):
        st.subheader("AI分析报告")
        # 直接显示已清洗的内容
        st.markdown(f"```markdown\n{st.session_state.analysis_result}\n```") 
    show_navigation()