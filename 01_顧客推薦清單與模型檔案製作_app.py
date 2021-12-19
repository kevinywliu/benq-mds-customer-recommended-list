# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 1.主題二:購物籃多產品推薦分析
# %% [markdown]
# ## 1.1資料前處理

# %%
# pip install apyori

import streamlit as st 
import streamlit.components.v1 as stc 
from run_data_process_analysis2 import run_data_process_analysis,prediction
# from run_product_contribution import run_product_contribution
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">01_顧客推薦清單與模型檔案製作_app</h1>
		</div>
		"""

def main():
	# st.title("ML Web App with Streamlit")
    stc.html(html_temp)
    st.sidebar.image('docs/logo.png', width=250)
    st.sidebar.write('') # Line break
    st.sidebar.header('Navigation')

    menu = ['模型訓練',"顧客推薦清單預測"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == '模型訓練':
        run_data_process_analysis()
        
    elif choice =='顧客推薦清單預測':
        prediction()
    

if __name__ == '__main__':
	main()
