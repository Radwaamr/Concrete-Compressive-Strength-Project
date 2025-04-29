# **Concrete-Compressive-Strength-Project**
In Preif We did: Some analysis and story telling about Concrete Dataset from Kaggle. And doing ML modeling by both Regression and Classification.

🔗 **Live Demo:** [View the Web App](https://concrete-compressive-strength-project-h6rrsmvr55num8zo6ay8us.streamlit.app/)

More Details:-
# **1. Problem Definition:** Any developed country has paved roads.
   Also has a urban construction and infrastructure at the highest level.
   But.. A road with cracks !!, A bridge has been closed !!
                             
# **2. Dataset Description:** Concrete is the most important material in civil engineering.
   The concrete compressive strength is a highly nonlinear function of age and ingredients. 
   These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, 
   coarse aggregate, and fine aggregate.
                              
# **Data Characteristics:** The actual concrete compressive strength (MPa) 
   for a given mixture under a specific age (days) was determined from laboratory. Data is in raw form (not scaled).
                              
**Summary Statistics:**
   * Number of instances (observations): 1030
   * Number of Attributes: 9
   * Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
   * Missing Attribute Values: None
                              
**Variable Information:** Given is the variable name, variable type, the measurement unit and a brief description.
   The concrete compressive strength is the regression problem. The order of this listing corresponds to 
   the order of numerals along the rows of the database.
                              
      Name -- Data Type -- Measurement -- Description
      
   1- Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
   
   2- Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
   
   ![image](https://github.com/user-attachments/assets/1e7ee6a3-848f-408c-bec3-5718d60e6fc9)

   
   3- Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
   
   ![image](https://github.com/user-attachments/assets/34c3ecb6-6890-43d8-b114-d37ae7ce6a15)

   
   4- Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
   
   5- Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
   
   ![image](https://github.com/user-attachments/assets/316937cb-268e-432d-8d64-8c7868695777)

   
   6- Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
   
   7- Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
   
   ![image](https://github.com/user-attachments/assets/3bd37cf8-964f-49cd-8f51-667500d653fd)

   
   8- Age -- quantitative -- Day (1~365) -- Input Variable
   
   9- Concrete compressive strength -- quantitative -- MPa -- Output Variable

                              
# **3. Analysis:** 
* Univariate Analysis on Strength and making a classification for each category of strength, 
  that makes me converting the problem to not just a regression problem, but also classification!
* Bivariate Analysis on Period of Drying the Concrete with Strength
                    
# **4. Ml Model:** 
**1. Regression** (used models):
   1. MinMaxScaler
   2. StandardScaler
   3. RobustScaler
   4. Polynomial 
   5. Ridge
   6. Lasso
   7. XGBRegressor
      
      Highest accuracy: XGBRegressor model 
      Train: 96.99%             Test: 93.96%

        
**2. Classification** (used models):
   1. LogisticRegression ‘scaler’
   2. SVC
   3. KNeighbors
   4. RandomForest 
   5. DecisionTree
   6. AdaBoost
   7. XGBoost
      
      Highest accuracy: XGBoost model ‘minmax’ 
      Train: 93.66%             Test: 93.03%     
