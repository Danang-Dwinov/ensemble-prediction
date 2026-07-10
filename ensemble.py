from dataset import df_x
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

class Ensemble:
   def __init__ (self, df_x):
      self.dataset = df_x
      
      self.x = [ 'X1',
                     'X2',
                     'X3',
                     'X4',
                     'X5',
                     'RSI',
                     'MA',
                     'VA',
                     'RV']

      self.y = ["Target"]
      
      self._prepare_()
      
   def _prepare_(self):
      start_train = "2024-06-26"
      end_train = "2026-06-26"
      #This Prediction for date Monday 29, I set in 26 for get predict :)
      start_test = "2026-06-26"
      end_test = "2026-06-26"
      
      self.df_train = self.dataset.loc[start_train:end_train]
      self.df_test = self.dataset.loc[start_test:end_test]
      
      self.x_train, self.y_train = self.df_train[self.x], self.df_train[self.y]
      self.x_test, self.y_test = self.df_test[self.x], self.df_test[self.y]

   #Multi-Layer Perception
   def model_mlp(self):
      pipe = make_pipeline(
                  StandardScaler(),
                  MLPClassifier(
                     hidden_layer_sizes=(32,16),
                     activation='relu', solver='lbfgs',
                     learning_rate_init=0.005,
                     max_iter=2000,
                     alpha=0.0005))
      pipe.fit(self.x_train, self.y_train)
      predict = pipe.predict(self.x_test)
      
      return predict.astype(int)
   
   #Random Forest
   def model_rf(self):
      pipe = make_pipeline(
                  StandardScaler(),
                  RandomForestClassifier(
                     n_estimators=400,
                     max_depth=7))
   
      pipe.fit(self.x_train, self.y_train)
      predict = pipe.predict(self.x_test)
      
      return predict.astype(int)
      
   #XGBoost 
   def model_xgb(self):
      pipe = make_pipeline(
                  StandardScaler(),
                  xgb.XGBClassifier(
                     n_estimators=500,
                     max_depth=4,
                     learning_rate=0.001))
   
      pipe.fit(self.x_train, self.y_train)
      predict = pipe.predict(self.x_test)
      
      return predict.astype(int)
      
   def vote(self):
      model_rf = self.model_rf()
      model_mlp = self.model_mlp()
      model_xgb = self.model_xgb()
      
      result = (model_rf+model_mlp+model xgb)
      
      if result >= 2:
         return 1
      return 0

if __name__ == "__main__":
   classifier = Ensemble(df_x)
   
   finally_result = classifier.vote()
   #1 = Buy
   #0 = Sell
   print(finally_result)