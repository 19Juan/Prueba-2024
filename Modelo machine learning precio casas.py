import pandas as pd

df=pd.read_csv('melb_data.csv')
#print (df)
#para quitar las columnas que no tienen valor
df=df.dropna(axis=0)
# Para imprimir los estadísticos descriptivos del dataframe
#print(df.describe())
#print(df)

#MACHINE LEARNING BROOO

y = df['Price']
x = df[['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']]

from sklearn.tree import DecisionTreeRegressor
modelo=DecisionTreeRegressor()
modelo.fit(x,y)

print("Predicciones las primeras 5 casas:")
print(x.head())
print("Los precios reales son:")
print(y.head())
print("Las predicciones de precio son:")
print(modelo.predict(x.head()))
#LAS PREDICCIONES Y EL VALOR REAL SON IGUALES!! 

#MAE = ERROR ABSOLUTO QUE SE ESTÁ COMETIENDO , para calcular qué tan bien está funcionando el modelo de la realidad
#MAE ES EL ERROR PROMEDIO DE CADA PREDICCIÓN POR ARRIBA O POR ABAJO DEL VALOR REAL
from sklearn.metrics import mean_absolute_error

predicciones_de_precio=modelo.predict(x)
MAE = mean_absolute_error(y,predicciones_de_precio)
print("El error absoluto medio es: ", MAE)

#Validación ahora con datos que el modelo no haya visto
#train test split me separa los datos para usar unos para entrenar y otros para validar el modelo.
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(x,y,)

modelo=DecisionTreeRegressor()
modelo.fit(train_x,train_y)

validación_predicciones=modelo.predict(val_x)
print("El error absoluto medio de validación es: ", mean_absolute_error(val_y,validación_predicciones))

#El error de validación es mucho mayor que el error de entrenamiento, lo que significa que el modelo no está generalizando bien
#vamos a prevenir el overfitting
#Para ello vamos a usar un modelo más complejo, el Random Forest    
from sklearn.ensemble import RandomForestRegressor
modelo=RandomForestRegressor()
modelo.fit(train_x,train_y)
validación_predicciones=modelo.predict(val_x)
print("El error absoluto medio de validación es: ", mean_absolute_error(val_y,validación_predicciones))
#PARTE PARA QUE EL USUARIO INTRODUZCA LOS DATOS Y SE LE DE UNA PREDICCIÓN
def predict_house_price():
    # Hacer que el usuario meta la información que quiere predecir
    user_input = {}
    for column in x.columns:
        while True:
            try:
                value = float(input(f"Introduzca {column}: "))
                user_input[column] = value
                break
            except ValueError:
                print("Por favor introduce un número válido")
    
    # convertir a dataframe para que esté acorde al formato de los datos
    input_data = pd.DataFrame([user_input])
    
    # hacer la predicción
    prediction = modelo.predict(input_data)
    
    print(f"\nEl precio estimado de la casa es: ${prediction[0]:,.2f}")

# Ejemplo de uso
if __name__ == "__main__":
    predict_house_price()
