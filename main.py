from gurobipy import GRB, Model
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk

# Cargar datos
datos_capacidad_sacos = pd.read_csv("capacidad_por_saco.csv", header=None)
a = [int(datos_capacidad_sacos.iloc[j, 0]) for j in range(len(datos_capacidad_sacos))]

datos_costo_semillas = pd.read_csv("costo_saco.csv", header=None)
c = [[int(datos_costo_semillas.iloc[j, t]) for t in range(len(datos_costo_semillas.columns))] for j in range(len(datos_costo_semillas))]

datos_tiempo_demora = pd.read_csv("tiempo_demora.csv", header=None)
o = [int(datos_tiempo_demora.iloc[j, 0]) for j in range(len(datos_tiempo_demora))]

datos_kilos_fruta = pd.read_csv("kilos_fruta.csv", header=None)
f = [int(datos_kilos_fruta.iloc[j, 0]) for j in range(len(datos_kilos_fruta))]

datos_precio_venta = pd.read_csv("precio_venta.csv", header=None)
v = [[int(datos_precio_venta.iloc[j, t]) for t in range(len(datos_precio_venta.columns))] for j in range(len(datos_precio_venta))]

capital_inicial = int(pd.read_csv("capital_inicial.csv", header=None).iloc[0, 0])
cantidad_cuadrantes = int(pd.read_csv("cantidad_cuadrantes.csv", header=None).iloc[0, 0])

cantidad_semillas = len(datos_costo_semillas)
cantidad_periodos = len(datos_costo_semillas.columns)

# Definir los conjuntos (desde 0 hasta n-1)
cuadrantes = list(range(cantidad_cuadrantes))
semillas = list(range(cantidad_semillas))
periodos = list(range(cantidad_periodos))

# Crear el modelo de Gurobi
model = Model()
model.setParam("TimeLimit", 60)  # Establecer el tiempo máximo en segundos

# Crear variables
x = model.addVars([(j, k, t) for j in semillas for k in cuadrantes for t in periodos], vtype=GRB.BINARY, name="X")
y = model.addVars([(j, k, t) for j in semillas for k in cuadrantes for t in periodos], vtype=GRB.BINARY, name="Y")
i = model.addVars(periodos, vtype=GRB.CONTINUOUS, name="I", lb=0)
u = model.addVars([(j, t) for j in semillas for t in periodos], vtype=GRB.INTEGER, name="U", lb=0)
w = model.addVars([(j, t) for j in semillas for t in periodos], vtype=GRB.INTEGER, name="W", lb=0)

model.update()

# Restricciones

# Restricción 1: Activación de la plantación
for j in semillas:
    for k in cuadrantes:
        for t in periodos:
            limite = min(t + o[j], cantidad_periodos)
            model.addConstr(sum(y[j, k, l] for l in range(t, limite)) >= o[j] * x[j, k, t])

# Restricción 2: Solo una plantación por cuadrante
for k in cuadrantes:
    for t in periodos:
        model.addConstr(sum(y[j, k, t] for j in semillas) <= 1)

# Restricción 3: Inventario de dinero
for t in range(1, cantidad_periodos):
    gastos = sum(c[j][t] * w[j, t] for j in semillas)
    ingresos = sum(x[j, k, t - o[j]] * f[j] * v[j][t] for j in semillas for k in cuadrantes if t >= o[j])
    model.addConstr(i[t] == i[t - 1] - gastos + ingresos)

# Restricción 4: Condición inicial del inventario de dinero
model.addConstr(i[0] == capital_inicial - sum(c[j][0] * w[j, 0] for j in semillas))

# Restricción 5: Inventario de semillas
for j in semillas:
    for t in range(1, cantidad_periodos):
        model.addConstr(u[j, t] == u[j, t - 1] + a[j] * w[j, t] - sum(x[j, k, t] for k in cuadrantes))

# Restricción 6: Condición inicial del inventario de semillas
for j in semillas:
    model.addConstr(u[j, 0] == a[j] * w[j, 0] - sum(x[j, k, 0] for k in cuadrantes))

# Restricción 7: Terminar la cosecha antes de volver a plantar
for j in semillas:
    for k in cuadrantes:
        for t in periodos:
            model.addConstr(1 - x[j, k, t] >= sum(x[j, k, l] for l in range(t + 1, min(t + o[j], cantidad_periodos))))

model.update()

# Función objetivo
model.setObjective(i[cantidad_periodos - 1], GRB.MAXIMIZE)

# Optimización
model.optimize()



valor_optimo = model.objVal
matriz_x = np.full((cantidad_cuadrantes, cantidad_periodos), 0)
matriz_k = [0 for k in cuadrantes]


# Llenar la matriz con los tipos de semillas
for j in semillas:
    for k in cuadrantes:
        for t in periodos:
            if x[j, k, t].X > 0.5:  # Variable binaria, verificar si es 1
                matriz_x[k, t] = j + 1
                matriz_k[k] += 1


print('\n\n|-----------------------------------------|')

print('|Cantidad de veces plantadas por cuadrante|')

print('|-----------------------------------------|')

for k in cuadrantes:

    print(f'|Cuadrante {k}:      Planto {matriz_k[k]} veces         |')

print('|-----------------------------------------|')

print(f'|     Valor optimo: {valor_optimo} $CLP         |')

print('|-----------------------------------------|')

# Crear un DataFrame de Pandas para mejor legibilidad
df_decision = pd.DataFrame(matriz_x, index=[f'Cuadrante {k + 1}' for k in cuadrantes], columns=[f'{t + 1}' for t in periodos])

# Función para mostrar el DataFrame en una ventana de Tkinter
def mostrar_dataframe(df):
    root = tk.Tk()
    root.title("Tabla de Decisiones")
    
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    tree = ttk.Treeview(frame, columns=["Índice"] + list(df.columns), show='headings')
    tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Añadir encabezados de columna
    tree.heading("Índice", text="Periodo")
    tree.column("Índice", width=80)
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=20)

    # Añadir datos al treeview
    for index, row in df.iterrows():
        tree.insert("", "end", values=[index] + list(row))

    # Añadir una barra de desplazamiento
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    # Ejecutar la aplicación
    root.mainloop()

# Mostrar el DataFrame en una nueva ventana
mostrar_dataframe(df_decision)


#Matriz en csv por si hay algun problema de visualizacion
df_decision.to_csv("calendario.csv")






