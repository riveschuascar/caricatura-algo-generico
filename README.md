README - Proyecto de Procesamiento de Imágenes con Algoritmo Genético

Este proyecto utiliza un algoritmo genético para la transformación de imágenes, haciendo uso de MediaPipe y scikit-image.

------------------------------------------------------------
Versión de Python
------------------------------------------------------------
< 3.13
Sugerido: Python 3.12

------------------------------------------------------------
Dependencias
------------------------------------------------------------
- mediapipe
- scikit-image

Estas se instalarán automáticamente desde requirements.txt.

------------------------------------------------------------
Crear entorno virtual
------------------------------------------------------------
```py -3.12 -m venv .env```

------------------------------------------------------------
Activar entorno virtual
------------------------------------------------------------
Windows (PowerShell):
```.env\Scripts\Activate```

Windows (CMD):
```.env\Scripts\activate.bat```

Windows (Git Bash):
```source .env/Scripts/activate```

Mac / Linux:
```source .env/bin/activate```

------------------------------------------------------------
Instalar dependencias
------------------------------------------------------------
```pip install -r requirements.txt```

------------------------------------------------------------
Ejecutar transformación de imagen
------------------------------------------------------------
```py algoritmo_genetico.py 2>null```
(En Mac/Linux usar: 2>/dev/null)

------------------------------------------------------------
Ejecutar recopilación de resultados
------------------------------------------------------------
```py get_summary_data.py```

------------------------------------------------------------
Desactivar entorno virtual (opcional)
------------------------------------------------------------
```deactivate```
