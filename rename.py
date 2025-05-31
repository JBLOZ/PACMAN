import os

def renombrar_csv_sin_colisiones(
    carpeta_origen: str = r"pacman_data",
    prefijo_temp: str = "__temp__",
    prefijo_final: str = "game_"
):
    """
    Renombra TODOS los archivos .csv en 'carpeta_origen' en dos fases:
      1) original → __temp__<i>.csv
      2) __temp__<i>.csv → game_<i>.csv

    De esta forma evitamos colisiones entre nombres intermedios y finales.
    """

    # 1) Verificar que exista la carpeta de trabajo
    if not os.path.isdir(carpeta_origen):
        raise FileNotFoundError(f"La carpeta especificada no existe: {carpeta_origen}")

    # 2) Obtener la lista inicial de archivos .csv *origina*l, ignorando mayúsculas/minúsculas
    #    Hacemos esto antes de crear los temporales, para no recogerlos en este listado.
    todos_los_archivos = os.listdir(carpeta_origen)
    # Filtramos SOLO los que terminan en .csv (independientemente de si ya eran "game_X.csv" u otro nombre)
    csv_originales = [f for f in todos_los_archivos if f.lower().endswith(".csv")]

    # Ordenamos alfabéticamente (o por otro criterio si se desea; aquí es una convención sencilla)
    csv_originales.sort()

    # 3) Fase 1: Renombrar cada archivo original a un nombre temporal único
    #    Ejemplo: "cualquiera.csv" → "__temp__0.csv", "__temp__1.csv", ...
    contador = 0
    for nombre_orig in csv_originales:
        ruta_orig = os.path.join(carpeta_origen, nombre_orig)
        nombre_temp = f"{prefijo_temp}{contador}.csv"
        ruta_temp = os.path.join(carpeta_origen, nombre_temp)

        # Si ruta_temp ya existiera (quizás de una ejecución anterior),
        # eliminamos para evitar error, asumiendo que no es necesario conservarlo.
        if os.path.exists(ruta_temp):
            os.remove(ruta_temp)

        os.rename(ruta_orig, ruta_temp)
        print(f"[FASE 1] Renombrado: {nombre_orig} → {nombre_temp}")
        contador += 1

    total_temporales = contador
    print(f"\n→ Fase 1 completada: {total_temporales} archivos renombrados a temporales.")

    # 4) Fase 2: Listar todos los temporales creados (__temp__*.csv) y pasarlos a game_<i>.csv
    #    Para garantizar orden, volvemos a ordenar por nombre (que incluye el contador).
    temporales = [f for f in os.listdir(carpeta_origen)
                  if f.startswith(prefijo_temp) and f.lower().endswith(".csv")]
    temporales.sort()

    for temp_name in temporales:
        # Extraemos el índice <i> del nombre "__temp__<i>.csv"
        # Sabemos que el patrón es prefijo_temp + número + ".csv"
        indice_str = temp_name[len(prefijo_temp):-4]  # quita "__temp__" y ".csv"
        try:
            indice = int(indice_str)
        except ValueError:
            # Si por alguna razón no se ajusta al patrón (p.ej. "__temp__abc.csv"),
            # lo renombramos según el orden actual de la lista
            indice = temporales.index(temp_name)

        ruta_temp = os.path.join(carpeta_origen, temp_name)
        nombre_final = f"{prefijo_final}{indice}.csv"
        ruta_final = os.path.join(carpeta_origen, nombre_final)

        # Si existe un game_<i>.csv previo (muy improbable, dado que venimos de temporales),
        # lo eliminamos para evitar colisión
        if os.path.exists(ruta_final):
            os.remove(ruta_final)

        os.rename(ruta_temp, ruta_final)
        print(f"[FASE 2] Renombrado: {temp_name} → {nombre_final}")

    print(f"\n→ Fase 2 completada: {len(temporales)} archivos temporales renombrados a finales.")

if __name__ == "__main__":
    # SOLO necesitas ajustar 'carpeta_origen' si tu carpeta difiere de "pacman_data".
    carpeta = r"pacman_data"
    renombrar_csv_sin_colisiones(carpeta_origen=carpeta)
