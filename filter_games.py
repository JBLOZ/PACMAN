import os
import shutil

def filtrar_partidas_ganadas(record_str: str,
                             carpeta_origen: str = r"pacman_data",
                             carpeta_destino: str = r"pacman_wining_data"):
    """
    A partir de un string de registro tipo:
      "Record:        Win, Win, Loss, Win, Loss, ..."
    copia todos los archivos pacman_data\game_<i>.csv cuyo resultado sea "Win"
    a la carpeta pacman_wining_data\game_<j>.csv, donde j es el índice 0-basado de las wins.
    """

    # 1) Extraer solo los términos 'Win'/'Loss' (ignoramos 'Record:' y espacios)
    #    Separar por comas y limpiar espacios sobrantes.
    #    Ejemplo: "Record:   Win, Loss, Win" --> ["Win", "Loss", "Win"]
    # ----------------------------------------------------------------
    # Eliminamos el prefijo "Record:" (y cualquier espacio adyacente)
    if record_str.lower().startswith("record:"):
        contenido = record_str[len("record:"):].strip()
    else:
        contenido = record_str.strip()

    # Dividimos por comas, y hacemos strip() en cada elemento
    resultados = [r.strip() for r in contenido.split(",") if r.strip()]
    # Ahora resultados = ["Win", "Win", "Loss", "Win", ...] (longitud = número de partidas)

    # 2) Nos aseguramos de que la carpeta de destino existe (si no, la creamos)
    # ------------------------------------------------------------------------
    if not os.path.isdir(carpeta_destino):
        os.makedirs(carpeta_destino)

    # 3) Recorremos cada resultado. Para cada índice i donde resultados[i] == "Win",
    #    copiamos "pacman_data/game_i.csv" a "pacman_wining_data/game_<contador_win>.csv".
    # ------------------------------------------------------------------------
    contador_win = 63
    for indice_partida, estado in enumerate(resultados):
        # Normalizamos a minúsculas para comparar
        if estado.lower() == "win":
            archivo_origen = os.path.join(carpeta_origen, f"game_{indice_partida}.csv")
            archivo_destino = os.path.join(carpeta_destino, f"game_{contador_win}.csv")

            if os.path.isfile(archivo_origen):
                # Copiamos el archivo; si solo quieres moverlo en lugar de copiar, usa shutil.move()
                shutil.copy(archivo_origen, archivo_destino)
                print(f"Copiado: {archivo_origen}  →  {archivo_destino}")
            else:
                print(f"¡Advertencia! No existe el archivo de origen: {archivo_origen}")

            contador_win += 1

    print(f"\nTotal de wins encontradas y copiadas: {contador_win}")


if __name__ == "__main__":
    # Ejemplo de uso:
    registro = "Record:        Loss, Win, Win, Win, Win, Loss, Loss, Win, Loss, Win, Win, Loss, Win, Loss, Win, Win, Win, Loss, Loss, Win, Loss, Loss, Win, Loss, Win, Loss, Win, Loss, Loss, Win, Win, Win, Loss, Win, Win, Win, Loss, Win, Win, Win, Win, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Win, Loss, Win, Win, Win, Loss, Win, Win, Win, Loss, Win, Loss, Win, Win, Loss, Loss, Win, Loss, Loss, Win, Win, Win, Win, Loss, Loss, Win, Loss, Win, Win, Win, Loss, Loss, Win, Win, Loss, Loss, Loss, Win, Loss, Loss, Loss, Win, Win, Win, Win, Loss, Win, Loss, Loss, Loss, Win, Win, Win, Loss, Win, Loss, Win, Win, Loss, Loss, Win, Loss, Loss, Loss, Win, Loss, Win, Win, Win, Loss, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Win, Win, Win, Win, Win, Loss, Loss, Win, Loss, Win, Loss, Win, Win, Win, Loss, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Loss, Win, Win, Loss, Loss, Win, Win, Loss, Win, Win, Loss, Loss, Win, Win, Win, Loss, Win, Win, Loss, Win, Loss, Win, Win, Loss, Win, Loss, Win, Loss, Win, Loss, Win, Win, Win, Loss, Win, Loss, Loss, Loss, Win, Win, Win, Loss, Win, Win, Win, Win, Win, Loss, Win, Win, Loss, Loss, Loss, Loss, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Win, Loss, Loss, Win, Loss, Loss, Loss, Loss, Win, Loss, Win, Win, Loss, Win, Win, Loss, Win, Loss, Loss, Win, Loss, Win, Win, Loss, Loss, Loss, Loss, Win, Win, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Win, Loss, Win, Loss, Loss, Win, Win, Loss, Win, Loss, Loss, Loss, Loss, Win, Win, Loss, Win, Loss, Win, Loss, Win, Win, Win, Loss, Win, Loss, Win, Loss, Loss, Loss, Loss, Loss, Loss, Win, Loss, Loss, Win, Win, Win, Loss, Loss, Win, Loss, Win, Win, Win, Win, Loss, Loss, Loss, Win, Win, Loss, Loss, Loss, Win, Win, Win, Win, Loss, Loss, Win, Win, Loss, Loss, Win, Loss, Win, Win, Win, Loss, Loss, Win, Win, Loss, Win, Win, Loss, Loss, Win, Win, Win, Loss, Loss, Loss, Win, Loss, Loss, Loss, Win, Loss, Loss, Loss, Loss, Loss, Win, Loss, Loss, Loss, Loss, Loss, Win, Loss, Win, Win, Win, Loss, Loss, Loss, Win, Win, Win, Win, Win, Win, Loss, Loss, Loss, Loss, Loss, Win, Loss, Win, Win, Win, Loss, Win, Win, Loss, Win, Win, Loss, Loss, Loss, Win, Win, Win, Loss, Win, Win, Loss, Loss, Win"
    # (Aquí puede ir cualquier cadena con el mismo formato)
    filtrar_partidas_ganadas(registro)
