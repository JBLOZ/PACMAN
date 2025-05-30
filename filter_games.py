#!/usr/bin/env python3
"""
Script para filtrar y limpiar partidas de Pacman según diferentes criterios.
"""

import os
import glob
import csv
import json
import shutil
from datetime import datetime

def get_game_stats(csv_file):
    """Obtiene estadísticas básicas de un archivo de partida"""
    stats = {
        'file': os.path.basename(csv_file),
        'path': csv_file,
        'final_score': 0,
        'max_score': 0,
        'moves': 0,
        'is_win': False,
        'is_lose': False,
        'valid': True,
        'error': None
    }
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get('agent_index', 0)) == 0:  # Solo Pacman
                    stats['moves'] += 1
                    score = float(row.get('score', 0))
                    stats['final_score'] = score
                    stats['max_score'] = max(stats['max_score'], score)
                    stats['is_win'] = row.get('is_win', 'False').lower() == 'true'
                    stats['is_lose'] = row.get('is_lose', 'False').lower() == 'true'
    except Exception as e:
        stats['valid'] = False
        stats['error'] = str(e)
    
    return stats

def create_backup(source_dir, backup_dir):
    """Crea una copia de seguridad del directorio"""
    if os.path.exists(backup_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{backup_dir}_{timestamp}"
    
    shutil.copytree(source_dir, backup_dir)
    print(f"✓ Backup creado en: {backup_dir}")
    return backup_dir

def filter_by_score(data_dir, min_score, backup=True, dry_run=True):
    """Elimina partidas con puntuación menor al umbral"""
    print(f"\n{'='*60}")
    print(f"FILTRAR POR PUNTUACIÓN MÍNIMA: {min_score}")
    print(f"{'='*60}")
    
    if backup and not dry_run:
        backup_dir = f"{data_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_backup(data_dir, backup_dir)
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    to_remove = []
    to_keep = []
    
    for csv_file in csv_files:
        stats = get_game_stats(csv_file)
        if stats['valid'] and stats['final_score'] < min_score:
            to_remove.append(stats)
        else:
            to_keep.append(stats)
    
    print(f"Partidas a mantener: {len(to_keep)}")
    print(f"Partidas a eliminar: {len(to_remove)}")
    
    if to_remove:
        print(f"\nPartidas que se eliminarían:")
        for game in sorted(to_remove, key=lambda x: x['final_score']):
            print(f"  - {game['file']:15s} | {game['final_score']:6.1f} puntos")
        
        if not dry_run:
            confirm = input(f"\n¿Eliminar {len(to_remove)} archivos? (s/N): ").strip().lower()
            if confirm == 's':
                removed = 0
                for game in to_remove:
                    try:
                        os.remove(game['path'])
                        removed += 1
                        print(f"Eliminado: {game['file']}")
                    except Exception as e:
                        print(f"Error eliminando {game['file']}: {e}")
                print(f"\n✓ Se eliminaron {removed} archivos.")
            else:
                print("Operación cancelada.")
        else:
            print(f"\n[MODO SIMULACIÓN] No se eliminó ningún archivo.")
    
    return len(to_keep), len(to_remove)

def filter_by_moves(data_dir, min_moves, max_moves=None, backup=True, dry_run=True):
    """Elimina partidas con muy pocos o demasiados movimientos"""
    print(f"\n{'='*60}")
    move_range = f"{min_moves}-{max_moves if max_moves else '∞'}"
    print(f"FILTRAR POR NÚMERO DE MOVIMIENTOS: {move_range}")
    print(f"{'='*60}")
    
    if backup and not dry_run:
        backup_dir = f"{data_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_backup(data_dir, backup_dir)
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    to_remove = []
    to_keep = []
    
    for csv_file in csv_files:
        stats = get_game_stats(csv_file)
        if stats['valid']:
            should_remove = (stats['moves'] < min_moves or 
                           (max_moves and stats['moves'] > max_moves))
            if should_remove:
                to_remove.append(stats)
            else:
                to_keep.append(stats)
        else:
            to_keep.append(stats)
    
    print(f"Partidas a mantener: {len(to_keep)}")
    print(f"Partidas a eliminar: {len(to_remove)}")
    
    if to_remove:
        print(f"\nPartidas que se eliminarían:")
        for game in sorted(to_remove, key=lambda x: x['moves']):
            print(f"  - {game['file']:15s} | {game['moves']:3d} movimientos | {game['final_score']:6.1f} puntos")
        
        if not dry_run:
            confirm = input(f"\n¿Eliminar {len(to_remove)} archivos? (s/N): ").strip().lower()
            if confirm == 's':
                removed = 0
                for game in to_remove:
                    try:
                        os.remove(game['path'])
                        removed += 1
                        print(f"Eliminado: {game['file']}")
                    except Exception as e:
                        print(f"Error eliminando {game['file']}: {e}")
                print(f"\n✓ Se eliminaron {removed} archivos.")
            else:
                print("Operación cancelada.")
        else:
            print(f"\n[MODO SIMULACIÓN] No se eliminó ningún archivo.")
    
    return len(to_keep), len(to_remove)

def keep_best_n_games(data_dir, n, backup=True, dry_run=True):
    """Mantiene solo las N mejores partidas por puntuación"""
    print(f"\n{'='*60}")
    print(f"MANTENER SOLO LAS {n} MEJORES PARTIDAS")
    print(f"{'='*60}")
    
    if backup and not dry_run:
        backup_dir = f"{data_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_backup(data_dir, backup_dir)
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_games = []
    
    for csv_file in csv_files:
        stats = get_game_stats(csv_file)
        if stats['valid']:
            all_games.append(stats)
    
    # Ordenar por puntuación final (descendente)
    all_games.sort(key=lambda x: x['final_score'], reverse=True)
    
    to_keep = all_games[:n]
    to_remove = all_games[n:]
    
    print(f"Partidas a mantener: {len(to_keep)}")
    print(f"Partidas a eliminar: {len(to_remove)}")
    
    if to_keep:
        print(f"\nMejores {len(to_keep)} partidas que se mantendrán:")
        for i, game in enumerate(to_keep, 1):
            print(f"  {i:2d}. {game['file']:15s} | {game['final_score']:6.1f} puntos | {game['moves']:3d} movs")
    
    if to_remove:
        print(f"\nPartidas que se eliminarían:")
        for game in to_remove:
            print(f"  - {game['file']:15s} | {game['final_score']:6.1f} puntos")
        
        if not dry_run:
            confirm = input(f"\n¿Eliminar {len(to_remove)} archivos? (s/N): ").strip().lower()
            if confirm == 's':
                removed = 0
                for game in to_remove:
                    try:
                        os.remove(game['path'])
                        removed += 1
                        print(f"Eliminado: {game['file']}")
                    except Exception as e:
                        print(f"Error eliminando {game['file']}: {e}")
                print(f"\n✓ Se eliminaron {removed} archivos.")
            else:
                print("Operación cancelada.")
        else:
            print(f"\n[MODO SIMULACIÓN] No se eliminó ningún archivo.")
    
    return len(to_keep), len(to_remove)

def interactive_cleanup():
    """Modo interactivo para limpiar partidas"""
    data_dir = "pacman_data"
    
    print("🎮 LIMPIEZA INTERACTIVA DE PARTIDAS DE PACMAN")
    print("="*50)
    
    while True:
        print("\nOpciones disponibles:")
        print("1. Filtrar por puntuación mínima")
        print("2. Filtrar por número de movimientos")
        print("3. Mantener solo las N mejores partidas")
        print("4. Ver estadísticas actuales")
        print("5. Salir")
        
        choice = input("\nSelecciona una opción (1-5): ").strip()
        
        if choice == '1':
            try:
                min_score = float(input("Puntuación mínima: "))
                dry_run = input("¿Modo simulación? (S/n): ").strip().lower() != 'n'
                filter_by_score(data_dir, min_score, dry_run=dry_run)
            except ValueError:
                print("❌ Error: Introduce un número válido")
        
        elif choice == '2':
            try:
                min_moves = int(input("Número mínimo de movimientos: "))
                max_input = input("Número máximo de movimientos (Enter para sin límite): ").strip()
                max_moves = int(max_input) if max_input else None
                dry_run = input("¿Modo simulación? (S/n): ").strip().lower() != 'n'
                filter_by_moves(data_dir, min_moves, max_moves, dry_run=dry_run)
            except ValueError:
                print("❌ Error: Introduce números válidos")
        
        elif choice == '3':
            try:
                n = int(input("¿Cuántas mejores partidas mantener?: "))
                dry_run = input("¿Modo simulación? (S/n): ").strip().lower() != 'n'
                keep_best_n_games(data_dir, n, dry_run=dry_run)
            except ValueError:
                print("❌ Error: Introduce un número válido")
        
        elif choice == '4':
            os.system("python analyze_games.py")
        
        elif choice == '5':
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Filtrar partidas de Pacman')
    parser.add_argument('--dir', default='pacman_data', help='Directorio con archivos CSV')
    parser.add_argument('--min-score', type=float, help='Puntuación mínima')
    parser.add_argument('--min-moves', type=int, help='Número mínimo de movimientos')
    parser.add_argument('--max-moves', type=int, help='Número máximo de movimientos')
    parser.add_argument('--keep-best', type=int, help='Mantener solo las N mejores partidas')
    parser.add_argument('--no-backup', action='store_true', help='No crear backup')
    parser.add_argument('--execute', action='store_true', help='Ejecutar eliminación (no simulación)')
    parser.add_argument('--interactive', action='store_true', help='Modo interactivo')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_cleanup()
        return
    
    dry_run = not args.execute
    backup = not args.no_backup
    
    if args.min_score is not None:
        filter_by_score(args.dir, args.min_score, backup, dry_run)
    elif args.min_moves is not None:
        filter_by_moves(args.dir, args.min_moves, args.max_moves, backup, dry_run)
    elif args.keep_best is not None:
        keep_best_n_games(args.dir, args.keep_best, backup, dry_run)
    else:
        print("Usa --interactive para modo interactivo o especifica un filtro")
        print("Ejemplo: python filter_games.py --min-score 500 --execute")

if __name__ == "__main__":
    main()
