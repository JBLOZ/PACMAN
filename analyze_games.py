#!/usr/bin/env python3
"""
Script para analizar las partidas de Pacman y proporcionar estadísticas
para decidir qué partidas mantener o eliminar según un umbral de puntos.
"""

import os
import glob
import csv
import json
from collections import defaultdict
import pandas as pd
import argparse

def analyze_game_file(csv_file):
    """Analiza un archivo CSV de una partida y extrae estadísticas"""
    game_stats = {
        'file': os.path.basename(csv_file),
        'total_moves': 0,
        'pacman_moves': 0,
        'final_score': 0,
        'max_score': 0,
        'is_win': False,
        'is_lose': False,
        'game_over': False,
        'actions': [],
        'scores': [],
        'error': None
    }
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                game_stats['total_moves'] += 1
                
                # Solo contar movimientos de Pacman (agente 0)
                agent_index = int(row.get('agent_index', 0))
                if agent_index == 0:
                    game_stats['pacman_moves'] += 1
                    
                    # Extraer información del movimiento
                    action = row.get('action', '')
                    score = float(row.get('score', 0))
                    is_win = row.get('is_win', 'False').lower() == 'true'
                    is_lose = row.get('is_lose', 'False').lower() == 'true'
                    game_over = row.get('game_over', 'False').lower() == 'true'
                    
                    game_stats['actions'].append(action)
                    game_stats['scores'].append(score)
                    game_stats['final_score'] = score
                    game_stats['max_score'] = max(game_stats['max_score'], score)
                    
                    # Actualizar estado final
                    if is_win:
                        game_stats['is_win'] = True
                    if is_lose:
                        game_stats['is_lose'] = True
                    if game_over:
                        game_stats['game_over'] = True
                        
    except Exception as e:
        game_stats['error'] = str(e)
    
    return game_stats

def analyze_all_games(data_dir="pacman_data"):
    """Analiza todas las partidas en el directorio especificado"""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No se encontraron archivos CSV en {data_dir}")
        return []
    
    print(f"Analizando {len(csv_files)} archivos de partidas...")
    
    all_stats = []
    for csv_file in csv_files:
        stats = analyze_game_file(csv_file)
        all_stats.append(stats)
    
    return all_stats

def print_summary_statistics(all_stats):
    """Imprime estadísticas resumidas de todas las partidas"""
    if not all_stats:
        print("No hay datos para analizar")
        return
    
    # Filtrar partidas con errores
    valid_stats = [s for s in all_stats if s['error'] is None]
    error_stats = [s for s in all_stats if s['error'] is not None]
    
    print("\n" + "="*60)
    print("RESUMEN GENERAL DE PARTIDAS")
    print("="*60)
    print(f"Total de archivos analizados: {len(all_stats)}")
    print(f"Partidas válidas: {len(valid_stats)}")
    print(f"Partidas con errores: {len(error_stats)}")
    
    if error_stats:
        print("\nArchivos con errores:")
        for stats in error_stats:
            print(f"  - {stats['file']}: {stats['error']}")
    
    if not valid_stats:
        return
    
    # Estadísticas de puntuación
    scores = [s['final_score'] for s in valid_stats]
    max_scores = [s['max_score'] for s in valid_stats]
    
    print(f"\nESTADÍSTICAS DE PUNTUACIÓN:")
    print(f"Puntuación final promedio: {sum(scores)/len(scores):.2f}")
    print(f"Puntuación final mínima: {min(scores):.2f}")
    print(f"Puntuación final máxima: {max(scores):.2f}")
    print(f"Puntuación máxima alcanzada promedio: {sum(max_scores)/len(max_scores):.2f}")
    print(f"Puntuación máxima alcanzada máxima: {max(max_scores):.2f}")
    
    # Estadísticas de resultados
    wins = len([s for s in valid_stats if s['is_win']])
    losses = len([s for s in valid_stats if s['is_lose']])
    ongoing = len([s for s in valid_stats if not s['is_win'] and not s['is_lose']])
    
    print(f"\nESTADÍSTICAS DE RESULTADOS:")
    print(f"Partidas ganadas: {wins} ({wins/len(valid_stats)*100:.1f}%)")
    print(f"Partidas perdidas: {losses} ({losses/len(valid_stats)*100:.1f}%)")
    print(f"Partidas sin terminar: {ongoing} ({ongoing/len(valid_stats)*100:.1f}%)")
    
    # Estadísticas de movimientos
    moves = [s['pacman_moves'] for s in valid_stats]
    print(f"\nESTADÍSTICAS DE MOVIMIENTOS:")
    print(f"Movimientos promedio por partida: {sum(moves)/len(moves):.1f}")
    print(f"Movimientos mínimos: {min(moves)}")
    print(f"Movimientos máximos: {max(moves)}")

def print_detailed_ranking(all_stats, top_n=10):
    """Imprime ranking detallado de las mejores y peores partidas"""
    valid_stats = [s for s in all_stats if s['error'] is None]
    
    if not valid_stats:
        return
    
    print(f"\n" + "="*60)
    print(f"TOP {top_n} MEJORES PARTIDAS (por puntuación final)")
    print("="*60)
    
    # Ordenar por puntuación final
    best_games = sorted(valid_stats, key=lambda x: x['final_score'], reverse=True)[:top_n]
    
    for i, game in enumerate(best_games, 1):
        status = "GANADA" if game['is_win'] else "PERDIDA" if game['is_lose'] else "SIN TERMINAR"
        print(f"{i:2d}. {game['file']:15s} | Puntos: {game['final_score']:6.1f} | "
              f"Máx: {game['max_score']:6.1f} | Movs: {game['pacman_moves']:3d} | {status}")
    
    print(f"\n" + "="*60)
    print(f"TOP {top_n} PEORES PARTIDAS (por puntuación final)")
    print("="*60)
    
    # Ordenar por puntuación final (ascendente)
    worst_games = sorted(valid_stats, key=lambda x: x['final_score'])[:top_n]
    
    for i, game in enumerate(worst_games, 1):
        status = "GANADA" if game['is_win'] else "PERDIDA" if game['is_lose'] else "SIN TERMINAR"
        print(f"{i:2d}. {game['file']:15s} | Puntos: {game['final_score']:6.1f} | "
              f"Máx: {game['max_score']:6.1f} | Movs: {game['pacman_moves']:3d} | {status}")

def suggest_thresholds(all_stats):
    """Sugiere umbrales para filtrar partidas"""
    valid_stats = [s for s in all_stats if s['error'] is None]
    
    if not valid_stats:
        return
    
    scores = [s['final_score'] for s in valid_stats]
    scores.sort()
    
    print(f"\n" + "="*60)
    print("SUGERENCIAS DE UMBRALES")
    print("="*60)
    
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        idx = int(len(scores) * p / 100) - 1
        if idx >= 0:
            threshold = scores[idx]
            above_threshold = len([s for s in scores if s >= threshold])
            print(f"Percentil {p:2d}: {threshold:6.1f} puntos -> Mantendría {above_threshold:3d} partidas "
                  f"({above_threshold/len(valid_stats)*100:.1f}%)")
    
    # Umbrales específicos
    print(f"\nUMBRALES ESPECÍFICOS:")
    specific_thresholds = [0, 10, 25, 50, 100, 200, 500]
    for threshold in specific_thresholds:
        above_threshold = len([s for s in scores if s >= threshold])
        print(f"≥ {threshold:3d} puntos: {above_threshold:3d} partidas ({above_threshold/len(valid_stats)*100:.1f}%)")

def filter_games_by_threshold(all_stats, threshold, dry_run=True):
    """Filtra partidas según un umbral de puntuación"""
    valid_stats = [s for s in all_stats if s['error'] is None]
    
    to_keep = [s for s in valid_stats if s['final_score'] >= threshold]
    to_remove = [s for s in valid_stats if s['final_score'] < threshold]
    
    print(f"\n" + "="*60)
    print(f"FILTRADO CON UMBRAL ≥ {threshold} PUNTOS")
    print("="*60)
    print(f"Partidas a mantener: {len(to_keep)}")
    print(f"Partidas a eliminar: {len(to_remove)}")
    
    if to_remove:
        print(f"\nPartidas que se eliminarían:")
        for game in sorted(to_remove, key=lambda x: x['final_score']):
            status = "GANADA" if game['is_win'] else "PERDIDA" if game['is_lose'] else "SIN TERMINAR"
            print(f"  - {game['file']:15s} | {game['final_score']:6.1f} puntos | {status}")
    
    if not dry_run and to_remove:
        print(f"\n¿Proceder a eliminar {len(to_remove)} archivos? (s/N): ", end="")
        confirm = input().strip().lower()
        if confirm == 's':
            removed_count = 0
            for game in to_remove:
                try:
                    file_path = os.path.join("pacman_data", game['file'])
                    os.remove(file_path)
                    removed_count += 1
                    print(f"Eliminado: {game['file']}")
                except Exception as e:
                    print(f"Error eliminando {game['file']}: {e}")
            print(f"\nSe eliminaron {removed_count} archivos.")
        else:
            print("Operación cancelada.")
    
    return to_keep, to_remove

def save_analysis_report(all_stats, output_file="game_analysis_report.txt"):
    """Guarda un reporte detallado del análisis"""
    valid_stats = [s for s in all_stats if s['error'] is None]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE ANÁLISIS DE PARTIDAS DE PACMAN\n")
        f.write("="*50 + "\n\n")
        
        # Estadísticas por archivo
        f.write("DETALLE POR ARCHIVO:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Archivo':<20} {'Puntos':<8} {'Máx':<8} {'Movs':<6} {'Estado':<12}\n")
        f.write("-" * 80 + "\n")
        
        for game in sorted(valid_stats, key=lambda x: x['final_score'], reverse=True):
            status = "GANADA" if game['is_win'] else "PERDIDA" if game['is_lose'] else "SIN TERMINAR"
            f.write(f"{game['file']:<20} {game['final_score']:<8.1f} {game['max_score']:<8.1f} "
                   f"{game['pacman_moves']:<6} {status:<12}\n")
    
    print(f"\nReporte guardado en: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analizar partidas de Pacman')
    parser.add_argument('--dir', default='pacman_data', help='Directorio con archivos CSV')
    parser.add_argument('--threshold', type=float, help='Umbral de puntuación para filtrar')
    parser.add_argument('--remove', action='store_true', help='Eliminar archivos por debajo del umbral')
    parser.add_argument('--top', type=int, default=10, help='Número de mejores/peores partidas a mostrar')
    parser.add_argument('--report', help='Archivo para guardar reporte detallado')
    
    args = parser.parse_args()
    
    # Analizar todas las partidas
    all_stats = analyze_all_games(args.dir)
    
    if not all_stats:
        return
    
    # Mostrar estadísticas
    print_summary_statistics(all_stats)
    print_detailed_ranking(all_stats, args.top)
    suggest_thresholds(all_stats)
    
    # Filtrar por umbral si se especifica
    if args.threshold is not None:
        filter_games_by_threshold(all_stats, args.threshold, dry_run=not args.remove)
    
    # Guardar reporte si se especifica
    if args.report:
        save_analysis_report(all_stats, args.report)

if __name__ == "__main__":
    main()
