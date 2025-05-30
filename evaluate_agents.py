#!/usr/bin/env python3
"""
Script para evaluar y comparar el rendimiento de diferentes agentes de Pacman.
"""

import subprocess
import re
import os
from collections import defaultdict

def run_pacman_test(agent, layout, num_games=10, quiet=True, extra_args=None):
    """Ejecuta una prueba con un agente específico y extrae estadísticas"""
    cmd = [
        'python', 'pacman.py',
        '-p', agent,
        '-l', layout,
        '-n', str(num_games)
    ]
    
    if quiet:
        cmd.append('-q')
    
    # Agregar argumentos adicionales si se proporcionan
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd=r'c:\Users\jordi\Documents\UNI\PACMAN')
        
        # Extraer estadísticas del output
        output = result.stdout
        
        # Buscar puntuación promedio
        avg_match = re.search(r'Average Score: ([\d\.-]+)', output)
        avg_score = float(avg_match.group(1)) if avg_match else 0
        
        # Buscar tasa de victorias
        win_match = re.search(r'Win Rate:\s+(\d+)/(\d+)', output)
        if win_match:
            wins = int(win_match.group(1))
            total = int(win_match.group(2))
            win_rate = wins / total if total > 0 else 0
        else:
            win_rate = 0
        
        # Buscar puntuaciones individuales
        scores_match = re.search(r'Scores:\s+(.*)', output)
        scores = []
        if scores_match:
            score_str = scores_match.group(1)
            scores = [float(s.strip().rstrip(',')) for s in score_str.split(',') if s.strip()]
        
        return {
            'agent': agent,
            'layout': layout,
            'num_games': num_games,
            'avg_score': avg_score,
            'win_rate': win_rate,
            'scores': scores,
            'output': output
        }
        
    except Exception as e:
        print(f"Error ejecutando prueba para {agent} en {layout}: {e}")
        return None

def compare_agents(agents, layouts, num_games=5):
    """Compara múltiples agentes en múltiples layouts"""
    results = []
    
    print("🎮 EVALUACIÓN COMPARATIVA DE AGENTES DE PACMAN")
    print("=" * 60)
    
    for layout in layouts:
        print(f"\n📍 Layout: {layout}")
        print("-" * 40)
        
        layout_results = []
        
        for agent in agents:
            print(f"Probando {agent}... ", end="", flush=True)
            
            # Configurar parámetros específicos por agente
            if agent == 'MinimaxAgent':
                # Para MinimaxAgent usar depth=2 para mejor rendimiento
                result = run_pacman_test(agent, layout, num_games, quiet=True, extra_args=['-a', 'depth=2'])
            else:
                result = run_pacman_test(agent, layout, num_games, quiet=True)
            
            if result:
                layout_results.append(result)
                results.append(result)
                print(f"✓ Promedio: {result['avg_score']:.1f}, "
                      f"Victorias: {result['win_rate']*100:.0f}%")
            else:
                print("✗ Error")
        
        # Mostrar comparación para este layout
        if layout_results:
            best_avg = max(layout_results, key=lambda x: x['avg_score'])
            best_win = max(layout_results, key=lambda x: x['win_rate'])
            
            print(f"\n🏆 Mejor promedio: {best_avg['agent']} ({best_avg['avg_score']:.1f})")
            print(f"🎯 Mejor tasa de victoria: {best_win['agent']} ({best_win['win_rate']*100:.0f}%)")
    
    return results

def detailed_analysis(results):
    """Análisis detallado de los resultados"""
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS DETALLADO")
    print("=" * 60)
    
    # Agrupar por agente
    by_agent = defaultdict(list)
    for result in results:
        by_agent[result['agent']].append(result)
    
    # Estadísticas por agente
    print(f"\n{'Agente':<15} {'Prom. Global':<12} {'Victorias %':<11} {'Mejor':<8} {'Peor':<8}")
    print("-" * 60)
    
    agent_summaries = []
    for agent, agent_results in by_agent.items():
        all_scores = []
        total_wins = 0
        total_games = 0
        
        for result in agent_results:
            all_scores.extend(result['scores'])
            total_wins += result['win_rate'] * result['num_games']
            total_games += result['num_games']
        
        if all_scores:
            avg_global = sum(all_scores) / len(all_scores)
            win_rate_global = total_wins / total_games if total_games > 0 else 0
            best_score = max(all_scores)
            worst_score = min(all_scores)
            
            agent_summaries.append({
                'agent': agent,
                'global_avg': avg_global,
                'win_rate': win_rate_global,
                'best': best_score,
                'worst': worst_score
            })
            
            print(f"{agent:<15} {avg_global:<12.1f} {win_rate_global*100:<11.0f} "
                  f"{best_score:<8.0f} {worst_score:<8.0f}")
    
    return agent_summaries

def main():
    """Función principal que ejecuta todas las evaluaciones"""
    print("Iniciando evaluación comparativa...")
      # Definir agentes y layouts a probar
    agents = ['ReflexAgent', 'MinimaxAgent', 'AlphaBetaAgent', 'GreedyAgent']  # Agregado AlphaBetaAgent
    layouts = ['testClassic', 'smallClassic', 'mediumClassic']
    
    print(f"Agentes: {', '.join(agents)}")
    print(f"Layouts: {', '.join(layouts)}")
    print(f"Juegos por prueba: 5\n")
    
    # Ejecutar comparación
    results = compare_agents(agents, layouts, num_games=5)
    
    # Análisis detallado
    if results:
        agent_summaries = detailed_analysis(results)
        
        # Recomendación final
        print(f"\n{'='*60}")
        print("🎯 RECOMENDACIÓN")
        print("=" * 60)
        
        if agent_summaries:
            best_agent = max(agent_summaries, key=lambda x: x['global_avg'])
            
            print(f"✨ El mejor agente general es: {best_agent['agent']}")
            print(f"   Puntuación promedio global: {best_agent['global_avg']:.1f}")
            
            # Análisis específico para MinimaxAgent si está incluido
            minimax_stats = next((s for s in agent_summaries if s['agent'] == 'MinimaxAgent'), None)
            if minimax_stats:
                print(f"\n🤖 MinimaxAgent con depth=2:")
                print(f"   Puntuación promedio: {minimax_stats['global_avg']:.1f}")
                print(f"   Tasa de victorias: {minimax_stats['win_rate']:.1%}")
                print(f"   Mejor partida: {minimax_stats['best']:.0f}")
                print(f"   Peor partida: {minimax_stats['worst']:.0f}")
    
    else:
        print("❌ No se pudieron obtener resultados válidos")

if __name__ == "__main__":
    main()
