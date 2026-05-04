"""
quantize_model.py (개선 버전)
──────────────────────────────
FP32 모델을 다양한 양자화 방식으로 변환하고,
"양자화해도 성능 저하 없음"을 통계적으로 증명한다.

지원 양자화 방식:
  1. Dynamic Quantization (INT8)
  3. ONNX 변환 (배포용)

평가 지표:
  - 완주 성공률, 평균 속도, 라인 오차, 라인 전환 횟수
  - 추론 시간 (평균/p50/p95/p99)
  - 모델 크기 비교
  - 에피소드별 action 일치도 (FP32 vs 양자화)
  - 통계적 유의성 검정 (t-test)
"""

import os
import sys
import copy
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import gym
import f110_gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import (
    ENV_CONFIG, OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH, QUANTIZED_PATH,
    EVAL_MAX_STEPS, PROJECT_ROOT,
)
from sac_model import SAC, get_obs_dim
from pure_pursuit import PurePursuitController
from train.eval_node import (
    EvalMetrics, preprocess_obs, load_racing_lines,
    load_model, action_to_env, make_init_pose,
)


# ── 설정 ──────────────────────────────────────────────────────────────────────
QUANTIZE_EPISODES = 10           # 평가 에피소드 수 (통계 신뢰도를 위해 10 이상)
CALIBRATION_STEPS = 2000         # Static Quantization 캘리브레이션 스텝
ACTION_COMPARE_STEPS = 5000      # FP32 vs 양자화 action 비교 스텝

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'models', 'quantize_results')


# ══════════════════════════════════════════════════════════════════════════════
# 1. 양자화 방법들
# ══════════════════════════════════════════════════════════════════════════════

def quantize_dynamic(model: SAC) -> SAC:
    """
    동적 양자화: 가중치만 INT8로, 활성화는 런타임에 양자화.
    가장 간단하고 안전한 방법.
    """
    quantized = copy.deepcopy(model).cpu()
    quantized = torch.quantization.quantize_dynamic(
        quantized,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return quantized



def export_onnx(model: SAC, obs_dim: int, save_path: str):
    """
    ONNX 변환: 실차 배포용.
    PyTorch 없이도 ONNX Runtime으로 추론 가능.
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    dummy_input = torch.randn(1, obs_dim)

    # Actor만 export (추론 시 Actor만 필요)
    actor = model_cpu.actor

    torch.onnx.export(
        actor,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_mean', 'action_log_std'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_mean': {0: 'batch_size'},
            'action_log_std': {0: 'batch_size'},
        },
    )

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f'  ONNX 저장: {save_path} ({size_mb:.2f} MB)')
    return size_mb


# ══════════════════════════════════════════════════════════════════════════════
# 2. 평가 엔진
# ══════════════════════════════════════════════════════════════════════════════

class QuantEvalResult:
    """단일 에피소드의 평가 결과"""
    def __init__(self):
        self.laps = 0
        self.avg_speed = 0.0
        self.avg_deviation = 0.0
        self.line_switches = 0
        self.total_steps = 0
        self.progress_pct = 0.0
        self.collisions = 0
        self.inference_times = []

    def to_dict(self) -> dict:
        times = self.inference_times
        return {
            'laps': self.laps,
            'avg_speed': self.avg_speed,
            'avg_deviation': self.avg_deviation,
            'line_switches': self.line_switches,
            'total_steps': self.total_steps,
            'progress_pct': self.progress_pct,
            'collisions': self.collisions,
            'inference_time_mean_ms': float(np.mean(times)) * 1000 if times else 0,
            'inference_time_p50_ms': float(np.percentile(times, 50)) * 1000 if times else 0,
            'inference_time_p95_ms': float(np.percentile(times, 95)) * 1000 if times else 0,
            'inference_time_p99_ms': float(np.percentile(times, 99)) * 1000 if times else 0,
        }


def evaluate_model(
    model: SAC,
    env,
    waypoints_lines: list,
    controller: PurePursuitController,
    init_poses: np.ndarray,
    num_lines: int,
    n_episodes: int,
    label: str,
) -> list:
    """
    모델 평가: n_episodes 에피소드를 실행하고 결과 반환.
    """
    print(f'\n  [{label}] 평가 시작 ({n_episodes} 에피소드)')

    progress_ref = waypoints_lines[num_lines // 2]
    all_results = []

    for ep in range(n_episodes):
        result = QuantEvalResult()
        metrics = EvalMetrics(progress_ref)

        obs_raw, _, _, _ = env.reset(poses=init_poses)
        obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

        for step in range(EVAL_MAX_STEPS):
            # 추론 시간 측정
            t0 = time.perf_counter()
            action = model.select_action(obs, training=False)
            t1 = time.perf_counter()
            result.inference_times.append(t1 - t0)

            steering, target_speed, line_idx = action_to_env(
                action, obs_raw, model, waypoints_lines, controller,
            )

            env_action = np.array([[steering, target_speed]], dtype=np.float32)
            next_obs_raw, _, done, _ = env.step(env_action)

            if bool(next_obs_raw['collisions'][0]):
                result.collisions += 1

            metrics.update(next_obs_raw, target_speed, line_idx, waypoints_lines)

            obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
            obs_raw = next_obs_raw

            if done or int(next_obs_raw['lap_counts'][0]) >= 2:
                break

        result.laps = metrics.laps_completed
        result.avg_speed = float(np.mean(metrics.speeds)) if metrics.speeds else 0
        result.avg_deviation = float(np.mean(metrics.line_deviations)) if metrics.line_deviations else 0
        result.line_switches = metrics.line_switches
        result.total_steps = metrics.total_steps
        result.progress_pct = metrics.best_progress_pct

        all_results.append(result)

        print(
            f'    ep {ep+1:2d} | '
            f'lap: {result.laps} | '
            f'speed: {result.avg_speed:.2f} | '
            f'dev: {result.avg_deviation:.4f} | '
            f'col: {result.collisions} | '
            f'time: {np.mean(result.inference_times)*1000:.3f}ms'
        )

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Action 일치도 분석
# ══════════════════════════════════════════════════════════════════════════════

def compare_actions(
    fp32_model: SAC,
    quant_model: SAC,
    env,
    waypoints_lines: list,
    controller: PurePursuitController,
    init_poses: np.ndarray,
    num_lines: int,
    label: str,
    n_steps: int = ACTION_COMPARE_STEPS,
) -> dict:
    """
    동일 obs에 대해 FP32와 양자화 모델의 action 차이를 측정.
    action 일치도가 높을수록 양자화 품질이 좋음.
    """
    print(f'\n  [{label}] Action 일치도 분석 ({n_steps} 스텝)')

    obs_raw, _, _, _ = env.reset(poses=init_poses)
    obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

    action_diffs = []        # action 절대 차이
    line_matches = 0         # 라인 선택 일치 횟수
    speed_diffs = []         # 속도 차이
    total = 0

    for step in range(n_steps):
        fp32_action = fp32_model.select_action(obs, training=False)
        quant_action = quant_model.select_action(obs, training=False)

        # action 차이
        diff = np.abs(fp32_action - quant_action)
        action_diffs.append(diff)

        # 라인 선택 일치 여부
        fp32_line = fp32_model.action_to_line_index(fp32_action)
        quant_line = quant_model.action_to_line_index(quant_action)
        if fp32_line == quant_line:
            line_matches += 1

        # 속도 차이
        fp32_speed = fp32_model.action_to_speed(fp32_action, SPEED_MIN, SPEED_MAX)
        quant_speed = quant_model.action_to_speed(quant_action, SPEED_MIN, SPEED_MAX)
        speed_diffs.append(abs(fp32_speed - quant_speed))

        total += 1

        # FP32 모델 기준으로 env 진행
        steering, target_speed, _ = action_to_env(
            fp32_action, obs_raw, fp32_model, waypoints_lines, controller,
        )
        env_action = np.array([[steering, target_speed]], dtype=np.float32)
        next_obs_raw, _, done, _ = env.step(env_action)

        obs = preprocess_obs(next_obs_raw, waypoints_lines, num_lines)
        obs_raw = next_obs_raw

        if done:
            obs_raw, _, _, _ = env.reset(poses=init_poses)
            obs = preprocess_obs(obs_raw, waypoints_lines, num_lines)

    action_diffs = np.array(action_diffs)

    result = {
        'line_match_rate': line_matches / total * 100,
        'action_mean_diff': float(np.mean(action_diffs)),
        'action_max_diff': float(np.max(action_diffs)),
        'action_line_diff_mean': float(np.mean(action_diffs[:, 0])),
        'action_speed_diff_mean': float(np.mean(action_diffs[:, 1])),
        'speed_diff_mean_mps': float(np.mean(speed_diffs)),
        'speed_diff_max_mps': float(np.max(speed_diffs)),
    }

    print(f'    라인 선택 일치율 : {result["line_match_rate"]:.1f}%')
    print(f'    action 평균 차이 : {result["action_mean_diff"]:.6f}')
    print(f'    속도 평균 차이   : {result["speed_diff_mean_mps"]:.4f} m/s')

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. 통계 분석
# ══════════════════════════════════════════════════════════════════════════════

def t_test(a: list, b: list) -> tuple:
    """
    간단한 독립 t-검정.
    scipy 없이 직접 구현.

    Returns:
        (t_statistic, p_value_approx, significant)
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0, False

    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)

    se = np.sqrt(s1 / n1 + s2 / n2)
    if se < 1e-12:
        return 0.0, 1.0, False

    t_stat = (m1 - m2) / se

    # 자유도 (Welch's approximation)
    num = (s1 / n1 + s2 / n2) ** 2
    den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 1e-12 else 1

    # p-value 근사 (정규분포 근사, df > 30이면 충분히 정확)
    z = abs(t_stat)
    p_approx = 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)

    # 더 정확한 근사 (Abramowitz & Stegun)
    t_val = 1.0 / (1.0 + 0.2316419 * z)
    poly = t_val * (0.319381530 + t_val * (-0.356563782 + t_val * (1.781477937 + t_val * (-1.821255978 + 1.330274429 * t_val))))
    p_approx = 2.0 * poly * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    p_approx = min(max(p_approx, 0.0), 1.0)

    significant = p_approx < 0.05

    return float(t_stat), float(p_approx), significant


def statistical_comparison(fp32_results: list, quant_results: list, label: str):
    """
    FP32 vs 양자화 모델 결과를 통계적으로 비교.
    p < 0.05이면 유의미한 성능 차이가 있음.
    """
    print(f'\n  [{label}] 통계 비교 (t-검정, p < 0.05이면 유의미한 차이)')
    print(f'  {"─" * 70}')
    print(f'  {"지표":<18} {"FP32":>10} {"양자화":>10} {"차이":>10} {"p-value":>10} {"결론":>12}')
    print(f'  {"─" * 70}')

    metrics = [
        ('완주 lap', 'laps'),
        ('평균 속도', 'avg_speed'),
        ('라인 오차(m)', 'avg_deviation'),
        ('라인 전환', 'line_switches'),
        ('충돌 횟수', 'collisions'),
        ('진행률(%)', 'progress_pct'),
    ]

    all_pass = True

    for name, key in metrics:
        fp32_vals = [r.to_dict()[key] for r in fp32_results]
        quant_vals = [r.to_dict()[key] for r in quant_results]

        fp32_mean = np.mean(fp32_vals)
        quant_mean = np.mean(quant_vals)
        diff = quant_mean - fp32_mean

        t_stat, p_val, significant = t_test(fp32_vals, quant_vals)

        if significant:
            conclusion = '⚠️ 유의미 차이'
            all_pass = False
        else:
            conclusion = '✅ 차이 없음'

        print(
            f'  {name:<18} '
            f'{fp32_mean:>10.3f} '
            f'{quant_mean:>10.3f} '
            f'{diff:>+10.3f} '
            f'{p_val:>10.4f} '
            f'{conclusion:>12}'
        )

    # 추론 시간 비교
    fp32_times = [r.to_dict()['inference_time_mean_ms'] for r in fp32_results]
    quant_times = [r.to_dict()['inference_time_mean_ms'] for r in quant_results]
    speedup = np.mean(fp32_times) / np.mean(quant_times) if np.mean(quant_times) > 0 else 1.0

    print(f'  {"─" * 70}')
    print(
        f'  {"추론시간(ms)":<18} '
        f'{np.mean(fp32_times):>10.3f} '
        f'{np.mean(quant_times):>10.3f} '
        f'{np.mean(quant_times) - np.mean(fp32_times):>+10.3f} '
        f'{"":>10} '
        f'  {speedup:.2f}x'
    )
    print(f'  {"─" * 70}')

    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# 5. 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def get_model_size(model, path=None) -> float:
    """모델 크기(MB) 반환"""
    if path and os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)

    tmp = '/tmp/_tmp_model_size.pth'
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return size



def save_report(report: dict, path: str):
    """결과 리포트를 JSON으로 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # numpy 타입 → Python 기본 타입으로 변환
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert(report), f, indent=2, ensure_ascii=False)

    print(f'\n리포트 저장: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('\n' + '═' * 70)
    print('  SAC 모델 양자화 & 성능 비교 평가')
    print('  목표: 양자화 후에도 주행 성능 저하 없음을 증명')
    print('═' * 70)

    # ── 환경 준비 ─────────────────────────────────────────────────────────
    waypoints_lines = load_racing_lines()
    num_lines = MODEL_CONFIG.get('num_lines', LINE_CONFIG['num_lines'])
    controller = PurePursuitController(max_speed=SPEED_MAX, min_speed=SPEED_MIN)

    obs_dim = get_obs_dim(OBS_CONFIG['lidar_size'], num_lines)
    env = gym.make('f110_gym:f110-v0', **ENV_CONFIG)
    init_poses = make_init_pose(waypoints_lines)

    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'episodes': QUANTIZE_EPISODES,
            'max_steps': EVAL_MAX_STEPS,
            'map': ENV_CONFIG['map'],
            'speed_range': [SPEED_MIN, SPEED_MAX],
        },
        'models': {},
    }

    # ── FP32 모델 로드 ────────────────────────────────────────────────────
    print(f'\n{"─" * 70}')
    print('1. FP32 원본 모델 로드')
    print(f'{"─" * 70}')

    fp32_model = load_model(MODEL_SAVE_PATH)
    fp32_size = get_model_size(fp32_model, MODEL_SAVE_PATH)
    print(f'  모델 크기: {fp32_size:.2f} MB')

    # ── Dynamic Quantization ──────────────────────────────────────────────
    print(f'\n{"─" * 70}')
    print('2. Dynamic Quantization (INT8)')
    print(f'{"─" * 70}')

    dynamic_model = quantize_dynamic(fp32_model)
    dynamic_path = os.path.join(PROJECT_ROOT, 'models', 'sac_dynamic_int8.pth')
    torch.save(dynamic_model.state_dict(), dynamic_path)
    dynamic_size = get_model_size(dynamic_model, dynamic_path)
    print(f'  모델 크기: {dynamic_size:.2f} MB (압축률: {(1 - dynamic_size/fp32_size)*100:.1f}%)')


    # ── ONNX 변환 ─────────────────────────────────────────────────────────
    print(f'\n{"─" * 70}')
    print('4. ONNX 변환 (배포용)')
    print(f'{"─" * 70}')

    onnx_path = os.path.join(PROJECT_ROOT, 'models', 'sac_model.onnx')
    try:
        onnx_size = export_onnx(fp32_model, obs_dim, onnx_path)
    except Exception as e:
        print(f'  [경고] ONNX 변환 실패: {e}')
        onnx_size = 0

    # ── 모델 크기 요약 ────────────────────────────────────────────────────
    print(f'\n{"─" * 70}')
    print('5. 모델 크기 비교')
    print(f'{"─" * 70}')
    print(f'  {"모델":<25} {"크기(MB)":>10} {"압축률":>10}')
    print(f'  {"─" * 45}')
    print(f'  {"FP32 (원본)":<25} {fp32_size:>10.2f} {"기준":>10}')
    print(f'  {"Dynamic INT8":<25} {dynamic_size:>10.2f} {(1-dynamic_size/fp32_size)*100:>9.1f}%')
    if onnx_size > 0:
        print(f'  {"ONNX":<25} {onnx_size:>10.2f} {(1-onnx_size/fp32_size)*100:>9.1f}%')

    # ── 주행 성능 평가 ────────────────────────────────────────────────────
    print(f'\n{"═" * 70}')
    print('6. 주행 성능 평가')
    print(f'{"═" * 70}')

    models_to_eval = [
        ('FP32', fp32_model),
        ('Dynamic INT8', dynamic_model),
    ]

    all_eval_results = {}

    for label, model in models_to_eval:
        results = evaluate_model(
            model, env, waypoints_lines, controller, init_poses,
            num_lines, QUANTIZE_EPISODES, label,
        )
        all_eval_results[label] = results

        report['models'][label] = {
            'size_mb': get_model_size(model),
            'episodes': [r.to_dict() for r in results],
        }

    # ── Action 일치도 분석 ────────────────────────────────────────────────
    print(f'\n{"═" * 70}')
    print('7. Action 일치도 분석 (FP32 vs 양자화)')
    print(f'{"═" * 70}')

    for label, model in [('Dynamic INT8', dynamic_model)]:
        action_result = compare_actions(
            fp32_model, model, env, waypoints_lines,
            controller, init_poses, num_lines, label,
        )
        report['models'][label]['action_comparison'] = action_result

    # ── 통계 검정 ─────────────────────────────────────────────────────────
    print(f'\n{"═" * 70}')
    print('8. 통계적 유의성 검정')
    print(f'{"═" * 70}')

    all_pass = True
    for label in ['Dynamic INT8']:
        passed = statistical_comparison(
            all_eval_results['FP32'],
            all_eval_results[label],
            label,
        )
        if not passed:
            all_pass = False
        report['models'][label]['statistical_test_passed'] = passed

    # ── 최종 결론 ─────────────────────────────────────────────────────────
    print(f'\n{"═" * 70}')
    print('  최종 결론')
    print(f'{"═" * 70}')

    if all_pass:
        print('  ✅ 모든 양자화 모델에서 FP32 대비 유의미한 성능 저하 없음!')
        print('  → 양자화 모델을 실차에 안전하게 배포할 수 있습니다.')
        report['conclusion'] = 'PASS - 양자화 후 성능 저하 없음'
    else:
        print('  ⚠️ 일부 지표에서 유의미한 차이가 발견되었습니다.')
        print('  → 세부 결과를 확인하고, 해당 지표가 실제 주행에 영향을 주는지 검토하세요.')
        report['conclusion'] = 'REVIEW - 일부 지표에서 차이 발견'

    # ── 추천 모델 ─────────────────────────────────────────────────────────
    dynamic_time = np.mean([r.to_dict()['inference_time_mean_ms'] for r in all_eval_results['Dynamic INT8']])
    recommended = 'Dynamic INT8'
    rec_size = dynamic_size
    rec_time = dynamic_time

    print(f'\n  추천 양자화 모델: {recommended}')
    print(f'  크기: {rec_size:.2f} MB | 추론: {rec_time:.3f} ms')

    report['recommended_model'] = recommended

    # ── 리포트 저장 ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(RESULTS_DIR, f'quantize_report_{timestamp}.json')
    save_report(report, report_path)

    print(f'\n{"═" * 70}\n')

    env.close()


if __name__ == '__main__':
    main()