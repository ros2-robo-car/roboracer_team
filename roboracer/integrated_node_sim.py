"""
integrated_node_sim.py (시뮬레이션 환경용)
──────────────────────────────────────────
robo_bridge_f1tenth 브릿지와 연동되는 시뮬레이션 전용 통합 노드

실차용(integrated_node.py)과의 차이:
  - 구독: /scan (LaserScan) + /odom (Odometry)
          → f110_recv (Recv) 단일 토픽으로 변경
  - 발행: /drive (AckermannDriveStamped)
          → f110_send (Act) 으로 변경

토픽 구조:
  Subscribe: f110_recv (f110_gym_bridge_interface/Recv)
  Publish  : f110_send (f110_gym_bridge_interface/Act)
"""

import os
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import torch

from f110_gym_bridge_interface.msg import Recv, Act

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG, PURE_PURSUIT_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH,
)
from sac_model import SAC, get_obs_dim, build_observation
from waypoint_loader import load_waypoints
from pure_pursuit import PurePursuitController

# ── 실차 안전 속도 상수 ────────────────────────────────────────────────────────
REAL_SPEED_MAX = 3.0    # m/s
REAL_SPEED_MIN = 0.5    # m/s
MAX_STEERING   = 0.4189 # rad (~24도)

# ── LiDAR / 관측 상수 ────────────────────────────────────────────────────────
NUM_LINES        = LINE_CONFIG['num_lines']
LIDAR_SIZE       = OBS_CONFIG['lidar_size']       # 108 (다운샘플링 후)
LIDAR_RAW_SIZE   = 1080                            # 브릿지에서 오는 원본 크기
LIDAR_MIN        = OBS_CONFIG['lidar_range_min']
LIDAR_MAX        = OBS_CONFIG['lidar_range_max']
USE_CURVATURE    = OBS_CONFIG.get('use_line_curvature', False)
CURVATURE_MODE   = OBS_CONFIG.get('curvature_mode', 'max')
CURVATURE_MAX    = OBS_CONFIG.get('curvature_max_value', 1.5)
OBS_DIM          = get_obs_dim(LIDAR_SIZE, NUM_LINES, use_line_curvature=USE_CURVATURE)
OBS_DIM_FALLBACK = LIDAR_SIZE


# ── 곡률 계산 함수 (train_node.py 와 동일한 로직) ─────────────────────────────
def _compute_three_point_curvature(p0: np.ndarray,
                                   p1: np.ndarray,
                                   p2: np.ndarray) -> float:
    d01 = np.linalg.norm(p1 - p0)
    d12 = np.linalg.norm(p2 - p1)
    d02 = np.linalg.norm(p2 - p0)
    denom = d01 * d12 * d02
    if denom < 1e-9:
        return 0.0
    cross = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                (p1[1] - p0[1]) * (p2[0] - p0[0]))
    curvature = 2.0 * cross / denom
    if not np.isfinite(curvature):
        return 0.0
    return float(curvature)


def _compute_line_lookahead_curvatures(waypoints_lines: list,
                                       position: np.ndarray,
                                       speed: float) -> np.ndarray:
    if OBS_CONFIG.get('curvature_use_pp_window', True):
        base        = int(PURE_PURSUIT_CONFIG.get('lookahead_window_base', 5))
        scale       = int(PURE_PURSUIT_CONFIG.get('lookahead_window_speed_scale', 2))
        window      = base + int(abs(speed) * scale)
        sample_step = int(PURE_PURSUIT_CONFIG.get('curvature_sample_step', 2))
    else:
        window      = int(OBS_CONFIG.get('curvature_lookahead_window', 30))
        sample_step = int(OBS_CONFIG.get('curvature_sample_step', 2))

    max_curv = max(float(CURVATURE_MAX), 1e-6)
    line_curvatures = []

    for waypoints in waypoints_lines:
        dists   = np.linalg.norm(waypoints - position, axis=1)
        nearest = int(np.argmin(dists))
        n_wp    = len(waypoints)

        curvatures = []
        for i in range(nearest, min(nearest + window, n_wp - 2), sample_step):
            p0 = waypoints[i]
            p1 = waypoints[min(i + sample_step,     n_wp - 1)]
            p2 = waypoints[min(i + sample_step * 2, n_wp - 1)]
            curvatures.append(_compute_three_point_curvature(p0, p1, p2))

        if not curvatures:
            line_curvature = 0.0
        elif CURVATURE_MODE == 'max':
            line_curvature = float(np.max(curvatures))
        else:
            line_curvature = float(np.mean(curvatures))

        line_curvature = float(np.clip(line_curvature / max_curv, 0.0, 1.0))
        line_curvatures.append(line_curvature)

    return np.array(line_curvatures, dtype=np.float32)


class IntegratedNodeSim(Node):
    """
    시뮬레이션 환경용 통합 노드

    파이프라인 (recv_callback 내부):
      f110_recv ──► _process_lidar()         ← scans[1080] → lidar[108]
                         │
                         ▼
                  _extract_odom()            ← poses_x/y/theta, linear_vels_x
                         │
                         ▼
                  _compute_line_lookahead_curvatures()
                         │
                         ▼
                  build_observation()
                         │
                         ▼
                  SAC.select_action()
                  PurePursuit.compute()
                         │
                         ▼
                  _publish_act()
                         │
                         ▼
                  f110_send (Act)
    """

    def __init__(self):
        super().__init__('integrated_node_sim')

        self._load_waypoints()

        self.controller = PurePursuitController(
            max_speed=SPEED_MAX, min_speed=SPEED_MIN
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SAC(
            OBS_DIM,
            MODEL_CONFIG['action_dim'],
            MODEL_CONFIG['hidden_dims'],
            num_lines=NUM_LINES,
        ).to(self.device)

        if os.path.exists(MODEL_SAVE_PATH):
            ckpt = torch.load(MODEL_SAVE_PATH, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                self.model.load_state_dict(ckpt['model_state'])
            else:
                self.model.load_state_dict(ckpt)
            self.get_logger().info(f'모델 로드: {MODEL_SAVE_PATH}')
        else:
            self.get_logger().warn(f'모델 없음: {MODEL_SAVE_PATH}')
        self.model.eval()

        # 차량 상태 (f110_recv에서 직접 추출)
        self.position      = np.array([0.0, 0.0])
        self.heading       = 0.0
        self.speed         = 0.0
        self.odom_received = False

        # 실차용과 달리 단일 토픽으로 구독/발행
        self.recv_sub = self.create_subscription(
            Recv, 'f110_recv', self.recv_callback, 10
        )
        self.act_pub = self.create_publisher(
            Act, 'f110_send', 10
        )

        self.get_logger().info(
            f'integrated_node_sim started | obs_dim={self._obs_dim} '
            f'| use_curvature={USE_CURVATURE} | device={self.device}'
        )

    def _load_waypoints(self):
        try:
            csv = LINE_CONFIG['centerline_csv']
            if os.path.exists(csv):
                wp = load_waypoints(
                    centerline_path=csv,
                    num_lines=NUM_LINES,
                    line_spacing=LINE_CONFIG['line_spacing'],
                )
            else:
                wp = load_waypoints(
                    map_path=LINE_CONFIG['map_path'],
                    num_lines=NUM_LINES,
                    line_spacing=LINE_CONFIG['line_spacing'],
                )
            self.waypoints_lines = wp['lines']
            self._obs_dim = OBS_DIM
            self.get_logger().info(
                f'웨이포인트 로드 완료: {NUM_LINES}개 라인 | obs_dim={self._obs_dim}'
            )
        except Exception as e:
            self.get_logger().error(f'웨이포인트 로드 실패: {e}')
            self.waypoints_lines = None
            self._obs_dim = OBS_DIM_FALLBACK

    def _process_lidar(self, scans: list) -> np.ndarray:
        """
        브릿지에서 오는 scans[1080] → 정규화된 lidar[108]
        실차용 _process_lidar()와 동일한 전처리 로직
        """
        ranges = np.array(scans, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX)
        ranges = np.clip(ranges, LIDAR_MIN, LIDAR_MAX)
        ranges = (ranges - LIDAR_MIN) / (LIDAR_MAX - LIDAR_MIN)
        # 1080 → 108 다운샘플링
        step   = max(1, len(ranges) // LIDAR_SIZE)
        ranges = ranges[::step][:LIDAR_SIZE]
        if len(ranges) < LIDAR_SIZE:
            ranges = np.pad(
                ranges, (0, LIDAR_SIZE - len(ranges)), constant_values=1.0
            )
        return ranges

    def recv_callback(self, msg: Recv):
        """
        f110_recv 콜백 — 메인 파이프라인
        실차용에서 lidar_callback + odom_callback 역할을 하나로 통합
        """
        # [STEP 1] LiDAR 전처리 (scans → lidar)
        lidar = self._process_lidar(msg.obs.scans)

        # [STEP 2] odom 추출 (Odometry 토픽 대신 Recv.obs에서 직접 추출)
        position_snapshot = np.array([msg.obs.poses_x, msg.obs.poses_y], dtype=np.float32)
        heading_snapshot  = float(msg.obs.poses_theta)
        speed_snapshot    = float(msg.obs.linear_vels_x)
        self.odom_received = True

        if self.waypoints_lines is None:
            return

        # [STEP 3] 곡률 계산
        line_curvatures = None
        if USE_CURVATURE:
            line_curvatures = _compute_line_lookahead_curvatures(
                self.waypoints_lines, position_snapshot, speed_snapshot
            )

        # [STEP 4] 관측 벡터 생성
        obs = build_observation(
            lidar,
            position_snapshot,
            heading_snapshot,
            speed_snapshot,
            self.waypoints_lines,
            NUM_LINES,
            line_curvatures=line_curvatures,
            use_line_curvature=USE_CURVATURE,
        )

        if len(obs) != self._obs_dim:
            self.get_logger().warn(
                f'obs 크기 불일치: {len(obs)} != {self._obs_dim}'
            )
            return

        # [STEP 5] SAC 추론
        action    = self.model.select_action(obs, training=False)
        line_idx  = self.model.action_to_line_index(action)
        waypoints = self.waypoints_lines[line_idx]

        # [STEP 6] Pure Pursuit 조향/속도 계산
        steering, pp_speed = self.controller.compute(
            position_snapshot[0], position_snapshot[1],
            heading_snapshot, speed_snapshot, waypoints,
        )
        final_speed = min(
            self.model.action_to_speed(action, SPEED_MIN, SPEED_MAX),
            pp_speed,
        )

        # [STEP 7] f110_send 발행 (AckermannDrive 대신 Act 타입)
        self._publish_act(steering, final_speed)

    def _publish_act(self, steering: float, speed: float):
        steering = float(np.clip(steering, -MAX_STEERING, MAX_STEERING))
        speed    = float(np.clip(speed,    REAL_SPEED_MIN, REAL_SPEED_MAX))

        act_msg = Act()
        act_msg.steer = steering
        act_msg.speed = speed
        self.act_pub.publish(act_msg)

        self.get_logger().debug(
            f'act → 조향각: {steering:.3f} rad, 속도: {speed:.3f} m/s'
        )


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNodeSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
