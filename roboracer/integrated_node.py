import os
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import torch

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG, PURE_PURSUIT_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH,
)
from sac_model import SAC, get_obs_dim, build_observation
from waypoint_loader import load_waypoints
from pure_pursuit import PurePursuitController

# ── 실차 안전 속도 상수 (control_node.py 기준) ────────────────────────────────
REAL_SPEED_MAX = 3.0    # m/s
REAL_SPEED_MIN = 0.5    # m/s
MAX_STEERING   = 0.4189 # rad (~24도)

# ── LiDAR / 관측 상수 ────────────────────────────────────────────────────────
NUM_LINES        = LINE_CONFIG['num_lines']
LIDAR_SIZE       = OBS_CONFIG['lidar_size']
LIDAR_MIN        = OBS_CONFIG['lidar_range_min']
LIDAR_MAX        = OBS_CONFIG['lidar_range_max']
USE_CURVATURE    = OBS_CONFIG.get('use_line_curvature', False)
CURVATURE_MODE   = OBS_CONFIG.get('curvature_mode', 'max')
CURVATURE_MAX    = OBS_CONFIG.get('curvature_max_value', 1.5)
OBS_DIM          = get_obs_dim(LIDAR_SIZE, NUM_LINES, use_line_curvature=USE_CURVATURE)
OBS_DIM_FALLBACK = LIDAR_SIZE  # 웨이포인트 없을 때 fallback


# ── 곡률 계산 함수 (train_node.py / eval_node.py 와 동일한 로직) ──────────────
def _compute_three_point_curvature(p0: np.ndarray,
                                   p1: np.ndarray,
                                   p2: np.ndarray) -> float:
    """세 점으로 곡률 계산 (외접원 반지름의 역수)"""
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
    """
    각 라인의 전방 lookahead 구간 곡률을 계산 (train_node.py 로직 이식)
    반환: shape (NUM_LINES,), 0~1 정규화된 곡률값
    """
    # lookahead window 결정 (Pure Pursuit 속도 감속용 window 재사용)
    if OBS_CONFIG.get('curvature_use_pp_window', True):
        base   = int(PURE_PURSUIT_CONFIG.get('lookahead_window_base', 5))
        scale  = int(PURE_PURSUIT_CONFIG.get('lookahead_window_speed_scale', 2))
        window = base + int(abs(speed) * scale)
        sample_step = int(PURE_PURSUIT_CONFIG.get('curvature_sample_step', 2))
    else:
        window      = int(OBS_CONFIG.get('curvature_lookahead_window', 30))
        sample_step = int(OBS_CONFIG.get('curvature_sample_step', 2))

    max_curv = max(float(CURVATURE_MAX), 1e-6)
    line_curvatures = []

    for waypoints in waypoints_lines:
        # 현재 위치에서 가장 가까운 waypoint 인덱스 찾기
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


class IntegratedNode(Node):
    """
    Perception + Decision + Control 통합 노드

    파이프라인 (lidar_callback 내부):
      /scan  ──► _process_lidar()
                      │
                      ▼
               _compute_line_lookahead_curvatures()   ← [추가] 학습과 동일한 곡률 계산
                      │
                      ▼
               build_observation()
                      │
                      ▼
               SAC.select_action()
               action_to_line_index()
               PurePursuit.compute()
               action_to_speed()
                      │
                      ▼
               _publish_drive()
                      │
                      ▼
                   /drive (AckermannDriveStamped)
    """

    def __init__(self):
        super().__init__('integrated_node')

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

        self.position      = np.array([0.0, 0.0])
        self.heading       = 0.0
        self.speed         = 0.0
        self.odom_received = False

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        self.get_logger().info(
            f'integrated_node started | obs_dim={self._obs_dim} '
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

    def odom_callback(self, msg: Odometry):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.heading = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )
        self.speed         = msg.twist.twist.linear.x
        self.odom_received = True

    def _process_lidar(self, msg: LaserScan) -> np.ndarray:
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX)
        ranges = np.clip(ranges, LIDAR_MIN, LIDAR_MAX)
        ranges = (ranges - LIDAR_MIN) / (LIDAR_MAX - LIDAR_MIN)
        step   = max(1, len(ranges) // LIDAR_SIZE)
        ranges = ranges[::step][:LIDAR_SIZE]
        if len(ranges) < LIDAR_SIZE:
            ranges = np.pad(
                ranges, (0, LIDAR_SIZE - len(ranges)), constant_values=1.0
            )
        return ranges

    def lidar_callback(self, msg: LaserScan):
        # [STEP 1] LiDAR 전처리
        lidar = self._process_lidar(msg)

        if not self.odom_received or self.waypoints_lines is None:
            return

        # 레이스 컨디션 방지용 스냅샷
        position_snapshot = self.position.copy()
        heading_snapshot  = float(self.heading)
        speed_snapshot    = float(self.speed)

        # [STEP 2] 곡률 계산 (학습 때와 동일한 방식)
        # USE_CURVATURE=False면 None → build_observation 내부에서 zeros 처리
        line_curvatures = None
        if USE_CURVATURE:
            line_curvatures = _compute_line_lookahead_curvatures(
                self.waypoints_lines, position_snapshot, speed_snapshot
            )

        # [STEP 3] 관측 벡터 생성
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

        # [STEP 4] SAC 추론
        action    = self.model.select_action(obs, training=False)
        line_idx  = self.model.action_to_line_index(action)
        waypoints = self.waypoints_lines[line_idx]

        # [STEP 5] Pure Pursuit 조향/속도 계산
        steering, pp_speed = self.controller.compute(
            position_snapshot[0], position_snapshot[1],
            heading_snapshot, speed_snapshot, waypoints,
        )
        final_speed = min(
            self.model.action_to_speed(action, SPEED_MIN, SPEED_MAX),
            pp_speed,
        )

        # [STEP 6] /drive 발행
        self._publish_drive(steering, final_speed)

    def _publish_drive(self, steering: float, speed: float):
        steering = float(np.clip(steering, -MAX_STEERING,   MAX_STEERING))
        speed    = float(np.clip(speed,     REAL_SPEED_MIN, REAL_SPEED_MAX))

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp         = self.get_clock().now().to_msg()
        drive_msg.header.frame_id      = 'base_link'
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed          = speed
        self.drive_pub.publish(drive_msg)

        self.get_logger().debug(
            f'drive → 조향각: {steering:.3f} rad, 속도: {speed:.3f} m/s'
        )


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
