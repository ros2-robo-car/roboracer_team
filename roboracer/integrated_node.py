import os
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import torch

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    OBS_CONFIG, LINE_CONFIG, MODEL_CONFIG, PURE_PURSUIT_CONFIG,
    SPEED_MIN, SPEED_MAX, MODEL_SAVE_PATH,
)
from sac_model import SAC, get_obs_dim, build_observation
from waypoint_loader import load_waypoints
from pure_pursuit import PurePursuitController

# ── 실차 안전 속도 상수 ────────────────────────────────────────────────────────
REAL_SPEED_MAX = 3.0
REAL_SPEED_MIN = 0.5
MAX_STEERING   = 0.4189

# ── 타임아웃 상수 ────────────────────────────────────────────────────────────
ODOM_TIMEOUT   = 0.2
SCAN_TIMEOUT   = 0.2
POSE_TIMEOUT   = 1.0    # /amcl_pose 타임아웃 (amcl은 느리므로 여유있게)

# ── LiDAR / 관측 상수 ────────────────────────────────────────────────────────
NUM_LINES        = LINE_CONFIG['num_lines']
LIDAR_SIZE       = OBS_CONFIG['lidar_size']
LIDAR_MIN        = OBS_CONFIG['lidar_range_min']
LIDAR_MAX        = OBS_CONFIG['lidar_range_max']
USE_CURVATURE    = OBS_CONFIG.get('use_line_curvature', False)
CURVATURE_MODE   = OBS_CONFIG.get('curvature_mode', 'max')
CURVATURE_MAX    = OBS_CONFIG.get('curvature_max_value', 1.5)
OBS_DIM          = get_obs_dim(LIDAR_SIZE, NUM_LINES, use_line_curvature=USE_CURVATURE)
OBS_DIM_FALLBACK = LIDAR_SIZE


def _compute_three_point_curvature(p0, p1, p2):
    d01 = np.linalg.norm(p1 - p0)
    d12 = np.linalg.norm(p2 - p1)
    d02 = np.linalg.norm(p2 - p0)
    denom = d01 * d12 * d02
    if denom < 1e-9:
        return 0.0
    cross = abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]))
    curvature = 2.0 * cross / denom
    return float(curvature) if np.isfinite(curvature) else 0.0


def _compute_line_lookahead_curvatures(waypoints_lines, position, speed):
    if OBS_CONFIG.get('curvature_use_pp_window', True):
        base        = int(PURE_PURSUIT_CONFIG.get('lookahead_window_base', 5))
        scale       = int(PURE_PURSUIT_CONFIG.get('lookahead_window_speed_scale', 2))
        window      = base + int(abs(speed) * scale)
        sample_step = int(PURE_PURSUIT_CONFIG.get('curvature_sample_step', 2))  # ← 오타 수정
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

        line_curvatures.append(float(np.clip(line_curvature / max_curv, 0.0, 1.0)))

    return np.array(line_curvatures, dtype=np.float32)


class IntegratedNode(Node):
    """
    Perception + Decision + Control 통합 노드

    멘토님 피드백 반영:
      - Localization: /amcl_pose (Particle Filter) 로 정확한 맵 좌표 위치 추정
      - /amcl_pose 미수신 시 /odom으로 fallback (테스트/개발 편의성)
      - 속도는 항상 /odom에서 가져옴 (VESC 엔코더 기반)
      - Mapping: 실차 주행 전 SLAM으로 맵 생성 후 centerline CSV 추출 필요
      - AMCL 초기 위치 자동 발행 (/initialpose)

    안전장치:
      - LiDAR 타임아웃: /scan 0.2초 이상 끊기면 긴급 정지
      - 오돔 타임아웃: /odom 0.2초 이상 끊기면 긴급 정지
      - Pose 타임아웃: /amcl_pose 수신 후 1.0초 이상 끊기면 긴급 정지
    """

    def __init__(self):
        super().__init__('integrated_node')

        # ── 상태 변수 ─────────────────────────────────────────────────────
        self.position      = np.array([0.0, 0.0])
        self.heading       = 0.0
        self.speed         = 0.0
        self.odom_received = False
        self.pose_received = False  # /amcl_pose 수신 여부

        # ── 타임아웃 감시용 시각 ──────────────────────────────────────────
        self.last_odom_time = self.get_clock().now()
        self.last_scan_time = self.get_clock().now()
        self.last_pose_time = self.get_clock().now()
        self.scan_received  = False

        # ── 웨이포인트 로드 (모델보다 먼저!) ──────────────────────────────
        self._load_waypoints()

        self.controller = PurePursuitController(
            max_speed=SPEED_MAX, min_speed=SPEED_MIN
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SAC(
            self._obs_dim,  # _load_waypoints 후에 설정된 값 사용
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

        # ── 타임아웃 감시 타이머 (0.1초마다 체크) ─────────────────────────
        self.timeout_timer = self.create_timer(0.1, self._check_timeouts)

        # ── AMCL 초기 위치 발행용 일회성 타이머 ───────────────────────────
        self.init_pose_timer = self.create_timer(1.5, self._publish_initial_pose)

        # ── 구독 / 발행 ───────────────────────────────────────────────────
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        # /amcl_pose: Particle Filter 기반 정확한 맵 좌표 위치
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10
        )
        self.init_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        self.get_logger().info(
            f'integrated_node started | obs_dim={self._obs_dim} '
            f'| use_curvature={USE_CURVATURE} | device={self.device}'
        )
        self.get_logger().info(
            '/amcl_pose 미수신 시 /odom으로 위치 fallback 동작'
        )

    def _publish_initial_pose(self):
        """
        AMCL 노드를 깨우기 위해 초기 위치를 단 한 번 전송
        실차에서는 실제 출발 좌표로 수정 필요
        """
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x    = 0.0
        msg.pose.pose.position.y    = 0.0
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        cov[0]  = 0.25   # x 오차
        cov[7]  = 0.25   # y 오차
        cov[35] = 0.06   # yaw 오차
        msg.pose.covariance = cov
        self.init_pose_pub.publish(msg)
        self.get_logger().info('AMCL 초기 위치(/initialpose) 자동 발행 완료!')
        self.init_pose_timer.destroy()  # 한 번만 실행

    def _check_timeouts(self):
        """LiDAR, odom, amcl_pose 타임아웃 감시"""
        now = self.get_clock().now()

        if self.scan_received:
            dt_scan = (now - self.last_scan_time).nanoseconds / 1e9
            if dt_scan > SCAN_TIMEOUT:
                self.get_logger().error(
                    f'LiDAR 타임아웃! ({dt_scan:.3f}초) 차량을 정지합니다.'
                )
                self._publish_drive(0.0, 0.0, force_stop=True)

        if self.pose_received:
            dt_pose = (now - self.last_pose_time).nanoseconds / 1e9
            if dt_pose > POSE_TIMEOUT:
                self.get_logger().error(
                    f'Localization 타임아웃! ({dt_pose:.3f}초) 차량을 정지합니다.'
                )
                self._publish_drive(0.0, 0.0, force_stop=True)

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

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        /amcl_pose 콜백 — Particle Filter 기반 맵 좌표계 위치/방향
        SLAM 맵 위에서 현재 차량의 정확한 위치를 제공
        """
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.heading = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )
        self.pose_received  = True
        self.last_pose_time = self.get_clock().now()

    def odom_callback(self, msg: Odometry):
        """
        /odom 콜백
        - 속도는 항상 /odom에서 가져옴 (VESC 엔코더 기반, 실시간)
        - 위치/방향은 /amcl_pose 미수신 시에만 fallback으로 사용
        """
        self.speed         = msg.twist.twist.linear.x
        self.odom_received = True
        self.last_odom_time = self.get_clock().now()

        # /amcl_pose 미수신 시 odom으로 위치/방향 fallback
        if not self.pose_received:
            self.position[0] = msg.pose.pose.position.x
            self.position[1] = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self.heading = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
            )

    def _process_lidar(self, msg: LaserScan) -> np.ndarray:
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX)
        ranges = np.clip(ranges, LIDAR_MIN, LIDAR_MAX)
        ranges = (ranges - LIDAR_MIN) / (LIDAR_MAX - LIDAR_MIN)
        step   = max(1, len(ranges) // LIDAR_SIZE)
        ranges = ranges[::step][:LIDAR_SIZE]
        if len(ranges) < LIDAR_SIZE:
            ranges = np.pad(ranges, (0, LIDAR_SIZE - len(ranges)), constant_values=1.0)
        return ranges

    def lidar_callback(self, msg: LaserScan):
        self.last_scan_time = self.get_clock().now()
        self.scan_received  = True

        # 오돔 타임아웃 체크
        dt_odom = (self.get_clock().now() - self.last_odom_time).nanoseconds / 1e9
        if self.odom_received and dt_odom > ODOM_TIMEOUT:
            self.get_logger().error(
                f'오돔 타임아웃! ({dt_odom:.3f}초) 차량을 정지합니다.'
            )
            self._publish_drive(0.0, 0.0, force_stop=True)
            return

        # odom 미수신 시 대기 (/amcl_pose 미수신은 odom fallback으로 허용)
        if not self.odom_received or self.waypoints_lines is None:
            return

        # 스냅샷 (레이스 컨디션 방지)
        position_snapshot = self.position.copy()
        heading_snapshot  = float(self.heading)
        speed_snapshot    = float(self.speed)

        # [STEP 1] LiDAR 전처리
        lidar = self._process_lidar(msg)

        # [STEP 2] 곡률 계산
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

    def _publish_drive(self, steering: float, speed: float, force_stop: bool = False):
        steering = float(np.clip(steering, -MAX_STEERING, MAX_STEERING))

        if force_stop:
            speed = 0.0
        else:
            speed = float(np.clip(speed, REAL_SPEED_MIN, REAL_SPEED_MAX))

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
