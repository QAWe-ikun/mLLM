"""AI2-THOR环境Gym风格包装器"""
import numpy as np
from ai2thor.controller import Controller
from typing import Dict, Any, Optional, Tuple, List


class AI2THORWrapper:
    """AI2-THOR环境的Gym风格接口
    
    支持headless模式（适合WSL无GUI环境）
    """
    
    # 动作空间定义
    ACTION_SPACE = [
        'MoveAhead',      # 向前移动
        'RotateLeft',     # 向左旋转
        'RotateRight',    # 向右旋转
        'LookUp',         # 向上看
        'LookDown',       # 向下看
        'Pickup',         # 拾取物体
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """初始化环境
        
        Args:
            config: 环境配置字典
        """
        self.config = config
        ai2thor_config = config['environment']['ai2thor']
        
        self.grid_size = ai2thor_config['grid_size']
        self.rotate_step = ai2thor_config['rotate_step']
        self.camera_width = ai2thor_config['camera_width']
        self.camera_height = ai2thor_config['camera_height']
        self.visibility_distance = ai2thor_config['visibility_distance']
        
        # 奖励配置
        rewards_config = config['environment']['rewards']
        self.reward_success_nav = rewards_config['success_nav']
        self.reward_success_pickup = rewards_config['success_pickup']
        self.reward_step = rewards_config['step_penalty']
        self.reward_closer = rewards_config['closer_bonus']
        self.reward_farther = rewards_config['farther_penalty']
        self.reward_illegal = rewards_config['illegal_action']
        
        # Episode配置
        self.max_steps = config['environment']['episode']['max_steps']
        self.success_distance = config['environment']['episode']['success_distance']
        
        # 场景列表
        scenes = config['environment']['scenes']['train']
        self.train_scenes = []
        for room_type, scene_list in scenes.items():
            self.train_scenes.extend(scene_list)
        
        # 控制器（延迟初始化）
        self.controller: Optional[Controller] = None
        self.current_scene: Optional[str] = None
        self.current_step_count: int = 0
        self.target_object: Optional[str] = None
        self.target_position: Optional[Tuple[float, float, float]] = None
        
        # Agent状态
        self.agent_position: Optional[Tuple[float, float, float]] = None
        self.agent_rotation: Optional[float] = None
        
        # 距离历史（用于密集奖励）
        self.prev_distance: Optional[float] = None
        
        # 已拾取的物体
        self.held_object: Optional[str] = None
    
    def _init_controller(self, scene_name: str):
        """初始化AI2-THOR控制器"""
        if self.controller is not None:
            self.controller.stop()
        
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=self.visibility_distance,
            scene=scene_name,
            gridSize=self.grid_size,
            renderInstanceSegmentation=True,
            renderDepthImage=True,
            renderObjectImage=True,
            renderClassSegmentation=True,
            width=self.camera_width,
            height=self.camera_height,
            server_class='ThorUnityLocal',  # headless模式
        )
    
    def reset(self, scene_name: Optional[str] = None, 
              target_object: Optional[str] = None,
              seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """重置环境
        
        Args:
            scene_name: 指定场景，None则随机选择
            target_object: 指定目标物体，None则随机选择
            seed: 随机种子
            
        Returns:
            observation: 包含'image', 'position', 'rotation', 'target'的字典
        """
        # 随机选择场景
        if scene_name is None:
            scene_name = np.random.choice(self.train_scenes)
        
        self.current_scene = scene_name
        
        # 初始化控制器
        self._init_controller(scene_name)
        
        # 随机选择目标物体
        if target_object is None:
            task_config = self.config['environment']['tasks'][0]  # object_nav
            target_object = np.random.choice(task_config['target_objects'])
        
        self.target_object = target_object
        
        # 随机Teleport Agent位置
        event = self.controller.step(action="Teleport", position=dict(x=0, y=0.9, z=0))
        
        # 获取可行走位置并随机选择
        reachable_positions = self._get_reachable_positions()
        if reachable_positions:
            pos = np.random.choice(reachable_positions)
            event = self.controller.step(action="Teleport", position=pos)
        
        # 重置计数器
        self.current_step_count = 0
        self.held_object = None
        
        # 获取Agent状态
        self._update_agent_state()
        
        # 计算初始距离
        self.target_position = self._find_target_position()
        self.prev_distance = self._compute_distance_to_target()
        
        return self._get_observation()
    
    def step(self, action_idx: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """执行一步动作
        
        Args:
            action_idx: 动作索引 (0-5)
            
        Returns:
            observation: 新观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        action_name = self.ACTION_SPACE[action_idx]
        self.current_step_count += 1
        
        # 执行动作
        success = self._execute_action(action_name)
        
        # 更新Agent状态
        self._update_agent_state()
        
        # 计算距离和奖励
        current_distance = self._compute_distance_to_target()
        reward = self._compute_reward(current_distance, success, action_name)
        self.prev_distance = current_distance
        
        # 检查是否完成
        done = False
        info = {}
        
        if action_name == 'Pickup' and success:
            # 拾取成功
            self.held_object = self.target_object
            done = True
            info['success_pickup'] = True
        elif current_distance <= self.success_distance:
            # 到达目标
            info['success_nav'] = True
            # Pickup任务需要继续拾取
            if self.config['environment']['tasks'][1]['requires_nav']:
                # 下一步必须Pickup
                pass
            else:
                done = True
        
        # 检查最大步数
        max_steps = self.config['environment']['tasks'][0]['max_steps']
        if self.current_step_count >= max_steps:
            done = True
            info['timeout'] = True
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action_name: str) -> bool:
        """执行AI2-THOR动作"""
        if action_name == 'MoveAhead':
            event = self.controller.step(action="MoveAhead", moveMagnitude=self.grid_size)
        elif action_name == 'RotateLeft':
            event = self.controller.step(action="RotateLeft", degrees=self.rotate_step)
        elif action_name == 'RotateRight':
            event = self.controller.step(action="RotateRight", degrees=self.rotate_step)
        elif action_name == 'LookUp':
            event = self.controller.step(action="LookUp", degrees=self.rotate_step)
        elif action_name == 'LookDown':
            event = self.controller.step(action="LookDown", degrees=self.rotate_step)
        elif action_name == 'Pickup':
            event = self.controller.step(action="PickupObject", forceAction=True)
        else:
            raise ValueError(f"Unknown action: {action_name}")
        
        return event.metadata["lastActionSuccess"]
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前观测"""
        # RGB图像
        rgb_image = self.controller.last_event.frame
        rgb_array = np.array(rgb_image)
        
        # 深度图
        depth_array = np.array(self.controller.last_event.depth_frame)
        
        # 物体分割
        object_ids = np.array(self.controller.last_event.instance_segmentation_frame)
        
        return {
            'rgb': rgb_array.astype(np.float32) / 255.0,  # [H, W, 3], [0, 1]
            'depth': depth_array.astype(np.float32),       # [H, W]
            'object_ids': object_ids.astype(np.int32),     # [H, W]
            'position': np.array(self.agent_position, dtype=np.float32),  # [3]
            'rotation': np.array([self.agent_rotation], dtype=np.float32),  # [1]
            'target': self.target_object,
        }
    
    def _update_agent_state(self):
        """更新Agent位置和朝向"""
        metadata = self.controller.last_event.metadata
        agent_pos = metadata['agent']['position']
        self.agent_position = (agent_pos['x'], agent_pos['y'], agent_pos['z'])
        self.agent_rotation = metadata['agent']['rotation']['y']
    
    def _get_reachable_positions(self) -> List[Dict[str, float]]:
        """获取场景中所有可行走位置"""
        return self.controller.last_event.metadata['actionReturn']
    
    def _find_target_position(self) -> Optional[Tuple[float, float, float]]:
        """查找目标物体在场景中的位置"""
        objects = self.controller.last_event.metadata['objects']
        for obj in objects:
            if obj['objectType'] == self.target_object:
                pos = obj['position']
                return (pos['x'], pos['y'], pos['z'])
        return None
    
    def _compute_distance_to_target(self) -> float:
        """计算Agent到目标物体的欧氏距离"""
        if self.target_position is None:
            return float('inf')
        
        dx = self.agent_position[0] - self.target_position[0]
        dy = self.agent_position[1] - self.target_position[1]
        dz = self.agent_position[2] - self.target_position[2]
        
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def _compute_reward(self, current_distance: float, success: bool, action_name: str) -> float:
        """计算奖励
        
        奖励函数设计：
        - 到达目标: +10.0
        - 成功拾取: +5.0
        - 每步惩罚: -0.1
        - 靠近目标: +0.05/步
        - 远离目标: -0.05/步
        - 非法动作: -1.0
        """
        reward = 0.0
        
        # 每步惩罚
        reward += self.reward_step
        
        # 非法动作惩罚
        if not success:
            reward += self.reward_illegal
        
        # 距离奖励/惩罚
        if self.prev_distance is not None:
            distance_change = self.prev_distance - current_distance
            if distance_change > 0:
                reward += self.reward_closer  # 靠近
            else:
                reward += self.reward_farther  # 远离
        
        # 导航成功奖励
        if current_distance <= self.success_distance:
            reward += self.reward_success_nav
        
        return reward
    
    def is_target_visible(self) -> bool:
        """检查目标物体是否在当前视野中"""
        objects = self.controller.last_event.metadata['objects']
        for obj in objects:
            if obj['objectType'] == self.target_object and obj['visible']:
                return True
        return False
    
    def get_action_space_size(self) -> int:
        """返回动作空间大小"""
        return len(self.ACTION_SPACE)
    
    def get_scene_name(self) -> str:
        """返回当前场景名称"""
        return self.current_scene
    
    def close(self):
        """关闭环境"""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
    
    def __del__(self):
        self.close()
