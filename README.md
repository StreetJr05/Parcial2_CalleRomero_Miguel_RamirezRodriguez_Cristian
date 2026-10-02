# Parcial 2 – Robótica: Cinemática y Planeación de Movimiento con ROS2 / MoveIt2 (FANUC LR Mate 200iD)

Workspace de ROS 2 para la simulación, planificación y ejecución de un ciclo pick-and-place con un FANUC LR Mate 200iD. Incluye el paquete de descripción del robot, la configuración propia de MoveIt 2, la escena de colisión y los scripts de Python para gestión de poses, comparación de planificadores e interpolación (cúbica y quíntica).

**Autores:** Miguel Calle Romero, Cristian Ramírez Rodríguez
**Curso:** Robótica — Universidad EIA

## Paquetes

| Paquete | Contenido |
|---|---|
| `lrmate200id_support` | URDF/Xacro + mallas (`tool0` = cara de la brida, Z saliendo de la brida) |
| `lrmate200id_moveit_config` | Configuración propia de MoveIt 2 creada con el Setup Assistant (grupo `manipulator`, solver KDL, OMPL en `config/ompl_planning.yaml`) |
| `pose_manager_pkg` | `cell_layout.py` (todas las dimensiones de la escena y poses), `moveit_client.py` (cliente rclpy para `move_group`), nodo `manage_poses` (Partes 2–3) |
| `scene_builder_pkg` | Nodo `spawn_scene` (PlanningScene: pieza, mesa de pick, poste, mesa de depósito) |
| `planner_execution_pkg` | `compare_planners` (4A/4C), `fine_approach` (4B/4D, interpolación cúbica vs quíntica), `pick_place_cycle` (ciclo completo) |

## Estructura del repositorio

```
ws_lrmate200id/
├── README.md
├── .gitignore
└── src/
    ├── lrmate200id_support/
    ├── lrmate200id_moveit_config/
    ├── pose_manager_pkg/
    ├── scene_builder_pkg/
    └── planner_execution_pkg/
```

Las carpetas `build/`, `install/` y `log/` no se incluyen; se generan al compilar.

## Requisitos

- Ubuntu 24.04
- ROS 2 Jazzy (`ros-jazzy-desktop`)
- MoveIt 2: `sudo apt install ros-jazzy-moveit`
- Herramientas de compilación:
  ```bash
  sudo apt install python3-colcon-common-extensions python3-rosdep git
  ```
- Librerías de Python para gráficas y CSV (si `rosdep` no las instala):
  ```bash
  sudo apt install python3-numpy python3-matplotlib
  ```

## Compilación

```bash
# 1. Clonar el repositorio como workspace
git clone https://github.com/StreetJr05/Parcial2_CalleRomero_Miguel_RamirezRodriguez_Cristian.git ~/ws_lrmate200id
cd ~/ws_lrmate200id

# 2. Cargar ROS 2 e instalar dependencias
source /opt/ros/jazzy/setup.bash
sudo rosdep init      # solo la primera vez en el equipo
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 3. Compilar
colcon build --symlink-install

# 4. Cargar el entorno del workspace
source install/setup.bash
```

## Ejecución

Cada comando va en su propia terminal, después de ejecutar:

```bash
cd ~/ws_lrmate200id
source install/setup.bash
```

```bash
# 0. MoveIt + RViz (dejar abierto durante toda la sesión)
ros2 launch lrmate200id_moveit_config demo.launch.py

# 1. Escena de colisión
ros2 run scene_builder_pkg spawn_scene

# 2. Partes 2-3: transformación de HOME + cinemática inversa de pick/place
#    (escribe ~/tap02_results/key_joints.yaml)
ros2 run pose_manager_pkg manage_poses

# 3. Parte 4A: comparación de planificadores (solo planifica, 10 intentos por planificador)
ros2 run planner_execution_pkg compare_planners --ros-args -p segment:=4A -p trials:=10

# 4. Parte 4B: aproximación fina cúbica vs quíntica en la estación de pick (mueve el robot)
ros2 run planner_execution_pkg fine_approach --ros-args -p station:=pick

# 5. Parte 4C: comparación de planificadores en el segmento hacia la estación de depósito
ros2 run planner_execution_pkg compare_planners --ros-args -p segment:=4C -p trials:=10

# 6. Parte 4D: aproximación fina cúbica vs quíntica en la estación de depósito
ros2 run planner_execution_pkg fine_approach --ros-args -p station:=place

# 7. Ciclo completo 4A -> 4B -> 4C -> 4D (usado para el video)
ros2 run planner_execution_pkg pick_place_cycle --ros-args -p planner:=RRTConnectkConfigDefault
```

El orden importa: `spawn_scene` debe correr antes de planificar para que MoveIt tenga en cuenta los obstáculos, y `manage_poses` debe correr antes que los nodos de la Parte 4, porque estos leen `key_joints.yaml`.

## Resultados

Las gráficas (perfiles de posición, velocidad y aceleración) y los archivos CSV (tiempos de planificación, longitud de trayectoria, tasa de éxito) se guardan en:

```
~/tap02_results/
```

## Solución de problemas

- **`Package '...' not found`**: falta `source install/setup.bash` en esa terminal.
- **Los cambios en los scripts de Python no se reflejan**: compilar con `--symlink-install`, o recompilar solo el paquete modificado con `colcon build --packages-select <paquete>`.
- **La planificación atraviesa los obstáculos**: verificar que `spawn_scene` se ejecutó con `demo.launch.py` ya abierto, y que los objetos aparecen en RViz.
- **`key_joints.yaml` no existe**: ejecutar primero `manage_poses`.
- **El planificador no se reconoce**: usar el nombre exacto definido en `config/ompl_planning.yaml` (por ejemplo, `RRTConnectkConfigDefault`).
