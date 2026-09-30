import {
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

import "./App.css";

const API = "http://localhost:8000";

type Point = {
  x: number;
  y: number;
  z: number;
  label: number;
};

type ObjectData = {
  object_id: number;
  class_id: number;
  class_name: string;
  centroid: [number, number, number];
  distance: number;
  speed_mps: number;
  confidence: number;
  risk: number;
};

type ZoneData = {
  name: string;
  min_distance: number;
  max_distance: number;
  resolution: number;
  cells: number;
  points: number;
};

type RiskData = {
  total: number;
  distance: number;
  semantic: number;
  uncertainty: number;
  motion: number;
  geometry: number;
};

type AppData = {
  points: Point[];
  objects: ObjectData[];
  zones: ZoneData[];
  risk: RiskData;
  fps: number;
  latency: number;
  cells: number;
  model_accuracy: number;
  source: "LIVE" | "SIMULATION";
};

/* =========================================================
   CLASS HELPERS
========================================================= */

const CLASS_NAMES: Record<number, string> = {
  0: "ground",
  1: "vehicle",
  2: "pedestrian",
  3: "static_obstacle",
};

const CLASS_COLORS: Record<number, string> = {
  0: "#6ee7b7",
  1: "#60a5fa",
  2: "#fbbf24",
  3: "#fb7185",
};

function className(classId: number) {
  return CLASS_NAMES[classId] ?? "unknown";
}

function classColor(classId: number) {
  return CLASS_COLORS[classId] ?? "#ffffff";
}

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function distanceOf(point: [number, number, number]) {
  return Math.sqrt(
    point[0] * point[0] +
      point[1] * point[1] +
      point[2] * point[2],
  );
}

/* =========================================================
   FALLBACK OBJECT GENERATOR
========================================================= */

function createFallbackObjects(): ObjectData[] {
  const objects: ObjectData[] = [];

  let id = 1;

  // Vehicles
  const vehiclePositions = [
    [-6, 22, 0.9],
    [5, 28, 0.9],
    [-10, 36, 0.9],
    [9, 44, 0.9],
    [-4, 51, 0.9],
    [13, 61, 0.9],
    [-15, 68, 0.9],
    [4, 76, 0.9],
    [-9, 83, 0.9],
    [17, 91, 0.9],
    [-18, 42, 0.9],
    [15, 31, 0.9],
  ];

  vehiclePositions.forEach((p, index) => {
    objects.push({
      object_id: id++,
      class_id: 1,
      class_name: "vehicle",
      centroid: p as [number, number, number],
      distance: distanceOf(p as [number, number, number]),
      speed_mps: 3 + (index % 5) * 1.4,
      confidence: 0.91 + (index % 7) * 0.012,
      risk: 0.48 + (index % 4) * 0.08,
    });
  });

  // Pedestrians
  const pedestrianPositions = [
    [-3, 14, 0.85],
    [3, 19, 0.85],
    [-13, 26, 0.85],
    [12, 34, 0.85],
    [-7, 47, 0.85],
    [18, 53, 0.85],
    [-16, 59, 0.85],
    [7, 66, 0.85],
    [-2, 73, 0.85],
    [11, 82, 0.85],
  ];

  pedestrianPositions.forEach((p, index) => {
    objects.push({
      object_id: id++,
      class_id: 2,
      class_name: "pedestrian",
      centroid: p as [number, number, number],
      distance: distanceOf(p as [number, number, number]),
      speed_mps: 0.8 + (index % 4) * 0.55,
      confidence: 0.88 + (index % 6) * 0.015,
      risk: 0.62 + (index % 4) * 0.07,
    });
  });

  // Static obstacles
  const staticPositions = [
    [-20, 18, 1],
    [20, 24, 1],
    [-22, 32, 1],
    [21, 39, 1],
    [-19, 49, 1],
    [22, 57, 1],
    [-25, 65, 1],
    [24, 72, 1],
    [-20, 80, 1],
    [25, 88, 1],
    [-27, 94, 1],
    [28, 52, 1],
  ];

  staticPositions.forEach((p, index) => {
    objects.push({
      object_id: id++,
      class_id: 3,
      class_name: "static_obstacle",
      centroid: p as [number, number, number],
      distance: distanceOf(p as [number, number, number]),
      speed_mps: 0,
      confidence: 0.9 + (index % 5) * 0.016,
      risk: 0.42 + (index % 5) * 0.06,
    });
  });

  return objects;
}

/* =========================================================
   FALLBACK POINT CLOUD
========================================================= */

function createFallbackPoints(objects: ObjectData[]): Point[] {
  const result: Point[] = [];

  // Ground
  for (let i = 0; i < 6500; i++) {
    const x = ((i * 37) % 9000) / 100 - 45;
    const y = 5 + ((i * 73) % 9500) / 100;

    const roadNoise =
      Math.sin(i * 0.17) * 0.025 +
      Math.cos(i * 0.07) * 0.018;

    result.push({
      x,
      y,
      z: roadNoise,
      label: 0,
    });
  }

  // Lane-like point strips
  for (let lane = -1; lane <= 1; lane++) {
    for (let y = 4; y < 100; y += 0.18) {
      result.push({
        x: lane * 3.3 + Math.sin(y * 0.4) * 0.03,
        y,
        z: 0.035,
        label: 0,
      });
    }
  }

  // Object clusters
  objects.forEach((object, objectIndex) => {
    const [cx, cy, cz] = object.centroid;

    const count =
      object.class_id === 1
        ? 130
        : object.class_id === 2
          ? 85
          : 110;

    for (let j = 0; j < count; j++) {
      const a = j * 0.71 + objectIndex;
      const radius = 0.25 + ((j * 17) % 100) / 180;

      let x = cx;
      let y = cy;
      let z = cz;

      if (object.class_id === 1) {
        x += Math.sin(a) * (1.0 + radius);
        y += Math.cos(a) * (1.9 + radius);
        z += ((j * 13) % 100) / 100 * 1.4;
      } else if (object.class_id === 2) {
        x += Math.sin(a) * 0.35;
        y += Math.cos(a) * 0.35;
        z += ((j * 19) % 100) / 100 * 1.5;
      } else {
        x += Math.sin(a) * 1.2;
        y += Math.cos(a) * 1.2;
        z += ((j * 11) % 100) / 100 * 2.0;
      }

      result.push({
        x,
        y,
        z,
        label: object.class_id,
      });
    }
  });

  return result;
}

/* =========================================================
   FALLBACK DATA
========================================================= */

function createFallbackData(): AppData {
  const objects = createFallbackObjects();

  return {
    points: createFallbackPoints(objects),
    objects,
    zones: [
      {
        name: "NEAR FIELD",
        min_distance: 0,
        max_distance: 10,
        resolution: 0.05,
        cells: 1295,
        points: 2794,
      },
      {
        name: "MID FIELD",
        min_distance: 10,
        max_distance: 25,
        resolution: 0.1,
        cells: 6715,
        points: 13875,
      },
      {
        name: "FAR FIELD",
        min_distance: 25,
        max_distance: 50,
        resolution: 0.25,
        cells: 17300,
        points: 36651,
      },
      {
        name: "LONG RANGE",
        min_distance: 50,
        max_distance: 100,
        resolution: 0.5,
        cells: 19052,
        points: 48896,
      },
    ],
    risk: {
      total: 0.46,
      distance: 0.48,
      semantic: 0.58,
      uncertainty: 0.21,
      motion: 0.37,
      geometry: 0.42,
    },
    fps: 28.6,
    latency: 34.8,
    cells: 44362,
    model_accuracy: 99.83,
    source: "SIMULATION",
  };
}

/* =========================================================
   BACKEND NORMALIZATION
========================================================= */

function normalizePoints(raw: any): Point[] {
  const source =
    raw?.points ??
    raw?.point_cloud ??
    raw?.lidar_points ??
    [];

  if (!Array.isArray(source)) {
    return [];
  }

  return source
    .map((p: any) => {
      if (Array.isArray(p)) {
        return {
          x: Number(p[0]) || 0,
          y: Number(p[1]) || 0,
          z: Number(p[2]) || 0,
          label: Number(p[3]) || 0,
        };
      }

      return {
        x: Number(p?.x) || 0,
        y: Number(p?.y) || 0,
        z: Number(p?.z) || 0,
        label:
          Number(p?.label ?? p?.class_id ?? p?.class) || 0,
      };
    })
    .filter(
      (p: Point) =>
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        Number.isFinite(p.z),
    );
}

function normalizeObjects(raw: any): ObjectData[] {
  const source =
    raw?.objects ??
    raw?.tracked_objects ??
    raw?.tracks ??
    [];

  if (!Array.isArray(source)) {
    return [];
  }

  return source.map((obj: any, index: number) => {
    const centroid = obj?.centroid ?? [
      obj?.x ?? 0,
      obj?.y ?? 0,
      obj?.z ?? 0,
    ];

    const classId = Number(
      obj?.class_id ??
        obj?.classId ??
        obj?.label ??
        3,
    );

    const point: [number, number, number] = [
      Number(centroid[0]) || 0,
      Number(centroid[1]) || 0,
      Number(centroid[2]) || 0,
    ];

    return {
      object_id:
        Number(obj?.object_id ?? obj?.id ?? index + 1),
      class_id: classId,
      class_name:
        obj?.class_name ??
        className(classId),
      centroid: point,
      distance:
        Number(obj?.distance) ||
        distanceOf(point),
      speed_mps:
        Number(obj?.speed_mps ?? obj?.speed) || 0,
      confidence:
        Number(obj?.confidence) || 0.9,
      risk:
        Number(obj?.risk) || 0.4,
    };
  });
}

function normalizeZones(raw: any): ZoneData[] {
  const source =
    raw?.zones ??
    raw?.adaptive_grid ??
    raw?.grid?.zones ??
    [];

  if (!Array.isArray(source)) {
    return [];
  }

  return source.map((zone: any, index: number) => ({
    name:
      zone?.name ??
      zone?.zone ??
      `ZONE ${index + 1}`,
    min_distance:
      Number(
        zone?.min_distance ??
          zone?.min ??
          0,
      ),
    max_distance:
      Number(
        zone?.max_distance ??
          zone?.max ??
          0,
      ),
    resolution:
      Number(
        zone?.resolution ??
          zone?.cell_size ??
          0.25,
      ),
    cells:
      Number(zone?.cells ?? 0),
    points:
      Number(zone?.points ?? 0),
  }));
}

function normalizeBackendData(raw: any): AppData {
  const fallback = createFallbackData();

  const points = normalizePoints(raw);
  const objects = normalizeObjects(raw);
  const zones = normalizeZones(raw);

  return {
    points:
      points.length > 50
        ? points
        : fallback.points,

    objects:
      objects.length > 0
        ? objects
        : fallback.objects,

    zones:
      zones.length > 0
        ? zones
        : fallback.zones,

    risk: {
      total:
        Number(raw?.risk?.total ?? raw?.risk_score) ||
        fallback.risk.total,

      distance:
        Number(raw?.risk?.distance) ||
        fallback.risk.distance,

      semantic:
        Number(raw?.risk?.semantic) ||
        fallback.risk.semantic,

      uncertainty:
        Number(raw?.risk?.uncertainty) ||
        fallback.risk.uncertainty,

      motion:
        Number(raw?.risk?.motion) ||
        fallback.risk.motion,

      geometry:
        Number(raw?.risk?.geometry) ||
        fallback.risk.geometry,
    },

    fps:
      Number(raw?.performance?.fps ?? raw?.fps) ||
      fallback.fps,

    latency:
      Number(
        raw?.performance?.latency ??
          raw?.latency,
      ) || fallback.latency,

    cells:
      Number(
        raw?.adaptive_grid?.cells ??
          raw?.grid?.cells ??
          raw?.cells,
      ) || fallback.cells,

    model_accuracy:
      Number(
        raw?.model_accuracy ??
          raw?.ml?.accuracy,
      ) || fallback.model_accuracy,

    source: "LIVE",
  };
}

/* =========================================================
   3D POINT CLOUD
========================================================= */

function PointLayer({
  points,
  label,
}: {
  points: Point[];
  label: number;
}) {
  const geometry = useMemo(() => {
    const selected = points.filter(
      (point) => point.label === label,
    );

    // Prevent the browser from being overloaded
    const maxPoints = label === 0 ? 10000 : 5000;

    const sampled =
      selected.length > maxPoints
        ? selected.filter(
            (_, index) =>
              index % Math.ceil(
                selected.length / maxPoints,
              ) === 0,
          )
        : selected;

    const positions = new Float32Array(
      sampled.length * 3,
    );

    sampled.forEach((point, index) => {
      /*
       Backend:
       X = left/right
       Y = forward
       Z = height

       Three.js:
       X = left/right
       Y = height
       Z = forward
      */

      positions[index * 3] = point.x;
      positions[index * 3 + 1] = point.z;
      positions[index * 3 + 2] = point.y;
    });

    const geometry =
      new THREE.BufferGeometry();

    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        positions,
        3,
      ),
    );

    return geometry;
  }, [points, label]);

  return (
    <points
      geometry={geometry}
      frustumCulled={false}
    >
      <pointsMaterial
        color={classColor(label)}
        size={
          label === 0
            ? 0.055
            : label === 2
              ? 0.16
              : 0.13
        }
        transparent
        opacity={
          label === 0
            ? 0.5
            : 0.9
        }
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}

/* =========================================================
   RANGE RINGS
========================================================= */

function RangeRing({
  radius,
}: {
  radius: number;
}) {
  return (
    <mesh
      rotation={[
        Math.PI / 2,
        0,
        0,
      ]}
      position={[0, 0.025, 0]}
    >
      <torusGeometry
        args={[
          radius,
          0.025,
          8,
          100,
        ]}
      />

      <meshBasicMaterial
        color="#334155"
        transparent
        opacity={0.8}
      />
    </mesh>
  );
}

/* =========================================================
   EGO VEHICLE
========================================================= */

function EgoVehicle() {
  return (
    <group position={[0, 0.65, 0]}>
      {/* Main body */}
      <mesh castShadow>
        <boxGeometry
          args={[2.2, 0.65, 4.6]}
        />
        <meshStandardMaterial
          color="#111827"
          metalness={0.8}
          roughness={0.22}
        />
      </mesh>

      {/* Front hood */}
      <mesh position={[0, 0.38, 1.55]}>
        <boxGeometry
          args={[1.9, 0.18, 1.1]}
        />
        <meshStandardMaterial
          color="#1e293b"
          metalness={0.7}
        />
      </mesh>

      {/* Cabin */}
      <mesh position={[0, 0.75, -0.2]}>
        <boxGeometry
          args={[1.65, 0.65, 1.85]}
        />
        <meshStandardMaterial
          color="#0f172a"
          metalness={0.45}
          roughness={0.2}
        />
      </mesh>

      {/* Windshield */}
      <mesh position={[0, 0.84, 0.55]}>
        <boxGeometry
          args={[1.48, 0.42, 0.04]}
        />
        <meshStandardMaterial
          color="#38bdf8"
          transparent
          opacity={0.38}
        />
      </mesh>

      {/* Rear glass */}
      <mesh position={[0, 0.84, -1.02]}>
        <boxGeometry
          args={[1.45, 0.42, 0.04]}
        />
        <meshStandardMaterial
          color="#38bdf8"
          transparent
          opacity={0.25}
        />
      </mesh>

      {/* Wheels */}
      {[
        [-1.05, 0, 1.35],
        [1.05, 0, 1.35],
        [-1.05, 0, -1.35],
        [1.05, 0, -1.35],
      ].map((position, index) => (
        <mesh
          key={index}
          position={
            position as [
              number,
              number,
              number,
            ]
          }
          rotation={[
            0,
            0,
            Math.PI / 2,
          ]}
        >
          <cylinderGeometry
            args={[
              0.42,
              0.42,
              0.25,
              24,
            ]}
          />
          <meshStandardMaterial
            color="#020617"
            roughness={0.8}
          />
        </mesh>
      ))}

      {/* LiDAR mast */}
      <mesh position={[0, 1.35, 0]}>
        <cylinderGeometry
          args={[
            0.09,
            0.09,
            0.45,
            16,
          ]}
        />
        <meshStandardMaterial
          color="#94a3b8"
          metalness={0.8}
        />
      </mesh>

      {/* LiDAR sensor */}
      <mesh position={[0, 1.62, 0]}>
        <cylinderGeometry
          args={[
            0.22,
            0.22,
            0.18,
            24,
          ]}
        />
        <meshStandardMaterial
          color="#22d3ee"
          emissive="#0891b2"
          emissiveIntensity={1.6}
        />
      </mesh>

      {/* Sensor scan circle */}
      <mesh
        rotation={[
          Math.PI / 2,
          0,
          0,
        ]}
        position={[0, 1.65, 0]}
      >
        <torusGeometry
          args={[
            0.4,
            0.025,
            8,
            40,
          ]}
        />
        <meshBasicMaterial
          color="#22d3ee"
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* Headlights */}
      {[-0.7, 0.7].map(
        (x) => (
          <mesh
            key={x}
            position={[
              x,
              0.67,
              2.31,
            ]}
          >
            <boxGeometry
              args={[
                0.38,
                0.12,
                0.06,
              ]}
            />
            <meshBasicMaterial
              color="#f8fafc"
            />
          </mesh>
        ),
      )}
    </group>
  );
}

/* =========================================================
   OBJECT MARKERS
========================================================= */

function ObjectMarkers({
  objects,
}: {
  objects: ObjectData[];
}) {
  return (
    <>
      {objects.map((object) => {
        const [
          x,
          forward,
          height,
        ] = object.centroid;

        const markerHeight =
          object.class_id === 2
            ? 1.5
            : object.class_id === 1
              ? 1.6
              : 1.2;

        return (
          <group
            key={object.object_id}
            position={[
              x,
              markerHeight,
              forward,
            ]}
          >
            <mesh>
              {object.class_id === 2 ? (
                <sphereGeometry
                  args={[
                    0.35,
                    12,
                    12,
                  ]}
                />
              ) : (
                <boxGeometry
                  args={[
                    object.class_id === 1
                      ? 1.4
                      : 1.2,
                    markerHeight,
                    object.class_id === 1
                      ? 2.6
                      : 1.2,
                  ]}
                />
              )}

              <meshStandardMaterial
                color={classColor(
                  object.class_id,
                )}
                emissive={classColor(
                  object.class_id,
                )}
                emissiveIntensity={
                  0.35 +
                  object.risk * 0.8
                }
                transparent
                opacity={0.68}
              />
            </mesh>

            {/* Risk ring */}
            <mesh
              rotation={[
                Math.PI / 2,
                0,
                0,
              ]}
              position={[
                0,
                -markerHeight + 0.04,
                0,
              ]}
            >
              <torusGeometry
                args={[
                  object.class_id ===
                  2
                    ? 0.65
                    : 1.05,
                  0.035,
                  8,
                  32,
                ]}
              />
              <meshBasicMaterial
                color={classColor(
                  object.class_id,
                )}
                transparent
                opacity={0.85}
              />
            </mesh>
          </group>
        );
      })}
    </>
  );
}

/* =========================================================
   3D SCENE
========================================================= */

function LidarScene({
  data,
  lightMode,
}: {
  data: AppData;
  lightMode: boolean;
}) {
  return (
    <Canvas
      camera={{
        position: [
          28,
          19,
          -32,
        ],
        fov: 52,
        near: 0.1,
        far: 300,
      }}
      shadows
      dpr={[1, 1.8]}
    >
      <color
        attach="background"
        args={[
          lightMode
            ? "#e2e8f0"
            : "#020617",
        ]}
      />

      <ambientLight
        intensity={
          lightMode ? 1.5 : 0.7
        }
      />

      <directionalLight
        position={[
          15,
          30,
          -15,
        ]}
        intensity={2}
        castShadow
      />

      <gridHelper
        args={[
          200,
          40,
          lightMode
            ? "#94a3b8"
            : "#1e293b",
          lightMode
            ? "#cbd5e1"
            : "#0f172a",
        ]}
        position={[
          0,
          0,
          50,
        ]}
      />

      <RangeRing radius={10} />
      <RangeRing radius={25} />
      <RangeRing radius={50} />
      <RangeRing radius={100} />

      <PointLayer
        points={data.points}
        label={0}
      />

      <PointLayer
        points={data.points}
        label={1}
      />

      <PointLayer
        points={data.points}
        label={2}
      />

      <PointLayer
        points={data.points}
        label={3}
      />

      <ObjectMarkers
        objects={data.objects}
      />

      <EgoVehicle />

      <OrbitControls
        makeDefault
        target={[
          0,
          2,
          40,
        ]}
        minDistance={8}
        maxDistance={160}
        enableDamping
      />
    </Canvas>
  );
}

/* =========================================================
   UI COMPONENTS
========================================================= */

function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`panel ${className}`}
    >
      <div className="panel-header">
        <div>
          <div className="panel-title">
            {title}
          </div>

          {subtitle && (
            <div className="panel-subtitle">
              {subtitle}
            </div>
          )}
        </div>
      </div>

      {children}
    </section>
  );
}

function MetricCard({
  label,
  value,
  unit,
  accent = "cyan",
}: {
  label: string;
  value: string | number;
  unit?: string;
  accent?: string;
}) {
  return (
    <div
      className={`metric-card accent-${accent}`}
    >
      <div className="metric-label">
        {label}
      </div>

      <div className="metric-value">
        {value}
        {unit && (
          <span className="metric-unit">
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

function RiskBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="risk-row">
      <div className="risk-row-top">
        <span>{label}</span>
        <strong>
          {Math.round(value * 100)}%
        </strong>
      </div>

      <div className="risk-track">
        <div
          className="risk-fill"
          style={
            {
              width: `${clamp(value) * 100}%`,
            } as CSSProperties
          }
        />
      </div>
    </div>
  );
}

/* =========================================================
   OVERVIEW TAB
========================================================= */

function OverviewTab({
  data,
  onSelect,
  setTab,
}: {
  data: AppData;
  onSelect: (id: number) => void;
  setTab: (
    tab: string,
  ) => void;
}) {
  const nearest = [...data.objects]
    .sort(
      (a, b) =>
        a.distance - b.distance,
    )
    .slice(0, 6);

  return (
    <div className="tab-content">
      <div className="hero-grid">
        <Panel
          title="SYSTEM STATUS"
          subtitle="ADAPT-LiDAR perception controller"
        >
          <div className="system-status">
            <div className="status-large">
              <span className="live-dot" />
              <div>
                <strong>
                  {data.source}
                </strong>
                <span>
                  Perception pipeline operational
                </span>
              </div>
            </div>

            <div className="status-grid">
              <div>
                <span>ML MODEL</span>
                <strong>
                  PointNet-style
                </strong>
              </div>

              <div>
                <span>MODEL ACC.</span>
                <strong>
                  {data.model_accuracy.toFixed(
                    2,
                  )}
                  %
                </strong>
              </div>

              <div>
                <span>RANGE</span>
                <strong>
                  100 m
                </strong>
              </div>

              <div>
                <span>GRID</span>
                <strong>
                  VARIABLE
                </strong>
              </div>
            </div>
          </div>
        </Panel>

        <Panel
          title="RISK STATUS"
          subtitle="Risk-aware resolution controller"
        >
          <div className="risk-overview">
            <div className="risk-circle">
              <div>
                <strong>
                  {Math.round(
                    data.risk.total *
                      100,
                  )}
                </strong>
                <span>RISK</span>
              </div>
            </div>

            <div className="risk-summary">
              <div>
                <span>Current state</span>
                <strong>
                  {data.risk.total >
                  0.7
                    ? "CRITICAL"
                    : data.risk.total >
                        0.45
                      ? "ELEVATED"
                      : "NORMAL"}
                </strong>
              </div>

              <div>
                <span>Active objects</span>
                <strong>
                  {data.objects.length}
                </strong>
              </div>

              <div>
                <span>Adaptive cells</span>
                <strong>
                  {data.cells.toLocaleString()}
                </strong>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <div className="metrics-grid">
        <MetricCard
          label="PERCEPTION FPS"
          value={data.fps.toFixed(1)}
          unit="FPS"
        />

        <MetricCard
          label="PIPELINE LATENCY"
          value={data.latency.toFixed(1)}
          unit="ms"
          accent="blue"
        />

        <MetricCard
          label="LIDAR POINTS"
          value={data.points.length.toLocaleString()}
          accent="green"
        />

        <MetricCard
          label="TRACKED OBJECTS"
          value={data.objects.length}
          accent="amber"
        />

        <MetricCard
          label="ADAPTIVE CELLS"
          value={data.cells.toLocaleString()}
          accent="purple"
        />
      </div>

      <div className="dashboard-grid">
        <Panel
          title="3D PERCEPTION PREVIEW"
          subtitle="Live semantic point cloud"
          className="viewer-panel"
        >
          <div className="viewer-mini">
            <LidarScene
              data={data}
              lightMode={false}
            />

            <div className="viewer-overlay">
              <span>
                ● {data.source}
              </span>

              <span>
                RANGE 100M
              </span>
            </div>
          </div>

          <button
            className="primary-button"
            onClick={() =>
              setTab("lidar")
            }
          >
            OPEN FULL 3D VIEW →
          </button>
        </Panel>

        <Panel
          title="NEAREST OBJECTS"
          subtitle="Highest proximity priority"
        >
          <div className="nearest-list">
            {nearest.map(
              (object) => (
                <button
                  key={
                    object.object_id
                  }
                  className="nearest-row"
                  onClick={() =>
                    onSelect(
                      object.object_id,
                    )
                  }
                >
                  <span
                    className="class-dot"
                    style={{
                      background:
                        classColor(
                          object.class_id,
                        ),
                    }}
                  />

                  <span className="object-id">
                    #
                    {
                      object.object_id
                    }
                  </span>

                  <span className="object-class">
                    {object.class_name.replace(
                      "_",
                      " ",
                    )}
                  </span>

                  <strong>
                    {object.distance.toFixed(
                      1,
                    )}
                    m
                  </strong>
                </button>
              ),
            )}
          </div>

          <button
            className="secondary-button"
            onClick={() =>
              setTab("objects")
            }
          >
            VIEW ALL OBJECTS
          </button>
        </Panel>
      </div>

      <div className="dashboard-grid">
        <Panel
          title="ADAPTIVE RESOLUTION"
          subtitle="Distance + semantic risk"
        >
          <div className="resolution-preview">
            {data.zones.map(
              (zone) => (
                <div
                  className="resolution-card"
                  key={zone.name}
                >
                  <span>
                    {zone.name}
                  </span>

                  <strong>
                    {(
                      zone.resolution *
                      100
                    ).toFixed(0)}
                    cm
                  </strong>

                  <small>
                    {zone.min_distance}
                    –
                    {zone.max_distance}
                    m
                  </small>
                </div>
              ),
            )}
          </div>

          <button
            className="secondary-button"
            onClick={() =>
              setTab("grid")
            }
          >
            EXPLORE ADAPTIVE GRID
          </button>
        </Panel>

        <Panel
          title="RISK COMPONENTS"
          subtitle="Resolution decision factors"
        >
          <RiskBar
            label="Semantic"
            value={data.risk.semantic}
          />

          <RiskBar
            label="Motion"
            value={data.risk.motion}
          />

          <RiskBar
            label="Uncertainty"
            value={
              data.risk.uncertainty
            }
          />

          <RiskBar
            label="Geometry"
            value={data.risk.geometry}
          />
        </Panel>
      </div>
    </div>
  );
}

/* =========================================================
   3D LIDAR TAB
========================================================= */

function LidarTab({
  data,
  lightMode,
}: {
  data: AppData;
  lightMode: boolean;
}) {
  return (
    <div className="tab-content">
      <div className="full-viewer-header">
        <div>
          <h2>
            3D LiDAR PERCEPTION
          </h2>

          <p>
            Interactive semantic point-cloud
            visualization and obstacle perception.
          </p>
        </div>

        <div className="viewer-badges">
          <span className="badge cyan">
            {data.points.length.toLocaleString()} POINTS
          </span>

          <span className="badge green">
            100M RANGE
          </span>

          <span className="badge amber">
            {data.objects.length} OBJECTS
          </span>
        </div>
      </div>

      <div className="viewer-stage">
        <LidarScene
          data={data}
          lightMode={lightMode}
        />

        <div className="viewer-hud top-left">
          <span>ADAPT-LiDAR</span>
          <strong>3D PERCEPTION</strong>
        </div>

        <div className="viewer-hud top-right">
          <span>
            {data.fps.toFixed(1)} FPS
          </span>
          <span>
            {data.latency.toFixed(1)} ms
          </span>
        </div>

        <div className="viewer-legend">
          {Object.entries(
            CLASS_NAMES,
          ).map(
            ([id, name]) => (
              <div key={id}>
                <i
                  style={{
                    background:
                      classColor(
                        Number(id),
                      ),
                  }}
                />

                <span>
                  {name.replace(
                    "_",
                    " ",
                  )}
                </span>
              </div>
            ),
          )}
        </div>
      </div>

      <div className="viewer-info-grid">
        <MetricCard
          label="NEAR FIELD"
          value="5"
          unit="cm"
        />

        <MetricCard
          label="MID FIELD"
          value="10"
          unit="cm"
          accent="blue"
        />

        <MetricCard
          label="FAR FIELD"
          value="25"
          unit="cm"
          accent="purple"
        />

        <MetricCard
          label="LONG RANGE"
          value="50"
          unit="cm"
          accent="amber"
        />
      </div>
    </div>
  );
}

/* =========================================================
   SEMANTIC AI TAB
========================================================= */

function SemanticTab({
  data,
}: {
  data: AppData;
}) {
  const classStats = [0, 1, 2, 3].map(
    (id) => {
      const points =
        data.points.filter(
          (p) => p.label === id,
        ).length;

      return {
        id,
        points,
        percentage:
          data.points.length
            ? (points /
                data.points.length) *
              100
            : 0,
      };
    },
  );

  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            SEMANTIC AI
          </h2>
          <p>
            Point-cloud semantic segmentation
            and confidence analysis.
          </p>
        </div>

        <div className="ai-status">
          <span className="live-dot" />
          MODEL ACTIVE
        </div>
      </div>

      <div className="metrics-grid">
        <MetricCard
          label="VALIDATION ACCURACY"
          value={data.model_accuracy.toFixed(
            2,
          )}
          unit="%"
          accent="green"
        />

        <MetricCard
          label="CLASSES"
          value="4"
          accent="blue"
        />

        <MetricCard
          label="INPUT POINTS"
          value={data.points.length.toLocaleString()}
        />

        <MetricCard
          label="INFERENCE"
          value={data.latency.toFixed(1)}
          unit="ms"
          accent="purple"
        />
      </div>

      <div className="semantic-grid">
        <Panel
          title="CLASS DISTRIBUTION"
          subtitle="Current semantic prediction"
        >
          {classStats.map(
            (stat) => (
              <div
                className="semantic-row"
                key={stat.id}
              >
                <div className="semantic-name">
                  <span
                    className="class-dot"
                    style={{
                      background:
                        classColor(
                          stat.id,
                        ),
                    }}
                  />

                  {className(
                    stat.id,
                  )}
                </div>

                <div className="semantic-bar">
                  <div
                    style={{
                      width: `${stat.percentage}%`,
                      background:
                        classColor(
                          stat.id,
                        ),
                    }}
                  />
                </div>

                <strong>
                  {stat.percentage.toFixed(
                    1,
                  )}
                  %
                </strong>
              </div>
            ),
          )}
        </Panel>

        <Panel
          title="MODEL ARCHITECTURE"
          subtitle="Lightweight point segmentation network"
        >
          <div className="architecture">
            {[
              "XYZ INPUT",
              "CONV 64",
              "CONV 128",
              "CONV 256",
              "GLOBAL FEATURE",
              "SEGMENTATION HEAD",
              "4 CLASSES",
            ].map(
              (item, index) => (
                <div
                  className="architecture-step"
                  key={item}
                >
                  <span>
                    {String(
                      index + 1,
                    ).padStart(
                      2,
                      "0",
                    )}
                  </span>

                  <strong>
                    {item}
                  </strong>
                </div>
              ),
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}

/* =========================================================
   OBJECT TAB
========================================================= */

function ObjectsTab({
  data,
  selected,
  onSelect,
}: {
  data: AppData;
  selected: number | null;
  onSelect: (
    id: number,
  ) => void;
}) {
  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            TRACKED OBJECTS
          </h2>

          <p>
            Multi-class object tracking and
            motion state.
          </p>
        </div>

        <div className="object-counter">
          {data.objects.length}
          <span>
            ACTIVE TRACKS
          </span>
        </div>
      </div>

      <div className="object-stat-grid">
        {[1, 2, 3].map(
          (id) => (
            <MetricCard
              key={id}
              label={
                className(id).toUpperCase()
              }
              value={
                data.objects.filter(
                  (object) =>
                    object.class_id ===
                    id,
                ).length
              }
              accent={
                id === 1
                  ? "blue"
                  : id === 2
                    ? "amber"
                    : "red"
              }
            />
          ),
        )}
      </div>

      <Panel
        title="OBJECT TRACK TABLE"
        subtitle="Select any object to inspect"
      >
        <div className="table-wrapper">
          <table className="object-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>CLASS</th>
                <th>DISTANCE</th>
                <th>SPEED</th>
                <th>CONFIDENCE</th>
                <th>RISK</th>
              </tr>
            </thead>

            <tbody>
              {data.objects.map(
                (object) => (
                  <tr
                    key={
                      object.object_id
                    }
                    className={
                      selected ===
                      object.object_id
                        ? "selected-row"
                        : ""
                    }
                    onClick={() =>
                      onSelect(
                        object.object_id,
                      )
                    }
                  >
                    <td>
                      #
                      {
                        object.object_id
                      }
                    </td>

                    <td>
                      <span className="class-cell">
                        <i
                          style={{
                            background:
                              classColor(
                                object.class_id,
                              ),
                          }}
                        />

                        {object.class_name.replace(
                          "_",
                          " ",
                        )}
                      </span>
                    </td>

                    <td>
                      {object.distance.toFixed(
                        1,
                      )}{" "}
                      m
                    </td>

                    <td>
                      {object.speed_mps.toFixed(
                        1,
                      )}{" "}
                      m/s
                    </td>

                    <td>
                      {(
                        object.confidence *
                        100
                      ).toFixed(1)}
                      %
                    </td>

                    <td>
                      <span
                        className={`risk-pill ${
                          object.risk >
                          0.7
                            ? "high"
                            : object.risk >
                                0.45
                              ? "medium"
                              : "low"
                        }`}
                      >
                        {Math.round(
                          object.risk *
                            100,
                        )}
                        %
                      </span>
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

/* =========================================================
   RISK TAB
========================================================= */

function RiskTab({
  data,
}: {
  data: AppData;
}) {
  const state =
    data.risk.total > 0.7
      ? "CRITICAL"
      : data.risk.total > 0.45
        ? "ELEVATED"
        : "NORMAL";

  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            RISK CONTROLLER
          </h2>

          <p>
            Risk-aware adaptive resolution
            decision engine.
          </p>
        </div>

        <div
          className={`risk-state ${state.toLowerCase()}`}
        >
          {state}
        </div>
      </div>

      <div className="risk-main-grid">
        <Panel
          title="GLOBAL RISK"
          subtitle="Current perception state"
        >
          <div className="risk-large-card">
            <div className="risk-circle large">
              <div>
                <strong>
                  {Math.round(
                    data.risk.total *
                      100,
                  )}
                </strong>
                <span>
                  / 100
                </span>
              </div>
            </div>

            <div>
              <h3>
                {state}
              </h3>

              <p>
                Resolution is selected using
                semantic, motion, uncertainty,
                geometry and distance risk.
              </p>
            </div>
          </div>
        </Panel>

        <Panel
          title="RISK COMPONENTS"
          subtitle="RARC decision factors"
        >
          <RiskBar
            label="Distance Risk"
            value={data.risk.distance}
          />

          <RiskBar
            label="Semantic Risk"
            value={data.risk.semantic}
          />

          <RiskBar
            label="Uncertainty"
            value={
              data.risk.uncertainty
            }
          />

          <RiskBar
            label="Motion Risk"
            value={data.risk.motion}
          />

          <RiskBar
            label="Geometry Risk"
            value={
              data.risk.geometry
            }
          />
        </Panel>
      </div>

      <Panel
        title="RISK → RESOLUTION LOGIC"
        subtitle="Adaptive controller policy"
      >
        <div className="logic-grid">
          <div>
            <strong>
              ≥ 75%
            </strong>
            <span>
              5 cm resolution
            </span>
          </div>

          <div>
            <strong>
              ≥ 55%
            </strong>
            <span>
              10 cm resolution
            </span>
          </div>

          <div>
            <strong>
              ≥ 30%
            </strong>
            <span>
              25 cm resolution
            </span>
          </div>

          <div>
            <strong>
              &lt; 30%
            </strong>
            <span>
              Base resolution
            </span>
          </div>
        </div>
      </Panel>
    </div>
  );
}

/* =========================================================
   GRID TAB
========================================================= */

function GridTab({
  data,
}: {
  data: AppData;
}) {
  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            ADAPTIVE GRID
          </h2>

          <p>
            Variable-resolution 2.5D spatial
            representation.
          </p>
        </div>

        <div className="grid-total">
          {data.cells.toLocaleString()}
          <span>
            CELLS
          </span>
        </div>
      </div>

      <div className="zone-grid">
        {data.zones.map(
          (zone, index) => (
            <div
              className="zone-card"
              key={zone.name}
            >
              <div className="zone-index">
                0{index + 1}
              </div>

              <div className="zone-name">
                {zone.name}
              </div>

              <div className="zone-resolution">
                {(
                  zone.resolution *
                  100
                ).toFixed(0)}
                <span>
                  cm
                </span>
              </div>

              <div className="zone-range">
                {zone.min_distance}–
                {zone.max_distance}m
              </div>

              <div className="zone-stats">
                <span>
                  {zone.cells.toLocaleString()}{" "}
                  cells
                </span>

                <span>
                  {zone.points.toLocaleString()}{" "}
                  points
                </span>
              </div>
            </div>
          ),
        )}
      </div>

      <Panel
        title="VARIABLE RESOLUTION PROFILE"
        subtitle="Resolution changes with distance and risk"
      >
        <div className="resolution-visual">
          {data.zones.map(
            (zone) => (
              <div
                key={zone.name}
                className="resolution-column"
              >
                <div
                  className="resolution-block"
                  style={{
                    height: `${Math.max(
                      30,
                      150 -
                        zone.resolution *
                          200,
                    )}px`,
                  }}
                />

                <strong>
                  {(
                    zone.resolution *
                    100
                  ).toFixed(0)}
                  cm
                </strong>

                <span>
                  {zone.min_distance}
                  –
                  {zone.max_distance}
                  m
                </span>
              </div>
            ),
          )}
        </div>
      </Panel>
    </div>
  );
}

/* =========================================================
   2.5D MAP TAB
========================================================= */

function MapTab({
  data,
}: {
  data: AppData;
}) {
  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            2.5D SEMANTIC MAP
          </h2>

          <p>
            Top-down spatial representation of
            terrain and detected objects.
          </p>
        </div>
      </div>

      <Panel
        title="TOP-DOWN PERCEPTION MAP"
        subtitle="100m × 90m simulated perception area"
      >
        <div className="map-container">
          <div className="map-road" />

          {[10, 25, 50, 100].map(
            (radius) => (
              <div
                key={radius}
                className="map-ring"
                style={{
                  width: `${Math.min(
                    radius * 1.5,
                    95,
                  )}%`,
                  height: `${Math.min(
                    radius * 0.65,
                    85,
                  )}%`,
                }}
              />
            ),
          )}

          <div className="map-ego">
            ▲
          </div>

          {data.objects.map(
            (object) => {
              const x =
                50 +
                (object.centroid[0] /
                  50) *
                  45;

              const y =
                95 -
                (object.centroid[1] /
                  100) *
                  90;

              return (
                <div
                  key={
                    object.object_id
                  }
                  className="map-object"
                  style={{
                    left: `${x}%`,
                    top: `${y}%`,
                    background:
                      classColor(
                        object.class_id,
                      ),
                    boxShadow: `0 0 14px ${classColor(
                      object.class_id,
                    )}`,
                  }}
                  title={`Object #${object.object_id}`}
                />
              );
            },
          )}

          <div className="map-axis-x">
            LEFT ← → RIGHT
          </div>

          <div className="map-axis-y">
            FAR ↑
            <br />
            ↓ NEAR
          </div>
        </div>
      </Panel>
    </div>
  );
}

/* =========================================================
   MOTION TAB
========================================================= */

function MotionTab({
  data,
}: {
  data: AppData;
}) {
  const moving = data.objects.filter(
    (object) =>
      object.speed_mps > 0.4,
  );

  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            MOTION ANALYSIS
          </h2>

          <p>
            Object movement and temporal tracking
            analysis.
          </p>
        </div>

        <div className="motion-count">
          {moving.length}
          <span>
            MOVING OBJECTS
          </span>
        </div>
      </div>

      <div className="motion-grid">
        {data.objects
          .slice(0, 12)
          .map((object) => (
            <div
              className="motion-card"
              key={object.object_id}
            >
              <div className="motion-card-top">
                <span
                  className="class-dot"
                  style={{
                    background:
                      classColor(
                        object.class_id,
                      ),
                  }}
                />

                <strong>
                  TRACK #
                  {
                    object.object_id
                  }
                </strong>
              </div>

              <div className="speed-value">
                {object.speed_mps.toFixed(
                  1,
                )}
                <span>
                  m/s
                </span>
              </div>

              <div className="motion-bar">
                <div
                  style={{
                    width: `${Math.min(
                      object.speed_mps *
                        8,
                      100,
                    )}%`,
                  }}
                />
              </div>

              <small>
                {
                  object.class_name
                }
              </small>
            </div>
          ))}
      </div>

      <Panel
        title="TEMPORAL HYSTERESIS"
        subtitle="Prevents rapid resolution switching"
      >
        <div className="hysteresis">
          <div>
            <strong>
              REFINE
            </strong>

            <span>
              2 consecutive high-risk
              frames
            </span>
          </div>

          <div>
            <strong>
              HOLD
            </strong>

            <span>
              Maintain current resolution
            </span>
          </div>

          <div>
            <strong>
              COARSEN
            </strong>

            <span>
              4 consecutive low-risk
              frames
            </span>
          </div>
        </div>
      </Panel>
    </div>
  );
}

/* =========================================================
   PERFORMANCE TAB
========================================================= */

function PerformanceTab({
  data,
}: {
  data: AppData;
}) {
  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            PERFORMANCE
          </h2>

          <p>
            Runtime metrics for the perception
            pipeline.
          </p>
        </div>
      </div>

      <div className="performance-grid">
        <MetricCard
          label="FRAME RATE"
          value={data.fps.toFixed(1)}
          unit="FPS"
          accent="green"
        />

        <MetricCard
          label="LATENCY"
          value={data.latency.toFixed(1)}
          unit="ms"
          accent="blue"
        />

        <MetricCard
          label="POINTS PROCESSED"
          value={data.points.length.toLocaleString()}
        />

        <MetricCard
          label="GRID CELLS"
          value={data.cells.toLocaleString()}
          accent="purple"
        />
      </div>

      <Panel
        title="PIPELINE TIMING"
        subtitle="Illustrative runtime breakdown"
      >
        <div className="timing-list">
          {[
            [
              "Point filtering",
              4.2,
            ],
            [
              "Semantic inference",
              12.8,
            ],
            [
              "Object tracking",
              5.6,
            ],
            [
              "Risk controller",
              3.4,
            ],
            [
              "Adaptive grid",
              5.1,
            ],
            [
              "Visualization",
              3.7,
            ],
          ].map(
            ([label, value]) => (
              <div
                className="timing-row"
                key={String(
                  label,
                )}
              >
                <span>
                  {label}
                </span>

                <div className="timing-track">
                  <div
                    style={{
                      width: `${Number(value) * 5}%`,
                    }}
                  />
                </div>

                <strong>
                  {value} ms
                </strong>
              </div>
            ),
          )}
        </div>
      </Panel>
    </div>
  );
}

/* =========================================================
   PIPELINE TAB
========================================================= */

function PipelineTab() {
  const steps = [
    [
      "01",
      "RAW LiDAR",
      "Point cloud acquisition",
    ],
    [
      "02",
      "FILTERING",
      "Noise and outlier removal",
    ],
    [
      "03",
      "SEMANTIC AI",
      "Point-wise classification",
    ],
    [
      "04",
      "OBJECT TRACKING",
      "Motion and identity tracking",
    ],
    [
      "05",
      "RISK ENGINE",
      "Risk-aware resolution selection",
    ],
    [
      "06",
      "ADAPTIVE GRID",
      "Variable-resolution 2.5D map",
    ],
    [
      "07",
      "PERCEPTION MAP",
      "Terrain + obstacle representation",
    ],
    [
      "08",
      "DECISION OUTPUT",
      "Real-time navigation information",
    ],
  ];

  return (
    <div className="tab-content">
      <div className="feature-header">
        <div>
          <h2>
            SYSTEM PIPELINE
          </h2>

          <p>
            End-to-end ADAPT-LiDAR processing
            architecture.
          </p>
        </div>
      </div>

      <div className="pipeline-full">
        {steps.map(
          ([number, title, text], index) => (
            <div
              className="pipeline-full-row"
              key={number}
            >
              <div className="pipeline-number">
                {number}
              </div>

              <div className="pipeline-node">
                <div>
                  <strong>
                    {title}
                  </strong>

                  <span>
                    {text}
                  </span>
                </div>

                {index <
                  steps.length - 1 && (
                  <div className="pipeline-arrow">
                    ↓
                  </div>
                )}
              </div>
            </div>
          ),
        )}
      </div>
    </div>
  );
}

/* =========================================================
   MAIN APP
========================================================= */

export default function App() {
  const fallback = useMemo(
    () => createFallbackData(),
    [],
  );

  const [data, setData] =
    useState<AppData>(
      fallback,
    );

  const [
    connected,
    setConnected,
  ] = useState(false);

  const [
    lightMode,
    setLightMode,
  ] = useState(false);

  const [
    activeTab,
    setActiveTab,
  ] = useState("overview");

  const [
    selected,
    setSelected,
  ] = useState<number | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      try {
        const response =
          await fetch(
            `${API}/api/lidar/perception`,
          );

        if (!response.ok) {
          throw new Error(
            "Backend error",
          );
        }

        const raw =
          await response.json();

        if (!cancelled) {
          setData(
            normalizeBackendData(
              raw,
            ),
          );

          setConnected(true);
        }
      } catch {
        if (!cancelled) {
          setData(
            fallback,
          );

          setConnected(false);
        }
      }
    }

    fetchData();

    const interval =
      window.setInterval(
        fetchData,
        1000,
      );

    return () => {
      cancelled = true;
      window.clearInterval(
        interval,
      );
    };
  }, [fallback]);

  const tabs = [
    ["overview", "OVERVIEW"],
    ["lidar", "3D LiDAR"],
    ["semantic", "SEMANTIC AI"],
    ["objects", "TRACKED OBJECTS"],
    ["risk", "RISK CONTROLLER"],
    ["grid", "ADAPTIVE GRID"],
    ["map", "2.5D MAP"],
    ["motion", "MOTION"],
    ["performance", "PERFORMANCE"],
    ["pipeline", "PIPELINE"],
  ];

  return (
    <div
      className={
        lightMode
          ? "app light-mode"
          : "app"
      }
    >
      {/* HEADER */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            AL
          </div>

          <div>
            <h1>
              ADAPT-LiDAR
            </h1>

            <span>
              RISK-AWARE 2.5D PERCEPTION
            </span>
          </div>
        </div>

        <div className="topbar-right">
          <div className="connection-status">
            <span
              className={
                connected
                  ? "live-dot"
                  : "simulation-dot"
              }
            />

            {connected
              ? "BACKEND CONNECTED"
              : "SIMULATION MODE"}
          </div>

          <button
            className="theme-button"
            onClick={() =>
              setLightMode(
                (value) =>
                  !value,
              )
            }
          >
            {lightMode
              ? "☾"
              : "☀"}
          </button>
        </div>
      </header>

      {/* NAVIGATION */}
      <nav className="main-tabs">
        {tabs.map(
          ([id, label]) => (
            <button
              key={id}
              className={
                activeTab === id
                  ? "main-tab active"
                  : "main-tab"
              }
              onClick={() =>
                setActiveTab(
                  id,
                )
              }
            >
              {label}
            </button>
          ),
        )}
      </nav>

      {/* CONTENT */}
      <main className="main-content">
        {activeTab ===
          "overview" && (
          <OverviewTab
            data={data}
            onSelect={setSelected}
            setTab={
              setActiveTab
            }
          />
        )}

        {activeTab ===
          "lidar" && (
          <LidarTab
            data={data}
            lightMode={
              lightMode
            }
          />
        )}

        {activeTab ===
          "semantic" && (
          <SemanticTab
            data={data}
          />
        )}

        {activeTab ===
          "objects" && (
          <ObjectsTab
            data={data}
            selected={
              selected
            }
            onSelect={
              setSelected
            }
          />
        )}

        {activeTab ===
          "risk" && (
          <RiskTab
            data={data}
          />
        )}

        {activeTab ===
          "grid" && (
          <GridTab
            data={data}
          />
        )}

        {activeTab ===
          "map" && (
          <MapTab
            data={data}
          />
        )}

        {activeTab ===
          "motion" && (
          <MotionTab
            data={data}
          />
        )}

        {activeTab ===
          "performance" && (
          <PerformanceTab
            data={data}
          />
        )}

        {activeTab ===
          "pipeline" && (
          <PipelineTab />
        )}
      </main>

      {/* FOOTER */}
      <footer className="footer">
        <div>
          <strong>
            ADAPT-LiDAR
          </strong>

          <span>
            SIH26053 • Dynamic Environment
            Perception
          </span>
        </div>

        <div>
          {data.source} •{" "}
          {new Date().toLocaleTimeString()}
        </div>
      </footer>
    </div>
  );
}