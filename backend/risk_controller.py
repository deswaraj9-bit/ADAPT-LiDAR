import math


class RiskAwareResolutionController:
    """
    Risk-Aware Adaptive Resolution Controller.

    The controller starts with a distance-based resolution and
    refines the resolution when the area/object has higher risk.

    Resolution levels:
        0.05 m = 5 cm
        0.10 m = 10 cm
        0.25 m = 25 cm
        0.50 m = 50 cm
    """

    CLASS_RISK = {
        0: 0.05,   # Ground
        1: 0.70,   # Vehicle
        2: 0.90,   # Pedestrian
        3: 0.60,   # Static obstacle
    }

    CLASS_NAMES = {
        0: "ground",
        1: "vehicle",
        2: "pedestrian",
        3: "static_obstacle",
    }

    def __init__(self):

        # Risk weights
        self.distance_weight = 0.20
        self.semantic_weight = 0.30
        self.uncertainty_weight = 0.20
        self.motion_weight = 0.20
        self.geometry_weight = 0.10

    # ========================================================
    # DISTANCE BASELINE
    # ========================================================

    def get_base_resolution(self, distance):
        """
        Distance-only adaptive resolution.

        0-10m     -> 5cm
        10-25m    -> 10cm
        25-50m    -> 25cm
        50-100m   -> 50cm
        """

        distance = float(distance)

        if distance <= 10:
            return 0.05

        elif distance <= 25:
            return 0.10

        elif distance <= 50:
            return 0.25

        else:
            return 0.50

    # ========================================================
    # DISTANCE RISK
    # ========================================================

    def get_distance_risk(self, distance):
        """
        Converts distance into a normalized risk.

        Closer objects receive higher distance risk.
        """

        distance = max(
            0.0,
            float(distance)
        )

        # 0m -> 1.0 risk
        # 100m -> 0.0 risk

        risk = 1.0 - (
            min(distance, 100.0) / 100.0
        )

        return max(
            0.0,
            min(1.0, risk)
        )

    # ========================================================
    # SEMANTIC RISK
    # ========================================================

    def get_semantic_risk(self, class_id):
        """
        Risk according to semantic class.
        """

        return self.CLASS_RISK.get(
            int(class_id),
            0.50
        )

    # ========================================================
    # UNCERTAINTY RISK
    # ========================================================

    def get_uncertainty_risk(self, confidence):
        """
        Low ML confidence means higher uncertainty risk.

        confidence = 1.0 -> uncertainty 0
        confidence = 0.0 -> uncertainty 1
        """

        confidence = max(
            0.0,
            min(1.0, float(confidence))
        )

        return 1.0 - confidence

    # ========================================================
    # MOTION RISK
    # ========================================================

    def get_motion_risk(
        self,
        speed_mps=0.0
    ):
        """
        Converts estimated object speed into normalized risk.

        0 m/s -> 0 risk
        >= 10 m/s -> 1 risk
        """

        speed = abs(
            float(speed_mps)
        )

        risk = speed / 10.0

        return max(
            0.0,
            min(1.0, risk)
        )

    # ========================================================
    # GEOMETRY RISK
    # ========================================================

    def get_geometry_risk(
        self,
        terrain_variation=0.0
    ):
        """
        Terrain/elevation variation risk.

        terrain_variation should be normalized between 0 and 1.
        """

        return max(
            0.0,
            min(
                1.0,
                float(terrain_variation)
            )
        )

    # ========================================================
    # TOTAL RISK
    # ========================================================

    def calculate_risk(
        self,
        distance,
        class_id,
        confidence,
        speed_mps=0.0,
        terrain_variation=0.0
    ):

        distance_risk = self.get_distance_risk(
            distance
        )

        semantic_risk = self.get_semantic_risk(
            class_id
        )

        uncertainty_risk = self.get_uncertainty_risk(
            confidence
        )

        motion_risk = self.get_motion_risk(
            speed_mps
        )

        geometry_risk = self.get_geometry_risk(
            terrain_variation
        )

        total_risk = (

            self.distance_weight *
            distance_risk

            +

            self.semantic_weight *
            semantic_risk

            +

            self.uncertainty_weight *
            uncertainty_risk

            +

            self.motion_weight *
            motion_risk

            +

            self.geometry_weight *
            geometry_risk
        )

        return max(
            0.0,
            min(1.0, total_risk)
        )

    # ========================================================
    # RISK → RESOLUTION
    # ========================================================

    def resolution_from_risk(
        self,
        base_resolution,
        risk
    ):
        """
        Risk can refine the base resolution.

        It never makes the resolution coarser than
        the distance-based baseline.
        """

        base_resolution = float(
            base_resolution
        )

        risk = float(risk)

        # Very high risk
        if risk >= 0.75:

            return min(
                base_resolution,
                0.05
            )

        # High risk
        elif risk >= 0.55:

            return min(
                base_resolution,
                0.10
            )

        # Medium risk
        elif risk >= 0.30:

            return min(
                base_resolution,
                0.25
            )

        # Low risk
        else:

            return base_resolution

    # ========================================================
    # COMPLETE DECISION
    # ========================================================

    def get_resolution(
        self,
        distance,
        class_id,
        confidence,
        speed_mps=0.0,
        terrain_variation=0.0
    ):

        base_resolution = self.get_base_resolution(
            distance
        )

        risk = self.calculate_risk(
            distance=distance,
            class_id=class_id,
            confidence=confidence,
            speed_mps=speed_mps,
            terrain_variation=terrain_variation
        )

        final_resolution = self.resolution_from_risk(
            base_resolution,
            risk
        )

        class_name = self.CLASS_NAMES.get(
            int(class_id),
            "unknown"
        )

        return {
            "class_id": int(class_id),
            "class_name": class_name,
            "distance": round(
                float(distance),
                2
            ),
            "confidence": round(
                float(confidence),
                3
            ),
            "distance_risk": round(
                self.get_distance_risk(distance),
                3
            ),
            "semantic_risk": round(
                self.get_semantic_risk(class_id),
                3
            ),
            "uncertainty_risk": round(
                self.get_uncertainty_risk(confidence),
                3
            ),
            "motion_risk": round(
                self.get_motion_risk(speed_mps),
                3
            ),
            "geometry_risk": round(
                self.get_geometry_risk(
                    terrain_variation
                ),
                3
            ),
            "total_risk": round(
                risk,
                3
            ),
            "normal_resolution": base_resolution,
            "risk_aware_resolution": final_resolution,
        }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    controller = (
        RiskAwareResolutionController()
    )

    tests = [

        # Empty/ground far away
        {
            "name": "Empty road at 70m",
            "distance": 70,
            "class_id": 0,
            "confidence": 0.98,
            "speed_mps": 0,
            "terrain_variation": 0.05,
        },

        # Vehicle
        {
            "name": "Vehicle at 40m",
            "distance": 40,
            "class_id": 1,
            "confidence": 0.95,
            "speed_mps": 4,
            "terrain_variation": 0.10,
        },

        # Pedestrian
        {
            "name": "Pedestrian at 35m",
            "distance": 35,
            "class_id": 2,
            "confidence": 0.90,
            "speed_mps": 1.5,
            "terrain_variation": 0.10,
        },

        # Uncertain object
        {
            "name": "Uncertain obstacle at 60m",
            "distance": 60,
            "class_id": 3,
            "confidence": 0.55,
            "speed_mps": 2,
            "terrain_variation": 0.40,
        },
    ]

    print()
    print("=" * 70)
    print("ADAPT-LiDAR RISK-AWARE RESOLUTION CONTROLLER")
    print("=" * 70)

    for test in tests:

        result = controller.get_resolution(
            distance=test["distance"],
            class_id=test["class_id"],
            confidence=test["confidence"],
            speed_mps=test["speed_mps"],
            terrain_variation=test["terrain_variation"],
        )

        print()
        print(test["name"])
        print("-" * 70)

        print(
            "Class:",
            result["class_name"]
        )

        print(
            "Distance:",
            result["distance"],
            "m"
        )

        print(
            "Confidence:",
            result["confidence"]
        )

        print(
            "Total risk:",
            result["total_risk"]
        )

        print(
            "Normal resolution:",
            result["normal_resolution"],
            "m"
        )

        print(
            "Risk-aware resolution:",
            result["risk_aware_resolution"],
            "m"
        )