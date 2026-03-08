#!/usr/bin/env python3
"""
Layer 1 – SimulationEngine
Produces realistic physics-based nanosatellite telemetry at a configurable
sample rate using poliastro (orbital mechanics), ahrs (attitude dynamics),
pymap3d (coordinate transforms) and numpy (sensor noise).
"""

import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, List

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from ahrs.filters import Madgwick
import pymap3d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants – no magic numbers inline
# ---------------------------------------------------------------------------
GYRO_NOISE_SIGMA = 0.001          # rad/s
ACCEL_NOISE_SIGMA = 0.01          # m/s²
MAG_NOISE_SIGMA = 0.5             # µT
SHADOW_CYLINDER_RADIUS_KM = 6371.0 + 100.0  # Earth radius + atmosphere margin

EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3S2 = 398600.4418     # GM for quick orbital-period calc
EARTH_MAG_MOMENT_UT_KM3 = 7.94e6 # simplified dipole (µT·km³)
STANDARD_GRAVITY_MS2 = 9.80665


class SimulationEngine:
    """Physics-based orbital + attitude + IMU telemetry generator."""

    def __init__(
        self,
        orbit_altitude_km: float = 550.0,
        inclination_deg: float = 45.0,
        sample_rate_hz: float = 1.0,
        mission_duration_s: float = 5400.0,
        noise_seed: int = 42,
    ):
        self.orbit_altitude_km = orbit_altitude_km
        self.inclination_deg = inclination_deg
        self.sample_rate_hz = sample_rate_hz
        self.mission_duration_s = mission_duration_s
        self.dt = 1.0 / sample_rate_hz
        self.rng = np.random.default_rng(noise_seed)

        # Orbital state
        self.epoch = Time(datetime(2025, 1, 1, tzinfo=timezone.utc))
        self._orbit_base = Orbit.circular(
            Earth,
            alt=orbit_altitude_km * u.km,
            inc=inclination_deg * u.deg,
            epoch=self.epoch,
        )

        # Approximate orbital period for sun-vector rotation
        r_orbit_km = EARTH_RADIUS_KM + orbit_altitude_km
        self.orbital_period_s = 2.0 * np.pi * np.sqrt(r_orbit_km ** 3 / EARTH_MU_KM3S2)

        # Attitude filter (Madgwick)
        self._madgwick = Madgwick(frequency=sample_rate_hz)
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Internal counters
        self._sample_index = 0
        self._mission_time = 0.0
        self._epoch_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)

        logger.info(
            "SimulationEngine initialised: alt=%.0f km, inc=%.1f°, "
            "rate=%.1f Hz, duration=%.0f s",
            orbit_altitude_km, inclination_deg, sample_rate_hz, mission_duration_s,
        )

    # ------------------------------------------------------------------
    # Orbital mechanics
    # ------------------------------------------------------------------
    def _propagate_orbit(self, mission_time_s: float):
        """Propagate orbit to *mission_time_s* and return (r_km, v_km_s)."""
        orb = self._orbit_base.propagate(mission_time_s * u.s)
        r_km = orb.r.to(u.km).value   # ndarray [3]
        v_kms = orb.v.to(u.km / u.s).value
        return r_km, v_kms

    @staticmethod
    def _sun_vector(mission_time_s: float) -> np.ndarray:
        """Simplified Sun direction in ECI (rotates once per year)."""
        angle = 2.0 * np.pi * mission_time_s / (365.25 * 86400.0)
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    def _is_sunlit(self, position_eci_km: np.ndarray, mission_time_s: float) -> bool:
        """Shadow-cylinder model: sunlit when not behind Earth."""
        sun_dir = self._sun_vector(mission_time_s)
        projection = np.dot(position_eci_km, sun_dir)
        if projection > 0.0:
            return True
        # Behind Earth – check if within shadow cylinder
        perp = position_eci_km - projection * sun_dir
        return np.linalg.norm(perp) > SHADOW_CYLINDER_RADIUS_KM

    def _eci_to_geodetic(self, r_km: np.ndarray, mission_time_s: float):
        """Convert ECI position (km) to lat/lon/alt using pymap3d."""
        t = self._epoch_dt + timedelta(seconds=mission_time_s)
        # pymap3d expects metres
        lat, lon, alt_m = pymap3d.eci2geodetic(
            r_km[0] * 1e3, r_km[1] * 1e3, r_km[2] * 1e3, t,
        )
        return float(lat), float(lon), float(alt_m / 1e3)

    # ------------------------------------------------------------------
    # Attitude dynamics
    # ------------------------------------------------------------------
    def _compute_nadir_target(self, r_km: np.ndarray, v_kms: np.ndarray):
        """Nadir-pointing reference: z_body→Earth, x_body→velocity."""
        z_body = -r_km / np.linalg.norm(r_km)
        x_body = v_kms / np.linalg.norm(v_kms)
        y_body = np.cross(z_body, x_body)
        y_body /= np.linalg.norm(y_body)
        x_body = np.cross(y_body, z_body)
        return x_body, y_body, z_body

    def _update_attitude(self, r_km: np.ndarray, v_kms: np.ndarray):
        """Run one Madgwick filter step toward the nadir-pointing target."""
        _, _, z_body = self._compute_nadir_target(r_km, v_kms)

        # Orbital angular velocity about the velocity-normal axis
        r_mag = np.linalg.norm(r_km)
        omega_orbit = np.sqrt(EARTH_MU_KM3S2 / r_mag ** 3)  # rad/s
        orbit_axis = np.cross(r_km, v_kms)
        orbit_axis /= np.linalg.norm(orbit_axis)
        true_angular_velocity = omega_orbit * orbit_axis  # rad/s

        # Synthetic accelerometer: gravity in body frame ≈ nadir direction
        acc_ref = -z_body * STANDARD_GRAVITY_MS2

        self._quaternion = self._madgwick.updateIMU(
            self._quaternion, true_angular_velocity, acc_ref, dt=self.dt,
        )

        return true_angular_velocity, acc_ref

    @staticmethod
    def _quat_to_euler(q):
        """Convert quaternion [w,x,y,z] to Euler angles [roll,pitch,yaw] in degrees."""
        w, x, y, z = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.degrees(np.arcsin(sinp))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

        return [roll, pitch, yaw]

    # ------------------------------------------------------------------
    # IMU sensor noise model
    # ------------------------------------------------------------------
    def _noisy_gyro(self, true_omega):
        return true_omega + self.rng.normal(0.0, GYRO_NOISE_SIGMA, size=3)

    def _noisy_accel(self, acc_ref):
        return acc_ref + self.rng.normal(0.0, ACCEL_NOISE_SIGMA, size=3)

    def _noisy_magnetometer(self, r_km: np.ndarray):
        """Simplified dipole magnetic field + noise (µT)."""
        r_mag = np.linalg.norm(r_km)
        r_hat = r_km / r_mag
        # Dipole: B = (moment / r^3) * (3(m·r_hat)r_hat - m)
        # Use z-aligned dipole
        m_hat = np.array([0.0, 0.0, 1.0])
        dot = np.dot(m_hat, r_hat)
        B = (EARTH_MAG_MOMENT_UT_KM3 / r_mag ** 3) * (3.0 * dot * r_hat - m_hat)
        return B + self.rng.normal(0.0, MAG_NOISE_SIGMA, size=3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self) -> dict:
        """Advance one timestep and return the full telemetry dict."""
        t = self._mission_time

        # 1. Orbital mechanics
        r_km, v_kms = self._propagate_orbit(t)
        sunlit = self._is_sunlit(r_km, t)
        lat, lon, alt_km = self._eci_to_geodetic(r_km, t)

        # 2. Attitude dynamics
        true_omega, acc_ref = self._update_attitude(r_km, v_kms)
        euler = self._quat_to_euler(self._quaternion)

        # 3. IMU sensor output
        gyro = self._noisy_gyro(true_omega)
        accel = self._noisy_accel(acc_ref)
        mag = self._noisy_magnetometer(r_km)

        sample = {
            "timestamp": t,
            "mission_time_s": t,
            "sample_index": self._sample_index,
            # Orbital
            "position_eci": r_km.tolist(),
            "velocity_eci": v_kms.tolist(),
            "orbit_sunlight": bool(sunlit),
            "latitude_deg": lat,
            "longitude_deg": lon,
            "altitude_km": alt_km,
            # Attitude
            "quaternion": self._quaternion.tolist(),
            "euler_angles_deg": euler,
            "angular_velocity_rad_s": true_omega.tolist(),
            # IMU
            "gyroscope_rad_s": gyro.tolist(),
            "accelerometer_m_s2": accel.tolist(),
            "magnetometer_uT": mag.tolist(),
        }

        # Advance internal clock
        self._sample_index += 1
        self._mission_time += self.dt

        return sample

    def run(self) -> List[dict]:
        """Run the full simulation and return all timestep dicts."""
        total_steps = int(self.mission_duration_s * self.sample_rate_hz)
        results: List[dict] = []
        for _ in range(total_steps):
            results.append(self.step())
        logger.info("Batch run complete: %d samples generated.", len(results))
        return results

    def stream(self, callback: Callable[[dict], None]) -> None:
        """Call *callback(step_dict)* at real-time rate."""
        total_steps = int(self.mission_duration_s * self.sample_rate_hz)
        interval = 1.0 / self.sample_rate_hz
        for _ in range(total_steps):
            sample = self.step()
            callback(sample)
            time.sleep(interval)


# -----------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    EXPECTED_KEYS = {
        "timestamp", "mission_time_s", "sample_index",
        "position_eci", "velocity_eci", "orbit_sunlight",
        "latitude_deg", "longitude_deg", "altitude_km",
        "quaternion", "euler_angles_deg", "angular_velocity_rad_s",
        "gyroscope_rad_s", "accelerometer_m_s2", "magnetometer_uT",
    }

    engine = SimulationEngine()

    print("=== SimulationEngine self-test (10 steps) ===\n")
    for i in range(10):
        sample = engine.step()

        # Assertions
        missing = EXPECTED_KEYS - set(sample.keys())
        assert not missing, f"Step {i}: missing keys {missing}"
        assert isinstance(sample["orbit_sunlight"], bool), \
            f"Step {i}: orbit_sunlight is not bool"
        assert len(sample["quaternion"]) == 4, \
            f"Step {i}: quaternion has {len(sample['quaternion'])} elements"
        assert len(sample["position_eci"]) == 3
        assert len(sample["velocity_eci"]) == 3
        assert len(sample["gyroscope_rad_s"]) == 3
        assert len(sample["accelerometer_m_s2"]) == 3
        assert len(sample["magnetometer_uT"]) == 3

        print(json.dumps(sample, indent=2))
        print()

    print("All assertions passed.")
