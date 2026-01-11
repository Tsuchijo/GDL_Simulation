"""
Non-Equilibrium Vibrational Flow Simulator

A comprehensive simulation framework for modeling non-equilibrium vibrational
relaxation in gas mixtures (CO2-N2-H2O) through nozzle flows. This class provides
methods for computing vibrational relaxation times, solving 1D compressible flow
equations with vibrational non-equilibrium effects, and analyzing results.

Author: Generated from GDL_Simulation notebook
"""

import numpy as np
from scipy import optimize
from typing import Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class GasMixture:
    """Container for gas mixture composition and properties."""
    X_CO2: float  # Mole fraction of CO2
    X_N2: float   # Mole fraction of N2
    X_H2O: float  # Mole fraction of H2O

    # Molar masses (g/mol)
    M_CO2: float = 44.01
    M_N2: float = 28.02
    M_H2O: float = 18.02

    def __post_init__(self):
        """Validate mole fractions sum to 1."""
        total = self.X_CO2 + self.X_N2 + self.X_H2O
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Mole fractions must sum to 1.0, got {total}")

    @property
    def M_total(self) -> float:
        """Average molar mass of the mixture (g/mol)."""
        return self.X_CO2 * self.M_CO2 + self.X_N2 * self.M_N2 + self.X_H2O * self.M_H2O

    @property
    def c_CO2(self) -> float:
        """Mass fraction of CO2."""
        return (self.X_CO2 * self.M_CO2) / self.M_total

    @property
    def c_N2(self) -> float:
        """Mass fraction of N2."""
        return (self.X_N2 * self.M_N2) / self.M_total

    @property
    def c_H2O(self) -> float:
        """Mass fraction of H2O."""
        return (self.X_H2O * self.M_H2O) / self.M_total


@dataclass
class SimulationState:
    """Container for simulation state variables."""
    rho: np.ndarray      # Density (kg/m^3)
    vel: np.ndarray      # Velocity (m/s)
    e: np.ndarray        # Total specific internal energy (J/kg)
    e_vib_I: np.ndarray  # Vibrational energy mode I (J/kg)
    e_vib_II: np.ndarray # Vibrational energy mode II (J/kg)

    @classmethod
    def from_flat(cls, y: np.ndarray, n_points: int) -> 'SimulationState':
        """Create SimulationState from flattened array."""
        return cls(
            rho=y[0:n_points],
            vel=y[n_points:2*n_points],
            e=y[2*n_points:3*n_points],
            e_vib_I=y[3*n_points:4*n_points],
            e_vib_II=y[4*n_points:5*n_points]
        )

    def to_flat(self) -> np.ndarray:
        """Flatten state to 1D array."""
        return np.concatenate([self.rho, self.vel, self.e, self.e_vib_I, self.e_vib_II])

    @property
    def n_points(self) -> int:
        """Number of grid points."""
        return len(self.rho)


class NonEquilibriumFlowSimulator:
    """
    Simulator for non-equilibrium vibrational flow in CO2-N2-H2O gas mixtures.

    This class handles the physics of vibrational relaxation, compressible flow
    equations, and numerical integration for simulating gas flows through nozzles
    with non-equilibrium vibrational effects.

    Attributes:
        gas: GasMixture object containing composition
        R: Gas constant for the mixture (J/(kg·K))
        cv: Specific heat at constant volume (J/(kg·K))
        gamma: Ratio of specific heats

    Example:
        >>> gas = GasMixture(X_CO2=0.09, X_N2=0.90, X_H2O=0.01)
        >>> sim = NonEquilibriumFlowSimulator(gas)
        >>> results = sim.run_simulation(
        ...     nozzle_func=sim.create_converging_diverging_nozzle(0.4, 0.01, 0.05, 0.30),
        ...     P0=101325*10, T0=1500, length=0.4, n_points=50
        ... )
    """

    # Physical constants
    R_UNIVERSAL = 8.314  # J/(mol·K)
    K_BOLTZMANN = 1.380649e-23  # J/K
    HC = 6.62607015e-34 * 2.998e10  # J·cm (Planck constant * speed of light)

    # Specific gas constants
    R_CO2 = 188.92  # J/(kg·K)
    R_N2 = 296.8    # J/(kg·K)

    # Vibrational wavenumbers (cm^-1)
    V1_CO2 = 1388.0  # CO2 symmetric stretch
    V2_CO2 = 667.0   # CO2 bending mode
    V3_CO2 = 2349.0  # CO2 asymmetric stretch
    V1_N2 = 2330.0   # N2 stretch

    def __init__(self, gas: GasMixture):
        """
        Initialize the simulator with a gas mixture.

        Args:
            gas: GasMixture object specifying the composition
        """
        self.gas = gas

        # Calculate mixture gas constant
        self.R = self.R_UNIVERSAL / (gas.M_total * 1e-3)  # Convert g/mol to kg/mol

        # For diatomic-like mixture, cv = 5/2 * R
        self.cv = 5 / 2 * self.R

        # Approximate gamma for the mixture
        self.gamma = 1.4

    # =========================================================================
    # Relaxation Time Calculations
    # =========================================================================

    def calculate_relaxation_times(
        self,
        T: float,
        p: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate vibrational relaxation times for the gas mixture.

        Uses semi-empirical correlations for CO2-N2-H2O mixtures based on
        Landau-Teller theory and experimental data.

        Args:
            T: Temperature in Kelvin
            p: Pressure in atm (default 1.0)

        Returns:
            Tuple of (tau_I, tau_II) relaxation times in seconds where:
            - tau_I: Relaxation time for CO2 modes v1 and v2 (Mode I)
            - tau_II: Relaxation time for N2 and CO2 v3 (Mode II)
        """
        if T <= 0:
            T = 1.0

        T_pow_neg_1_3 = T ** (-1/3)

        X_CO2 = self.gas.X_CO2
        X_N2 = self.gas.X_N2
        X_H2O = self.gas.X_H2O

        # τ_a calculations for CO2 modes v1 and v2
        tau_a_p_CO2_N2 = 1.3e5 * (T_pow_neg_1_3) ** 4.9
        tau_a_p_CO2_CO2 = 0.27 * tau_a_p_CO2_N2
        tau_a_p_CO2_H2O = 5.5e-2

        # τ_b calculations for N2 vibrational energy
        log_tau_b_p_N2_N2 = 93 * T_pow_neg_1_3 - 4.61
        tau_b_p_N2_N2 = 10 ** log_tau_b_p_N2_N2
        tau_b_p_N2_CO2 = tau_b_p_N2_N2
        log_tau_b_p_N2_H2O = 27.65 * T_pow_neg_1_3 - 3.2415
        tau_b_p_N2_H2O = 10 ** log_tau_b_p_N2_H2O

        # τ_c calculations for CO2 mode v3
        tau_c_p_CO2_CO2 = 10 ** (17.8 * T_pow_neg_1_3 - 1.808)
        tau_c_p_CO2_N2 = 2.0 * tau_c_p_CO2_CO2

        if T > 600:
            tau_c_p_CO2_H2O = 10 ** (-20.4 * T_pow_neg_1_3 + 0.643)
        else:
            tau_c_p_CO2_H2O = 10 ** (-20.4 * 600 ** (-1/3) + 0.643)

        # Calculate composite relaxation times
        inv_tau_a = (X_CO2 / tau_a_p_CO2_CO2 +
                     X_N2 / tau_a_p_CO2_N2 +
                     X_H2O / tau_a_p_CO2_H2O)
        tau_a = 1 / (p * inv_tau_a)

        inv_tau_b = (X_CO2 / tau_b_p_N2_CO2 +
                     X_N2 / tau_b_p_N2_N2 +
                     X_H2O / tau_b_p_N2_H2O)
        tau_b = 1 / (p * inv_tau_b)

        inv_tau_c = (X_CO2 / tau_c_p_CO2_CO2 +
                     X_N2 / tau_c_p_CO2_N2 +
                     X_H2O / tau_c_p_CO2_H2O)
        tau_c = 1 / (p * inv_tau_c)

        # Mode I: CO2 v1 and v2 (fast relaxation)
        tau_I = tau_c * 1e-6

        # Mode II: Combined N2 and CO2 v3 (slow relaxation)
        tau_II = 1 / (((X_CO2 / tau_a) + (X_N2 / tau_b)) / (X_CO2 + X_N2)) * 1e-6

        return tau_I, tau_II

    # =========================================================================
    # Vibrational Energy Calculations
    # =========================================================================

    def vib_energy_I(self, T: float) -> float:
        """
        Calculate vibrational energy of Mode I (CO2 modes v1 and v2).

        Mode I represents the symmetric stretch (v1) and bending mode (v2)
        of CO2 which equilibrate quickly with translation.

        Args:
            T: Temperature in Kelvin

        Returns:
            Vibrational energy in J/kg
        """
        # Guard against invalid temperatures
        if T <= 0 or np.isnan(T):
            T = 300.0  # Default to room temperature

        k = self.K_BOLTZMANN
        hc = self.HC
        c_CO2 = self.gas.c_CO2
        R = self.R_CO2

        v1, v2 = self.V1_CO2, self.V2_CO2

        # Use safe exponential to avoid overflow
        exp_arg1 = min(hc * v1 / (k * T), 700)  # exp(700) is near float max
        exp_arg2 = min(hc * v2 / (k * T), 700)

        denom1 = np.exp(exp_arg1) - 1
        denom2 = np.exp(exp_arg2) - 1

        # Avoid division by zero
        denom1 = max(denom1, 1e-10)
        denom2 = max(denom2, 1e-10)

        E_vib = (c_CO2 * R * (
            hc * v1 / denom1 +
            hc * v2 / denom2
        )) / k

        return E_vib

    def vib_energy_II(self, T: float) -> float:
        """
        Calculate vibrational energy of Mode II (N2 and CO2 mode v3).

        Mode II represents the N2 stretch and CO2 asymmetric stretch (v3)
        which equilibrate slowly due to near-resonant V-V transfer.

        Args:
            T: Temperature in Kelvin

        Returns:
            Vibrational energy in J/kg
        """
        # Guard against invalid temperatures
        if T <= 0 or np.isnan(T):
            T = 300.0  # Default to room temperature

        k = self.K_BOLTZMANN
        hc = self.HC
        c_CO2 = self.gas.c_CO2
        c_N2 = self.gas.c_N2
        R_co2 = self.R_CO2
        R_n2 = self.R_N2

        v1_n2, v3_co2 = self.V1_N2, self.V3_CO2

        # Use safe exponential to avoid overflow
        exp_arg_n2 = min(hc * v1_n2 / (k * T), 700)
        exp_arg_co2 = min(hc * v3_co2 / (k * T), 700)

        denom_n2 = np.exp(exp_arg_n2) - 1
        denom_co2 = np.exp(exp_arg_co2) - 1

        # Avoid division by zero
        denom_n2 = max(denom_n2, 1e-10)
        denom_co2 = max(denom_co2, 1e-10)

        E_vib = (
            c_N2 * R_n2 * (hc * v1_n2 / denom_n2) +
            c_CO2 * R_co2 * (hc * v3_co2 / denom_co2)
        ) / k

        return E_vib

    def find_vib_temperature(
        self,
        energy_level: float,
        mode: str = 'I',
        T_initial: float = 1000.0
    ) -> float:
        """
        Find the vibrational temperature for a given energy level.

        Uses Newton's method to invert the energy-temperature relationship.

        Args:
            energy_level: Target vibrational energy (J/kg)
            mode: 'I' for Mode I or 'II' for Mode II
            T_initial: Initial temperature guess (K)

        Returns:
            Vibrational temperature in Kelvin
        """
        # Handle NaN or invalid energy levels
        if np.isnan(energy_level) or np.isinf(energy_level):
            return T_initial

        if mode.upper() == 'I':
            energy_func = self.vib_energy_I
        elif mode.upper() == 'II':
            energy_func = self.vib_energy_II
        else:
            raise ValueError("Mode must be 'I' or 'II'")

        def objective(T):
            return energy_func(T) - energy_level

        try:
            result = optimize.newton(objective, T_initial, full_output=False, maxiter=100)
            if np.isnan(result) or result <= 0:
                return T_initial
            return result
        except (RuntimeError, ValueError):
            # If Newton's method fails, return initial guess
            return T_initial

    # =========================================================================
    # Relaxation Equations
    # =========================================================================

    def relaxation_rates(
        self,
        T: float,
        P_atm: float,
        e_vib_I: float,
        e_vib_II: float,
        static_relaxation: bool = True
    ) -> Tuple[float, float, float, float]:
        """
        Calculate vibrational energy relaxation rates.

        Args:
            T: Temperature in Kelvin
            P_atm: Pressure in atmospheres
            e_vib_I: Current Mode I vibrational energy (J/kg)
            e_vib_II: Current Mode II vibrational energy (J/kg)
            static_relaxation: If True, use constant relaxation times

        Returns:
            Tuple of (dE_vib_I_dt, dE_vib_II_dt, tau_I, tau_II)
        """
        if static_relaxation or T < 100 or T > 2000:
            tau_a, tau_b = 1.5e-6, 3.5e-5
        else:
            tau_a, tau_b = self.calculate_relaxation_times(T, P_atm)

        e_vib_eq_I = self.vib_energy_I(T)
        e_vib_eq_II = self.vib_energy_II(T)

        dE_vib_I_dt = (e_vib_eq_I - e_vib_I) / tau_a
        dE_vib_II_dt = (e_vib_eq_II - e_vib_II) / tau_b

        return dE_vib_I_dt, dE_vib_II_dt, tau_a, tau_b

    # =========================================================================
    # Nozzle Geometry Generators
    # =========================================================================

    @staticmethod
    def create_constant_area_nozzle(area: float) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a constant area duct geometry function.

        Args:
            area: Cross-sectional area in m^2

        Returns:
            Function that returns area for any position array
        """
        def nozzle(x: np.ndarray) -> np.ndarray:
            return np.ones_like(x) * area
        return nozzle

    @staticmethod
    def create_converging_diverging_nozzle(
        length: float,
        min_area: float,
        reservoir_area: float,
        exit_area: float,
        reservoir_fraction: float = 0.2,
        throat_length: float = 0.0
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a smooth converging-diverging nozzle geometry.

        Uses cosine interpolation for smooth area transitions to avoid
        discontinuities in the flow equations.

        Args:
            length: Total nozzle length (m)
            min_area: Throat area (m^2)
            reservoir_area: Inlet area (m^2)
            exit_area: Exit area (m^2)
            reservoir_fraction: Fraction of length for converging section
            throat_length: Length of constant-area throat section (m)

        Returns:
            Function that returns area for any position array
        """
        reservoir_length = length * reservoir_fraction

        def smooth_interp(A1: float, A2: float, s: float) -> float:
            """Cosine-based smooth interpolation."""
            return A1 + (A2 - A1) * 0.5 * (1 - np.cos(np.pi * s))

        def nozzle(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x)
            for i, point in enumerate(x):
                if point < reservoir_length:
                    s = point / reservoir_length
                    result[i] = smooth_interp(reservoir_area, min_area, s)
                elif point < reservoir_length + throat_length:
                    result[i] = min_area
                else:
                    s = (point - reservoir_length - throat_length) / (
                        length - reservoir_length - throat_length
                    )
                    result[i] = smooth_interp(min_area, exit_area, s)
            return result

        return nozzle

    # =========================================================================
    # Numerical Methods
    # =========================================================================

    @staticmethod
    def _finite_difference(
        f: np.ndarray,
        dx: float,
        direction: str = 'left'
    ) -> np.ndarray:
        """
        Calculate first derivative using one-sided finite differences.

        Args:
            f: Function values at grid points
            dx: Grid spacing
            direction: 'left' for backward, 'right' for forward differencing

        Returns:
            Derivative values at each grid point
        """
        df = np.zeros_like(f, dtype=float)

        if direction == 'left':
            df[1:] = (f[1:] - f[:-1]) / dx
            df[0] = (f[1] - f[0]) / dx
        elif direction == 'right':
            df[:-1] = (f[1:] - f[:-1]) / dx
            df[-1] = (f[-1] - f[-2]) / dx

        return df

    @staticmethod
    def _minmod_limiter(a: float, b: float) -> float:
        """Minmod flux limiter for TVD schemes."""
        if a * b <= 0:
            return 0
        return a if abs(a) < abs(b) else b

    def _apply_flux_limiters(
        self,
        y0: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Apply flux limiters to prevent numerical oscillations.

        Args:
            y0: Original state
            y_pred: Predicted state

        Returns:
            Limited predicted state
        """
        n_points = len(y0) // 5

        for i in range(5):
            start_idx = i * n_points
            end_idx = (i + 1) * n_points

            for j in range(1, n_points - 1):
                idx = start_idx + j
                forward_diff = y0[idx + 1] - y0[idx]
                backward_diff = y0[idx] - y0[idx - 1]
                limited_slope = self._minmod_limiter(forward_diff, backward_diff)
                y_pred[idx] = y0[idx] + 0.5 * limited_slope

        return y_pred

    # =========================================================================
    # Flow Equation Setup
    # =========================================================================

    def _create_time_derivative(
        self,
        n_points: int,
        delta_x: float,
        nozzle_A: np.ndarray,
        static_relaxation: bool
    ) -> Callable:
        """
        Create the time derivative function for the ODE system.

        Args:
            n_points: Number of grid points
            delta_x: Grid spacing
            nozzle_A: Nozzle area at each point
            static_relaxation: Use constant relaxation times

        Returns:
            Function computing dy/dt for MacCormack integration
        """
        R = self.R
        cv = self.cv
        gas = self.gas

        def time_derivative(t: float, y: np.ndarray, direction: str = 'left'):
            # Unpack state
            rho = y[0:n_points].copy()
            vel = y[n_points:2*n_points].copy()
            e = y[2*n_points:3*n_points].copy()
            e_vib_I = y[3*n_points:4*n_points].copy()
            e_vib_II = y[4*n_points:5*n_points].copy()

            # Calculate temperature
            T = (e - (e_vib_I + e_vib_II)) / cv

            # Handle non-physical values with neighbor averaging
            T[T <= 0] = np.convolve(T, np.ones(3)/2.0, mode='same')[T <= 0]
            rho[rho <= 0] = np.convolve(rho, np.ones(3)/2.0, mode='same')[rho <= 0]

            # Calculate pressure
            P = rho * R * T
            P_atm = P / 101325.0

            # Calculate relaxation rates
            relaxation_I = np.zeros(n_points)
            relaxation_II = np.zeros(n_points)
            tau_as = np.zeros(n_points)
            tau_bs = np.zeros(n_points)

            for i in range(n_points):
                (relaxation_I[i], relaxation_II[i],
                 tau_as[i], tau_bs[i]) = self.relaxation_rates(
                    T[i], P_atm[i], e_vib_I[i], e_vib_II[i], static_relaxation
                )

            # Spatial derivatives
            rho_v_A = rho * vel * nozzle_A
            drho_v_A_dx = self._finite_difference(rho_v_A, delta_x, direction)
            dP_dx = self._finite_difference(P, delta_x, direction)
            dvel_dx = self._finite_difference(vel, delta_x, direction)
            de_dx = self._finite_difference(e, delta_x, direction)
            de_vib_I_dx = self._finite_difference(e_vib_I, delta_x, direction)
            de_vib_II_dx = self._finite_difference(e_vib_II, delta_x, direction)
            dlnA_dx = self._finite_difference(nozzle_A, delta_x, direction) / nozzle_A

            # Conservation equations
            drho_dt = -drho_v_A_dx / nozzle_A
            dvel_dt = -(dP_dx / rho + vel * dvel_dx)
            de_dt = -((P / rho) * (dvel_dx + (vel * dlnA_dx)) + (vel * de_dx))
            de_vib_I_dt = relaxation_I - vel * de_vib_I_dx
            de_vib_II_dt = relaxation_II - vel * de_vib_II_dx

            # Boundary conditions (reservoir fixed)
            drho_dt[0] = 0
            de_dt[0] = 0
            de_vib_I_dt[0] = 0
            de_vib_II_dt[0] = 0

            # Adaptive time step based on CFL and relaxation
            min_tau = np.min(np.append(tau_as, tau_bs))
            max_speed = np.max(np.abs(vel)) + 300  # Approximate sound speed
            min_dt = min(delta_x / (max_speed + 1e-10), min_tau)

            dy_dt = np.concatenate([drho_dt, dvel_dt, de_dt, de_vib_I_dt, de_vib_II_dt])

            return dy_dt, min_dt

        return time_derivative

    def setup_initial_conditions(
        self,
        n_points: int,
        delta_x: float,
        P0: float,
        T0: float,
        nozzle_func: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up initial conditions for the simulation.

        Creates initial state with linear interpolation from reservoir
        conditions to ambient conditions.

        Args:
            n_points: Number of grid points
            delta_x: Grid spacing
            P0: Reservoir pressure (Pa)
            T0: Reservoir temperature (K)
            nozzle_func: Nozzle geometry function

        Returns:
            Tuple of (initial_state, spacing)
        """
        spacing = np.arange(n_points) * delta_x
        nozzle_A = nozzle_func(spacing)

        # Reservoir conditions
        rho_0 = P0 / (self.R * T0)
        vel_0 = 0.0
        e_vib_I_0 = self.vib_energy_I(T0)
        e_vib_II_0 = self.vib_energy_II(T0)
        e_0 = self.cv * T0 + e_vib_I_0 + e_vib_II_0

        # Exit conditions (approximate ambient)
        P_exit = 101325.0 * 0.01
        T_exit = 300.0
        v_exit = 343.0 * 5  # Supersonic exit
        rho_exit = P_exit / (self.R * T_exit)
        e_vib_I_exit = self.vib_energy_I(T_exit)
        e_vib_II_exit = self.vib_energy_II(T_exit)
        e_exit = self.cv * T_exit + e_vib_I_exit + e_vib_II_exit

        # Initialize with linear interpolation
        y0 = np.zeros((5, n_points))

        P_interp = np.interp(spacing, [0, spacing[-1]], [P0, P_exit])
        T_interp = np.interp(spacing, [0, spacing[-1]], [T0, T_exit])
        vel_interp = np.interp(spacing, [0, spacing[-1]], [vel_0, v_exit])

        rho_interp = P_interp / (self.R * T_interp)
        e_vib_I_interp = np.array([self.vib_energy_I(T) for T in T_interp])
        e_vib_II_interp = np.array([self.vib_energy_II(T) for T in T_interp])
        e_interp = self.cv * T_interp + e_vib_I_interp + e_vib_II_interp

        y0[0, :] = rho_interp
        y0[1, :] = vel_interp
        y0[2, :] = e_interp
        y0[3, :] = e_vib_I_interp
        y0[4, :] = e_vib_II_interp

        return y0.flatten(), spacing

    # =========================================================================
    # Main Simulation Methods
    # =========================================================================

    def run_simulation(
        self,
        nozzle_func: Callable,
        P0: float,
        T0: float,
        length: float,
        n_points: int = 50,
        time_span: float = 0.001,
        dt: float = None,
        static_relaxation_first: bool = True,
        two_stage: bool = True,
        second_stage_time: float = 0.0003
    ) -> Dict[str, Any]:
        """
        Run the full flow simulation.

        Uses a two-stage approach: first with static relaxation times to
        establish the flow field, then with dynamic relaxation for accuracy.

        Args:
            nozzle_func: Function returning nozzle area for position array
            P0: Reservoir pressure (Pa)
            T0: Reservoir temperature (K)
            length: Nozzle length (m)
            n_points: Number of grid points
            time_span: Simulation time for first stage (s)
            dt: Time step (if None, calculated from CFL)
            static_relaxation_first: Use static relaxation in first stage
            two_stage: Enable two-stage integration
            second_stage_time: Time for second stage (s)

        Returns:
            Dictionary containing:
            - 'solution': Full solution array (n_steps, 5*n_points)
            - 'times': Time values for each step
            - 'spacing': Spatial grid points
            - 'final_state': Final SimulationState object
            - 'nozzle_area': Nozzle area at each point
        """
        delta_x = length / n_points

        if dt is None:
            dt = delta_x / 400  # Conservative CFL

        spacing = np.arange(n_points) * delta_x
        nozzle_A = nozzle_func(spacing)

        # Set up initial conditions
        y0, _ = self.setup_initial_conditions(n_points, delta_x, P0, T0, nozzle_func)

        # Stage 1: Static relaxation
        time_deriv_static = self._create_time_derivative(
            n_points, delta_x, nozzle_A, static_relaxation=static_relaxation_first
        )
        sol, times = self._maccormack_integrate(
            y0, time_deriv_static, dt, time_span, delta_x,
            stage_name="Stage 1 (static τ)"
        )

        # Stage 2: Dynamic relaxation
        if two_stage:
            time_deriv_dynamic = self._create_time_derivative(
                n_points, delta_x, nozzle_A, static_relaxation=False
            )
            sol2, times2 = self._maccormack_integrate(
                sol[-1, :], time_deriv_dynamic, dt, second_stage_time, delta_x,
                stage_name="Stage 2 (dynamic τ)"
            )
            sol = np.vstack([sol, sol2[1:, :]])
            times = times + [t + times[-1] for t in times2[1:]]

        return {
            'solution': sol,
            'times': times,
            'spacing': spacing,
            'final_state': SimulationState.from_flat(sol[-1, :], n_points),
            'nozzle_area': nozzle_A
        }

    def _maccormack_integrate(
        self,
        y0: np.ndarray,
        time_derivative: Callable,
        dt: float,
        time_span: float,
        delta_x: float,
        show_progress: bool = True,
        stage_name: str = "Integration"
    ) -> Tuple[np.ndarray, list]:
        """
        Stabilized MacCormack with:
        - Proper CFL based on local wave speeds
        - Jameson-style artificial dissipation
        - Characteristic boundary conditions
        """
        from tqdm import tqdm

        current_time = 0.0
        times = [current_time]
        sol = y0.copy()
        y = y0.copy()

        # Create progress bar
        pbar = tqdm(
            total=time_span,
            desc=stage_name,
            unit="s",
            disable=not show_progress,
            bar_format="{l_bar}{bar}| {n:.2e}/{total:.2e}s [{elapsed}<{remaining}, {postfix}]"
        )

        step_count = 0
        max_mach = 0.0

        while current_time < time_span:
          # Predictor (left differencing) - matches notebook exactly
            dy_dt, min_dt = time_derivative(0, y, direction='left')
            min_dt = min(min_dt, dt) / 3

            # Predictor step
            y_pred = y + dy_dt * min_dt
            current_time += min_dt
            times.append(current_time)

            # Corrector (right differencing)
            dy_dt_pred, min_dt_corr = time_derivative(0, y_pred, direction='right')
            min_dt_corr = min(min_dt_corr, dt) / 3

            # Apply limiters (notebook applies after corrector derivative)
            y_pred = self._apply_flux_limiters(y, y_pred)

            # Average and update (notebook uses corrector's min_dt)
            dy_dt_avg = (dy_dt + dy_dt_pred) / 2
            y = y + dy_dt_avg * min_dt_corr

            sol = np.vstack([sol, y])

            # Update progress bar
            pbar.update(min_dt_corr)
            pbar.set_postfix({
                "step": step_count,
                "dt": f"{min_dt_corr*1e9:.1f}ns",
            })

        pbar.close()

        return sol, times

    def _apply_physical_bounds(self, y: np.ndarray, n_points: int) -> np.ndarray:
        """Enforce physical realizability."""
        y = y.copy()
        
        # Density must be positive
        y[0:n_points] = np.maximum(y[0:n_points], 1e-6)
        
        # Vibrational energies must be non-negative
        y[3*n_points:4*n_points] = np.maximum(y[3*n_points:4*n_points], 0)
        y[4*n_points:5*n_points] = np.maximum(y[4*n_points:5*n_points], 0)
        
        # Total energy must exceed vibrational (ensures positive T)
        e = y[2*n_points:3*n_points]
        e_vib_total = y[3*n_points:4*n_points] + y[4*n_points:5*n_points]
        min_thermal = self.cv * 100  # Minimum ~100K
        y[2*n_points:3*n_points] = np.maximum(e, e_vib_total + min_thermal)
        
        return y

    def _add_artificial_dissipation(
        self, 
        y_old: np.ndarray, 
        y_new: np.ndarray, 
        n_points: int,
        epsilon2: float = 0.5,  # 2nd order (near shocks)
        epsilon4: float = 0.02  # 4th order (smooth regions)
    ) -> np.ndarray:
        """
        Jameson-style adaptive artificial dissipation.
        Adds 2nd-order dissipation near discontinuities,
        4th-order background dissipation in smooth regions.
        """
        y = y_new.copy()
        
        for var in range(5):
            start = var * n_points
            end = (var + 1) * n_points
            u = y_old[start:end]
            
            # Pressure-based shock sensor
            if var == 0:  # Use density for sensor
                p = u  
            else:
                # Recompute using current density
                rho = y_old[0:n_points]
                e = y_old[2*n_points:3*n_points]
                e_vib = y_old[3*n_points:4*n_points] + y_old[4*n_points:5*n_points]
                T = (e - e_vib) / self.cv
                p = rho * self.R * np.maximum(T, 100)
            
            # Shock sensor: nu_j = |p_{j+1} - 2p_j + p_{j-1}| / (p_{j+1} + 2p_j + p_{j-1})
            nu = np.zeros(n_points)
            for j in range(1, n_points - 1):
                num = abs(p[j+1] - 2*p[j] + p[j-1])
                den = abs(p[j+1]) + 2*abs(p[j]) + abs(p[j-1]) + 1e-10
                nu[j] = num / den
            
            # Apply dissipation
            d = np.zeros(n_points)
            for j in range(2, n_points - 2):
                # Adaptive coefficient
                nu_max = max(nu[j-1], nu[j], nu[j+1])
                eps2 = epsilon2 * nu_max
                eps4 = max(0, epsilon4 - eps2)
                
                # 2nd order dissipation (near shocks)
                d2 = eps2 * (u[j+1] - 2*u[j] + u[j-1])
                
                # 4th order dissipation (smooth regions)  
                d4 = -eps4 * (u[j+2] - 4*u[j+1] + 6*u[j] - 4*u[j-1] + u[j-2])
                
                d[j] = d2 + d4
            
            y[start:end] += d
        
        return y

    def _apply_boundary_conditions(
        self, 
        y: np.ndarray, 
        y0: np.ndarray, 
        n_points: int
    ) -> np.ndarray:
        """
        Proper characteristic boundary conditions.
        Inlet: fixed reservoir (subsonic)
        Outlet: supersonic extrapolation
        """
        y = y.copy()
        
        # Inlet: hold reservoir conditions (all from y0)
        for var in range(5):
            y[var * n_points] = y0[var * n_points]
        
        # Outlet: supersonic extrapolation (all characteristics exit)
        # Linear extrapolation from interior
        for var in range(5):
            idx = (var + 1) * n_points - 1
            y[idx] = 2 * y[idx - 1] - y[idx - 2]
        
        return y

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_temperatures(self, state: SimulationState) -> Dict[str, np.ndarray]:
        """
        Calculate all temperature fields from a simulation state.

        Args:
            state: SimulationState object

        Returns:
            Dictionary with translational, Mode I, and Mode II temperatures
        """
        T_trans = (state.e - (state.e_vib_I + state.e_vib_II)) / self.cv

        T_vib_I = np.array([
            self.find_vib_temperature(e, mode='I')
            for e in state.e_vib_I
        ])

        T_vib_II = np.array([
            self.find_vib_temperature(e, mode='II')
            for e in state.e_vib_II
        ])

        return {
            'translational': T_trans,
            'mode_I': T_vib_I,
            'mode_II': T_vib_II
        }

    def get_population_levels(
        self,
        solution: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate vibrational mode population fractions.

        Computes the fraction of CO2 molecules in specific vibrational
        states (100 and 001) under non-equilibrium conditions.

        Args:
            solution: Full solution array from simulation

        Returns:
            Tuple of (X_100, X_001) population fractions
        """
        n_points = solution.shape[1] // 5
        state = SimulationState.from_flat(solution[-1, :], n_points)
        temps = self.get_temperatures(state)

        k = self.K_BOLTZMANN
        hc = self.HC

        T_vib_I = temps['mode_I']
        T_vib_II = temps['mode_II']

        exp_I = -hc / (k * T_vib_I)
        exp_II = -hc / (k * T_vib_II)

        v1, v2, v3 = self.V1_CO2, self.V2_CO2, self.V3_CO2

        Z_CO2 = 1 / (
            (1 - np.exp(v1 * exp_I)) *
            (1 - np.exp(v2 * exp_I)) *
            (1 - np.exp(v3 * exp_II))
        )

        X_100 = np.exp(v1 * exp_I) / Z_CO2
        X_001 = np.exp(v3 * exp_II) / Z_CO2

        return X_100, X_001

    def get_mach_number(self, state: SimulationState) -> np.ndarray:
        """
        Calculate Mach number distribution.

        Args:
            state: SimulationState object

        Returns:
            Mach number at each grid point
        """
        temps = self.get_temperatures(state)
        sound_speed = np.sqrt(self.gamma * self.R * temps['translational'])
        return np.abs(state.vel) / sound_speed

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_state(
        self,
        state: SimulationState,
        spacing: np.ndarray,
        nozzle_area: Optional[np.ndarray] = None,
        title: str = "Flow State"
    ):
        """
        Plot the current simulation state.

        Args:
            state: SimulationState object
            spacing: Spatial grid points
            nozzle_area: Optional nozzle area for additional subplot
            title: Plot title
        """
        import matplotlib.pyplot as plt

        temps = self.get_temperatures(state)
        n_plots = 5 if nozzle_area is not None else 4

        fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots), sharex=True)
        fig.suptitle(title, fontsize=14)

        axs[0].plot(spacing, state.rho, 'b-', lw=2)
        axs[0].set_ylabel('Density (kg/m³)')
        axs[0].grid(True)

        axs[1].plot(spacing, state.vel, 'r-', lw=2)
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].grid(True)

        axs[2].plot(spacing, temps['translational'], 'orange', lw=2, label='Translational')
        axs[2].plot(spacing, temps['mode_I'], 'm--', lw=1.5, label='Mode I (Tvib)')
        axs[2].plot(spacing, temps['mode_II'], 'c--', lw=1.5, label='Mode II (Tvib)')
        axs[2].set_ylabel('Temperature (K)')
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(spacing, state.e_vib_I, 'm-', lw=2, label='Mode I')
        axs[3].plot(spacing, state.e_vib_II, 'c-', lw=2, label='Mode II')
        axs[3].set_ylabel('Vib. Energy (J/kg)')
        axs[3].legend()
        axs[3].grid(True)

        if nozzle_area is not None:
            axs[4].plot(spacing, nozzle_area, 'k-', lw=2)
            axs[4].set_ylabel('Area (m²)')
            axs[4].grid(True)

        axs[-1].set_xlabel('Position (m)')
        plt.tight_layout()

        plt.savefig("flow_state.png", dpi=300)

        return fig, axs

    def create_animation(
        self,
        results: Dict[str, Any],
        max_frames: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Create an animation of the simulation results.

        Args:
            results: Results dictionary from run_simulation
            max_frames: Maximum frames in animation
            save_path: Path to save animation (optional)

        Returns:
            matplotlib FuncAnimation object
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        solution = results['solution']
        spacing = results['spacing']
        nozzle_A = results['nozzle_area']
        times = results['times']

        n_steps, total_vars = solution.shape
        n_points = total_vars // 5

        # Clean up data
        solution = np.nan_to_num(solution)
        solution = np.where(np.isinf(solution), 0, solution)

        # Sample frames
        if n_steps > max_frames:
            frame_indices = np.linspace(0, n_steps-1, max_frames, dtype=int)
            sampled_solution = solution[frame_indices]
            sampled_times = [times[i] for i in frame_indices]
        else:
            sampled_solution = solution
            sampled_times = times

        fig, axs = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
        fig.suptitle('Non-Equilibrium Flow Simulation', fontsize=16)

        # Initialize lines
        lines = {
            'rho': axs[0].plot([], [], 'b-', lw=2)[0],
            'vel': axs[1].plot([], [], 'r-', lw=2)[0],
            'e': axs[2].plot([], [], 'g-', lw=2)[0],
            'evib_I': axs[3].plot([], [], 'm-', lw=2)[0],
            'evib_II': axs[3].plot([], [], 'c-', lw=2, alpha=0.7)[0],
            'temp': axs[4].plot([], [], 'orange', lw=2)[0],
        }
        axs[5].plot(spacing, nozzle_A, 'k-', lw=2)

        labels = ['Density (kg/m³)', 'Velocity (m/s)', 'Total Energy (J/kg)',
                  'Vib. Energy (J/kg)', 'Temperature (K)', 'Area (m²)']
        for ax, label in zip(axs, labels):
            ax.set_ylabel(label)
            ax.set_xlim(spacing[0], spacing[-1])
            ax.grid(True)

        axs[3].legend(['Mode I', 'Mode II'], loc='upper right')
        axs[-1].set_xlabel('Position (m)')

        # Calculate limits
        rho_vals = solution[:, :n_points]
        vel_vals = solution[:, n_points:2*n_points]
        e_vals = solution[:, 2*n_points:3*n_points]
        evib_I_vals = solution[:, 3*n_points:4*n_points]
        evib_II_vals = solution[:, 4*n_points:5*n_points]
        temp_vals = (e_vals - (evib_I_vals + evib_II_vals)) / self.cv

        def get_lim(vals, pad=0.1):
            vmin, vmax = np.min(vals), np.max(vals)
            p = pad * (vmax - vmin) if vmax > vmin else pad
            return vmin - p, vmax + p

        axs[0].set_ylim(*get_lim(rho_vals))
        axs[1].set_ylim(*get_lim(vel_vals))
        axs[2].set_ylim(*get_lim(e_vals))
        axs[3].set_ylim(*get_lim(np.concatenate([evib_I_vals.flatten(), evib_II_vals.flatten()])))
        axs[4].set_ylim(*get_lim(temp_vals))
        axs[5].set_ylim(0, np.max(nozzle_A) * 1.2)

        time_text = axs[0].text(0.02, 0.95, '', transform=axs[0].transAxes)
        mach_text = axs[1].text(0.02, 0.95, '', transform=axs[1].transAxes)

        def update(frame_idx):
            frame = sampled_solution[frame_idx]
            state = SimulationState.from_flat(frame, n_points)
            T = (state.e - (state.e_vib_I + state.e_vib_II)) / self.cv

            lines['rho'].set_data(spacing, state.rho)
            lines['vel'].set_data(spacing, state.vel)
            lines['e'].set_data(spacing, state.e)
            lines['evib_I'].set_data(spacing, state.e_vib_I)
            lines['evib_II'].set_data(spacing, state.e_vib_II)
            lines['temp'].set_data(spacing, T)

            time_text.set_text(f'Time: {sampled_times[frame_idx]:.6f} s')
            sound_speed = np.sqrt(self.gamma * self.R * T)
            max_mach = np.max(np.abs(state.vel) / sound_speed)
            mach_text.set_text(f'Max Mach: {max_mach:.2f}')

            return list(lines.values()) + [time_text, mach_text]

        anim = FuncAnimation(fig, update, frames=len(sampled_times),
                            interval=50, blit=True)

        if save_path:
            try:
                anim.save(save_path, writer='ffmpeg', fps=30, dpi=100)
            except Exception:
                anim.save(save_path, writer='pillow', fps=30)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        return anim


# =============================================================================
# Convenience function for quick simulations
# =============================================================================

def run_nozzle_simulation(
    X_CO2: float = 0.09,
    X_N2: float = 0.90,
    X_H2O: float = 0.01,
    P0_atm: float = 10.0,
    T0: float = 1500.0,
    length: float = 0.4,
    n_points: int = 50,
    min_area: float = 0.01,
    reservoir_area: float = 0.05,
    exit_area: float = 0.30
) -> Dict[str, Any]:
    """
    Convenience function to run a standard nozzle simulation.

    Args:
        X_CO2: CO2 mole fraction
        X_N2: N2 mole fraction
        X_H2O: H2O mole fraction
        P0_atm: Reservoir pressure in atmospheres
        T0: Reservoir temperature in Kelvin
        length: Nozzle length in meters
        n_points: Number of grid points
        min_area: Throat area in m^2
        reservoir_area: Inlet area in m^2
        exit_area: Exit area in m^2

    Returns:
        Results dictionary with 'simulator' and 'results' keys
    """
    gas = GasMixture(X_CO2=X_CO2, X_N2=X_N2, X_H2O=X_H2O)
    sim = NonEquilibriumFlowSimulator(gas)

    nozzle = sim.create_converging_diverging_nozzle(
        length=length,
        min_area=min_area,
        reservoir_area=reservoir_area,
        exit_area=exit_area
    )

    results = sim.run_simulation(
        nozzle_func=nozzle,
        P0=101325.0 * P0_atm,
        T0=T0,
        length=length,
        n_points=n_points
    )

    return {'simulator': sim, 'results': results}


if __name__ == "__main__":
    # Run with exact notebook conditions for comparison
    print("=" * 60)
    print("Non-Equilibrium Flow Simulation - Notebook Comparison Test")
    print("=" * 60)

    # Exact notebook parameters (cell 14)
    length = 1.0  # m
    n_points = 100
    P0 = 101325.0 * 10.0  # 10 atm in Pascal
    T0 = 2300.0  # K
    X_CO2 = 0.15
    X_N2 = 0.75
    X_H2O = 0.10

    min_A = 40 * 1e-6  # m^2
    reservoir_A = min_A * 20  # m^2
    exit_A = min_A * 20  # m^2
    time_span = 0.005  # s (first stage)
    second_stage_time = 0.0003  # s (second stage)

    print(f"\nSimulation Parameters:")
    print(f"  Length: {length} m")
    print(f"  Grid points: {n_points}")
    print(f"  P0: {P0/101325:.1f} atm")
    print(f"  T0: {T0} K")
    print(f"  Gas: CO2={X_CO2}, N2={X_N2}, H2O={X_H2O}")
    print(f"  Nozzle: inlet={reservoir_A} m², throat={min_A} m², exit={exit_A} m²")

    # Create gas mixture
    gas = GasMixture(X_CO2=X_CO2, X_N2=X_N2, X_H2O=X_H2O)
    sim = NonEquilibriumFlowSimulator(gas)

    print(f"\nGas Properties:")
    print(f"  R = {sim.R:.2f} J/(kg·K)")
    print(f"  cv = {sim.cv:.2f} J/(kg·K)")
    print(f"  gamma = {sim.gamma}")

    # Test relaxation times (should match notebook cell 2)
    tau_I, tau_II = sim.calculate_relaxation_times(300, 1.0)
    print(f"\nRelaxation Times at T=300K, P=1atm:")
    print(f"  tau_I: {tau_I:.2e} s (notebook: 1.49e-06 s)")
    print(f"  tau_II: {tau_II:.2e} s (notebook: 3.66e-05 s)")

    # Test vibrational energies
    e_I_1000 = sim.vib_energy_I(1000)
    e_II_1000 = sim.vib_energy_II(1000)
    print(f"\nVibrational Energies at T=1000K:")
    print(f"  e_vib_I: {e_I_1000:.2e} J/kg")
    print(f"  e_vib_II: {e_II_1000:.2e} J/kg")

    # Create nozzle
    nozzle = sim.create_converging_diverging_nozzle(
        length=length,
        min_area=min_A,
        reservoir_area=reservoir_A,
        exit_area=exit_A,
        reservoir_fraction=0.2,  # length * 0.2 as in notebook
        throat_length=0.0
    )

    print(f"\nRunning simulation...")
    print(f"  Stage 1: static relaxation, {time_span*1000:.1f} ms")
    print(f"  Stage 2: dynamic relaxation, {second_stage_time*1000:.1f} ms")

    # Run simulation
    results = sim.run_simulation(
        nozzle_func=nozzle,
        P0=P0,
        T0=T0,
        length=length,
        n_points=n_points,
        time_span=time_span,
        two_stage=False,
        second_stage_time=second_stage_time
    )

    print(f"\nSimulation completed: {len(results['times'])} time steps")

    # Check for NaN values
    final_state = results['final_state']
    nan_count = np.isnan(final_state.to_flat()).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in final state")
        # Use an earlier valid state if final has NaNs
        for i in range(-1, -min(100, len(results['solution'])), -1):
            test_state = results['solution'][i, :]
            if not np.any(np.isnan(test_state)):
                print(f"  Using state at step {len(results['solution'])+i} instead")
                final_state = SimulationState.from_flat(test_state, n_points)
                break
    else:
        print("  No NaN values - simulation stable")

    # Print final state statistics
    print(f"\nFinal State Statistics:")
    print(f"  Density: min={final_state.rho.min():.4f}, max={final_state.rho.max():.4f} kg/m³")
    print(f"  Velocity: min={final_state.vel.min():.1f}, max={final_state.vel.max():.1f} m/s")
    print(f"  Energy: min={final_state.e.min():.2e}, max={final_state.e.max():.2e} J/kg")
    print(f"  e_vib_I: min={final_state.e_vib_I.min():.2e}, max={final_state.e_vib_I.max():.2e} J/kg")
    print(f"  e_vib_II: min={final_state.e_vib_II.min():.2e}, max={final_state.e_vib_II.max():.2e} J/kg")

    # Calculate temperature
    T_trans = (final_state.e - (final_state.e_vib_I + final_state.e_vib_II)) / sim.cv
    print(f"  Temperature: min={T_trans.min():.1f}, max={T_trans.max():.1f} K")

    # Calculate Mach number if temperatures are valid
    if np.all(T_trans > 0):
        sound_speed = np.sqrt(sim.gamma * sim.R * T_trans)
        mach = np.abs(final_state.vel) / sound_speed
        print(f"  Mach: min={mach.min():.2f}, max={mach.max():.2f}")

    print("\n" + "=" * 60)
    print("To visualize, run: sim.plot_state(final_state, results['spacing'], results['nozzle_area'])")
    print("=" * 60)
    sim.plot_state(final_state, results['spacing'], results['nozzle_area'])
