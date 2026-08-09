"""Physical constants in the package's fixed internal unit system."""

from astropy import units as u
from astropy.constants import G, c, hbar

# Internal base units are pc, Myr, and Msun.
G_INTERNAL = G.to_value(u.pc**3 / (u.Msun * u.Myr**2))
C_PC_PER_MYR = c.to_value(u.pc / u.Myr)
KM_S_TO_PC_MYR = (1.0 * u.km / u.s).to_value(u.pc / u.Myr)

# q = Q_PER_EV_PC2_PER_MYR * m[eV/c^2] * v[pc/Myr] * r[pc].
# FDM particle masses conventionally quoted in eV are interpreted as rest
# energies and converted to inertial masses through E = mc^2.
Q_PER_EV_PC2_PER_MYR = (
    (1.0 * u.eV / c**2) * (1.0 * u.pc / u.Myr) * (1.0 * u.pc) / hbar
).decompose().value

SCHIVE_A = 0.091
