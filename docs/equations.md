# Model equations

The internal unit system is `pc`, `Myr`, and `Msun`; configuration values carry
explicit Astropy-compatible units.

For positions relative to the soliton centre,

\[
\mathbf D=\mathbf r_1-\mathbf r_2,\qquad
\mathbf w_i=\mathbf v_i-\mathbf u_{\rm FDM}.
\]

The default soliton profile is

\[
\rho(r)=\rho_0[1+0.091(r/r_c)^2]^{-8}.
\]

The configured soliton mass is normalized according to
`mass_definition = total_profile | within_rc`; these definitions are not
interchangeable.

The orbit-averaged wave-drag force is

\[
\mathbf F_{{\rm DF},i}=-4\pi G^2M_i^2\rho(r_i)C(q_i)
\frac{\mathbf w_i}{|\mathbf w_i|^3},
\]

with

\[
q_i=\frac{m_{\rm FDM}|\mathbf w_i|r_{{\rm eff},i}}{\hbar},\qquad
r_{\rm eff}=\alpha_{\rm DF}\min(D/2,r_c),
\]

and

\[
C(q)=\operatorname{Cin}(2q)+\frac{\sin(2q)}{2q}-1.
\]

For small `q`, the implementation evaluates

\[
C(q)=q^2/3-q^4/30+2q^6/945+O(q^8)
\]

to avoid cancellation.

The static-background diagnostic ledger is

\[
E_{\rm budget}=E_{\rm mech}+E_{\rm FDM},\qquad
E_{\rm FDM}=-\int\sum_i\mathbf F_{{\rm DF},i}\cdot\mathbf w_i\,dt.
\]

Reaching `0.01 pc` counts as success only on an inward crossing and when the
two-body osculating orbit is bound.
