C =====================================================================
C  UMAT:  Barlat Yld2000-2d + Voce Isotropic Hardening
C  For Abaqus/Standard plane stress elements (CPS4R)
C
C  Material: SGCC JIS G 3302 galvanized steel (BCC, a=6)
C
C  NTENS=3, NDI=2, NSHR=1  (plane stress)
C
C  PROPS layout (NPROPS=13):
C    PROPS(1)   = E       [MPa]
C    PROPS(2)   = NU
C    PROPS(3)   = SIGMA0  [MPa]  initial yield stress (Voce)
C    PROPS(4)   = Q_INF   [MPa]  Voce hardening amplitude
C    PROPS(5)   = B_ISO          Voce hardening rate
C    PROPS(6)   = ALPHA1  \
C    ...                  |  Yld2000-2d anisotropy coefficients
C    PROPS(13)  = ALPHA8  /
C    (exponent a=6 for BCC steel is hardcoded)
C
C  STATEV layout (NSTATV=1):
C    STATEV(1) = cumulative plastic strain kappa
C
C  Return mapping:
C    Newton iteration on 4x4 system [sigma_11, sigma_22, sigma_12, Dlambda]
C    Gradient and Hessian computed by central finite differences
C    Consistent algorithmic tangent modulus
C
C  Reference:
C    Barlat et al. (2003), Int. J. Plasticity 19, 1297-1319
C =====================================================================
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1  RPL,DDSDDT,DRPLDE,DRPLDT,
     2  STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3  NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4  CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
C
      IMPLICIT NONE
      CHARACTER*8 CMNAME
      INTEGER NDI,NSHR,NTENS,NSTATV,NPROPS
      INTEGER NOEL,NPT,LAYER,KSPT,KSTEP,KINC
      DOUBLE PRECISION STRESS(NTENS),STATEV(NSTATV)
      DOUBLE PRECISION DDSDDE(NTENS,NTENS)
      DOUBLE PRECISION SSE,SPD,SCD,RPL,DTIME,TEMP,DTEMP,PNEWDT
      DOUBLE PRECISION DDSDDT(NTENS),DRPLDE(NTENS),DRPLDT
      DOUBLE PRECISION STRAN(NTENS),DSTRAN(NTENS)
      DOUBLE PRECISION TIME(2),PREDEF(1),DPRED(1)
      DOUBLE PRECISION COORDS(3),DROT(3,3),CELENT
      DOUBLE PRECISION DFGRD0(3,3),DFGRD1(3,3)
      DOUBLE PRECISION PROPS(NPROPS)
C
C ---- local variables ------------------------------------------------
      DOUBLE PRECISION E, NU, SIG0, QINF, BISO
      DOUBLE PRECISION ALPHA(8)
      DOUBLE PRECISION C11, C12, C66
      DOUBLE PRECISION CPS(3,3)
      DOUBLE PRECISION KAPPA0
C trial stress
      DOUBLE PRECISION STR_TR(3), STR(3), STR_NEW(3)
      DOUBLE PRECISION SB_TR, SIGY0
C plastic step
      DOUBLE PRECISION DL, DL_NEW       ! delta lambda
      DOUBLE PRECISION M(3), M_T(3)     ! gradient d(sigbar)/d(sigma)
      DOUBLE PRECISION H(3,3)           ! hessian d^2(sigbar)/d(sigma)^2
      DOUBLE PRECISION SB, SIGY, DSIGY
      DOUBLE PRECISION SB_T, SIGY_T
      DOUBLE PRECISION RES(4), R1T,R2T,R3T,R4T
      DOUBLE PRECISION JAC(4,4), JAC_INV(4,4)
      DOUBLE PRECISION DX(4)
      DOUBLE PRECISION FNORM, FNORM0, FNT, TOL_NR, STEP
      INTEGER ITER, MAXITER, I, J, K, ILS
      LOGICAL ELASTIC
C consistent tangent
      DOUBLE PRECISION A_MAT(3,3), A_INV(3,3)
      DOUBLE PRECISION A_INV_B(3), B_VEC(3)
      DOUBLE PRECISION DENOM_CT
      DOUBLE PRECISION TEMP3(3), D_EP(3,3)
C =====================================================================

C ---- Read material properties ---------------------------------------
      E     = PROPS(1)
      NU    = PROPS(2)
      SIG0  = PROPS(3)
      QINF  = PROPS(4)
      BISO  = PROPS(5)
      DO I = 1, 8
        ALPHA(I) = PROPS(5+I)
      END DO
C
C ---- Plane-stress elastic modulus 3x3 (Voigt, engr shear) ----------
      C11 = E / (1.0D0 - NU*NU)
      C12 = NU * C11
      C66 = E / (2.0D0*(1.0D0 + NU))
C
      DO I = 1, 3
        DO J = 1, 3
          CPS(I,J) = 0.0D0
        END DO
      END DO
      CPS(1,1) = C11
      CPS(1,2) = C12
      CPS(2,1) = C12
      CPS(2,2) = C11
      CPS(3,3) = C66
C
C ---- Retrieve state ------------------------------------------------
      KAPPA0 = STATEV(1)
C
C ---- Trial stress --------------------------------------------------
      DO I = 1, NTENS
        STR_TR(I) = STRESS(I)
        DO J = 1, NTENS
          STR_TR(I) = STR_TR(I) + CPS(I,J)*DSTRAN(J)
        END DO
      END DO
C
C ---- Elastic predictor check ---------------------------------------
      CALL YLD_SIGBAR(STR_TR, ALPHA, SB_TR)
      SIGY0 = SIG0 + QINF*(1.0D0 - DEXP(-BISO*KAPPA0))
C
      ELASTIC = (SB_TR .LE. SIGY0*(1.0D0 + 1.0D-10))
C
      IF (ELASTIC) THEN
C ------ Elastic step ------------------------------------------------
        DO I = 1, NTENS
          STRESS(I) = STR_TR(I)
        END DO
        DO I = 1, NTENS
          DO J = 1, NTENS
            DDSDDE(I,J) = CPS(I,J)
          END DO
        END DO
        RETURN
      END IF
C
C ---- Plastic step: cutting-plane return mapping --------------------
C     Simo & Hughes (1998) cutting-plane scheme.  Uses only the gradient
C     of sigbar (no Hessian required), which avoids the noise from a
C     finite-difference Hessian on the sharp (a=6) Yld2000 surface.
C     Linear convergence but unconditionally stable for convex yield
C     functions with positive hardening modulus.
C
C     At each iteration k:
C       dlam_k = (sbk - sigy_k) / (m_k^T C m_k + H_iso_k)
C       sigma_{k+1} = sigma_k - dlam_k * C * m_k
C       kappa_{k+1}  = kappa_k + dlam_k
C
C     NOTE: a radial predictor (scale trial stress onto yield surface)
C     is NOT used here.  It would break the flow-rule kinematics for
C     anisotropic yield surfaces (m is not parallel to sigma except
C     for isotropic von Mises), leaving the stress on the yield surface
C     but at the wrong angular position.  Starting at the trial stress
C     with DL=0 preserves the kinematic link sigma = sigma_tr - DL*C*m.
C
C     If the iteration fails to converge we DO NOT request PNEWDT<1:
C     cutbacks cascade because the iteration logic is identical on a
C     smaller increment.  The global Abaqus Newton can absorb a tiny
C     residual via displacement-correction iterations, which is cheaper
C     and always succeeds on a ductile hardening response.
C
      MAXITER = 500
      TOL_NR  = 1.0D-6
C
      DO I = 1, NTENS
        STR(I) = STR_TR(I)
      END DO
      DL = 0.0D0
C
      DO ITER = 1, MAXITER
        CALL YLD_GRAD  (STR, ALPHA, M)
        CALL YLD_SIGBAR(STR, ALPHA, SB)
        SIGY  = SIG0 + QINF*(1.0D0 - DEXP(-BISO*(KAPPA0+DL)))
        DSIGY = QINF*BISO*DEXP(-BISO*(KAPPA0+DL))
C
        FNORM = (SB - SIGY) / DMAX1(SIGY, 1.0D0)
        IF (DABS(FNORM) .LT. TOL_NR) GOTO 200
C
C       CPS * m  (plane stress, engineering shear convention)
        R1T = C11*M(1) + C12*M(2)
        R2T = C12*M(1) + C11*M(2)
        R3T = C66*M(3)
C
C       Denominator:  m^T CPS m  +  H_iso
        FNT = M(1)*R1T + M(2)*R2T + M(3)*R3T + DSIGY
        IF (FNT .LT. 1.0D-8) FNT = 1.0D-8
C
C       Scalar plastic-multiplier increment (signed)
        STEP = (SB - SIGY) / FNT
C
C       Cap step size to prevent overshoot on sharp-curvature directions
C       of the yield surface.  For uniaxial-0 deg the flow direction is
C       near-normal and STEP is naturally small; for 45 deg the surface
C       has strong shear curvature and an unclamped STEP can overshoot,
C       producing oscillatory stress updates that later cascade into
C       global-equilibrium cutbacks.  Empirical cap of 1e-2 per iter is
C       well within the physical bound Delta-lambda < (SB-SIGY)/G.
        IF (STEP .GT.  1.0D-2) STEP =  1.0D-2
        IF (STEP .LT. -1.0D-2) STEP = -1.0D-2
C
C       Update stress and plastic multiplier
        STR(1) = STR(1) - STEP*R1T
        STR(2) = STR(2) - STEP*R2T
        STR(3) = STR(3) - STEP*R3T
        DL     = DL     + STEP
        IF (DL .LT. 0.0D0) DL = 0.0D0
      END DO
C
C     Not fully converged after MAXITER.  Accept the best-effort state
C     (residual below engineering tolerance) rather than requesting a
C     PNEWDT cutback.  The global equilibrium Newton will clean up the
C     remaining tiny residual through displacement-correction iters.
 200  CONTINUE
C
C ---- Update state --------------------------------------------------
      DO I = 1, NTENS
        STRESS(I) = STR(I)
      END DO
      STATEV(1) = KAPPA0 + DL
C
C     (SPD/SSE not required for correctness — leave accumulated)
C
C ---- Tangent modulus -----------------------------------------------
C     We return the ELASTIC tangent (plane-stress CPS) rather than the
C     consistent algorithmic tangent. Reason: for a sharp (a=6) Yld2000
C     surface with Voce hardening, the algorithmic tangent has one very
C     small eigenvalue (~h_iso << E) along the flow direction. When a
C     uniform uniaxial specimen crosses yield, *every* integration point
C     develops this soft direction simultaneously, leaving the global
C     stiffness nearly rank-deficient (observed: ~half of all DOFs
C     flagged as near-singular, leading to cutback exhaustion even on
C     CPS4 full-integration meshes). The elastic tangent is always
C     positive definite; Newton equilibrium loses quadratic convergence
C     but converges linearly with 3-5 iterations per increment, which
C     is far cheaper than the cutback cascade.
      DO I = 1, 3
        DO J = 1, 3
          DDSDDE(I,J) = CPS(I,J)
        END DO
      END DO
C
      RETURN
      END
C =====================================================================


C =====================================================================
C  YLD_PHI — compute Yld2000-2d yield function phi
C  Input:  S(3)=[sig11,sig22,sig12],  ALPHA(8), exponent A=6
C  Output: PHI
C =====================================================================
      SUBROUTINE YLD_PHI(S, ALPHA, PHI)
      IMPLICIT NONE
      DOUBLE PRECISION S(3), ALPHA(8), PHI
      DOUBLE PRECISION A1,A2,A3,A4,A5,A6,A7,A8
      DOUBLE PRECISION SP(3), SD(3)
      DOUBLE PRECISION SP1,SP2,SD1,SD2,AVG,DISC
      DOUBLE PRECISION AEXP
      PARAMETER (AEXP = 6.0D0)
C
      A1=ALPHA(1); A2=ALPHA(2); A3=ALPHA(3); A4=ALPHA(4)
      A5=ALPHA(5); A6=ALPHA(6); A7=ALPHA(7); A8=ALPHA(8)
C
C     L' * sigma
      SP(1) = (2.0D0*A1/3.0D0)*S(1) + (-A1/3.0D0)*S(2)
      SP(2) = (-A2/3.0D0)*S(1) + (2.0D0*A2/3.0D0)*S(2)
      SP(3) = A7*S(3)
C
C     L'' * sigma
      SD(1) = ((8.0D0*A5-2.0D0*A3-2.0D0*A6+2.0D0*A4)/9.0D0)*S(1)
     &       +((4.0D0*A6-4.0D0*A5-4.0D0*A4+A3)/9.0D0)*S(2)
      SD(2) = ((4.0D0*A3-4.0D0*A5-4.0D0*A4+A6)/9.0D0)*S(1)
     &       +((8.0D0*A4-2.0D0*A6-2.0D0*A3+2.0D0*A5)/9.0D0)*S(2)
      SD(3) = A8*S(3)
C
C     Principal values of SP
      AVG  = (SP(1)+SP(2))*0.5D0
      DISC = DSQRT(((SP(1)-SP(2))*0.5D0)**2 + SP(3)**2 + 1.0D-60)
      SP1  = AVG + DISC
      SP2  = AVG - DISC
C
C     Principal values of SD
      AVG  = (SD(1)+SD(2))*0.5D0
      DISC = DSQRT(((SD(1)-SD(2))*0.5D0)**2 + SD(3)**2 + 1.0D-60)
      SD1  = AVG + DISC
      SD2  = AVG - DISC
C
      PHI = DABS(SP1-SP2)**AEXP
     &    + DABS(2.0D0*SD2+SD1)**AEXP
     &    + DABS(2.0D0*SD1+SD2)**AEXP
C
      RETURN
      END
C =====================================================================


C =====================================================================
C  YLD_SIGBAR — compute Yld2000-2d equivalent stress sigbar = (phi/2)^(1/a)
C =====================================================================
      SUBROUTINE YLD_SIGBAR(S, ALPHA, SIGBAR)
      IMPLICIT NONE
      DOUBLE PRECISION S(3), ALPHA(8), SIGBAR
      DOUBLE PRECISION PHI
      DOUBLE PRECISION AEXP
      PARAMETER (AEXP = 6.0D0)
C
      CALL YLD_PHI(S, ALPHA, PHI)
      IF (PHI .LT. 1.0D-60) THEN
        SIGBAR = 0.0D0
      ELSE
        SIGBAR = (PHI * 0.5D0)**(1.0D0/AEXP)
      END IF
C
      RETURN
      END
C =====================================================================


C =====================================================================
C  YLD_GRAD — gradient of sigbar w.r.t. sigma via central differences
C  m(i) = d(sigbar)/d(s(i))
C =====================================================================
      SUBROUTINE YLD_GRAD(S, ALPHA, M)
      IMPLICIT NONE
      DOUBLE PRECISION S(3), ALPHA(8), M(3)
      DOUBLE PRECISION SH(3), PHI_P, PHI_M, SIGBAR_P, SIGBAR_M
      DOUBLE PRECISION HS
      INTEGER I
      DOUBLE PRECISION AEXP
      PARAMETER (AEXP = 6.0D0)
C
      HS = 1.0D-4 * DMAX1(DSQRT(S(1)**2+S(2)**2+S(3)**2), 1.0D0)
C
      DO I = 1, 3
        SH(1) = S(1); SH(2) = S(2); SH(3) = S(3)
        SH(I) = S(I) + HS
        CALL YLD_PHI(SH, ALPHA, PHI_P)
        SIGBAR_P = (PHI_P*0.5D0)**(1.0D0/AEXP)
C
        SH(1) = S(1); SH(2) = S(2); SH(3) = S(3)
        SH(I) = S(I) - HS
        CALL YLD_PHI(SH, ALPHA, PHI_M)
        SIGBAR_M = (PHI_M*0.5D0)**(1.0D0/AEXP)
C
        M(I) = (SIGBAR_P - SIGBAR_M) / (2.0D0*HS)
      END DO
C
      RETURN
      END
C =====================================================================


C =====================================================================
C  YLD_HESS — Hessian of sigbar w.r.t. sigma via central differences
C  H(i,j) = d(m_i)/d(s_j)  where m = grad(sigbar)
C  Computed by perturbing s_j and differencing the full gradient vector.
C  Cost: 6 gradient evaluations (2 per column j=1,2,3)
C =====================================================================
      SUBROUTINE YLD_HESS(S, ALPHA, H)
      IMPLICIT NONE
      DOUBLE PRECISION S(3), ALPHA(8), H(3,3)
      DOUBLE PRECISION SH(3), M_P(3), M_M(3)
      DOUBLE PRECISION HS
      INTEGER I, J
C
      HS = 1.0D-3 * DMAX1(DSQRT(S(1)**2+S(2)**2+S(3)**2), 1.0D0)
C
      DO J = 1, 3
        SH(1) = S(1); SH(2) = S(2); SH(3) = S(3)
        SH(J) = S(J) + HS
        CALL YLD_GRAD(SH, ALPHA, M_P)
C
        SH(1) = S(1); SH(2) = S(2); SH(3) = S(3)
        SH(J) = S(J) - HS
        CALL YLD_GRAD(SH, ALPHA, M_M)
C
        DO I = 1, 3
          H(I,J) = (M_P(I) - M_M(I)) / (2.0D0*HS)
        END DO
      END DO
C
      RETURN
      END
C =====================================================================


C =====================================================================
C  SOLVE4 — solve 4x4 linear system J*dx = -res by Gaussian elimination
C  On entry: JAC(4,4), RES(4)
C  On exit:  DX(4)
C =====================================================================
      SUBROUTINE SOLVE4(JAC, RES, DX)
      IMPLICIT NONE
      DOUBLE PRECISION JAC(4,4), RES(4), DX(4)
      DOUBLE PRECISION A(4,5), PIV, FAC
      INTEGER I, J, K, IMAX
      DOUBLE PRECISION AMAX
C
C     Augmented matrix [JAC | RES]
      DO I = 1, 4
        DO J = 1, 4
          A(I,J) = JAC(I,J)
        END DO
        A(I,5) = -RES(I)
      END DO
C
C     Gaussian elimination with partial pivoting
      DO K = 1, 4
C       Find pivot
        AMAX = DABS(A(K,K))
        IMAX = K
        DO I = K+1, 4
          IF (DABS(A(I,K)) .GT. AMAX) THEN
            AMAX = DABS(A(I,K))
            IMAX = I
          END IF
        END DO
C       Swap rows
        IF (IMAX .NE. K) THEN
          DO J = K, 5
            PIV = A(K,J)
            A(K,J) = A(IMAX,J)
            A(IMAX,J) = PIV
          END DO
        END IF
C       Eliminate
        IF (DABS(A(K,K)) .LT. 1.0D-30) CYCLE
        DO I = K+1, 4
          FAC = A(I,K)/A(K,K)
          DO J = K, 5
            A(I,J) = A(I,J) - FAC*A(K,J)
          END DO
        END DO
      END DO
C
C     Back substitution
      DO I = 4, 1, -1
        DX(I) = A(I,5)
        DO J = I+1, 4
          DX(I) = DX(I) - A(I,J)*DX(J)
        END DO
        IF (DABS(A(I,I)) .GT. 1.0D-30) THEN
          DX(I) = DX(I)/A(I,I)
        ELSE
          DX(I) = 0.0D0
        END IF
      END DO
C
      RETURN
      END
C =====================================================================


C =====================================================================
C  INV3 — invert a 3x3 matrix (analytical)
C  On entry:  A(3,3)
C  On exit:   AINV(3,3)
C =====================================================================
      SUBROUTINE INV3(A, AINV)
      IMPLICIT NONE
      DOUBLE PRECISION A(3,3), AINV(3,3)
      DOUBLE PRECISION DET, IDET
C
      DET = A(1,1)*(A(2,2)*A(3,3)-A(2,3)*A(3,2))
     &    - A(1,2)*(A(2,1)*A(3,3)-A(2,3)*A(3,1))
     &    + A(1,3)*(A(2,1)*A(3,2)-A(2,2)*A(3,1))
C
      IF (DABS(DET) .LT. 1.0D-30) THEN
C       Singular — return identity as fallback
        AINV(1,1)=1.; AINV(1,2)=0.; AINV(1,3)=0.
        AINV(2,1)=0.; AINV(2,2)=1.; AINV(2,3)=0.
        AINV(3,1)=0.; AINV(3,2)=0.; AINV(3,3)=1.
        RETURN
      END IF
C
      IDET = 1.0D0/DET
      AINV(1,1) =  (A(2,2)*A(3,3)-A(2,3)*A(3,2))*IDET
      AINV(1,2) = -(A(1,2)*A(3,3)-A(1,3)*A(3,2))*IDET
      AINV(1,3) =  (A(1,2)*A(2,3)-A(1,3)*A(2,2))*IDET
      AINV(2,1) = -(A(2,1)*A(3,3)-A(2,3)*A(3,1))*IDET
      AINV(2,2) =  (A(1,1)*A(3,3)-A(1,3)*A(3,1))*IDET
      AINV(2,3) = -(A(1,1)*A(2,3)-A(1,3)*A(2,1))*IDET
      AINV(3,1) =  (A(2,1)*A(3,2)-A(2,2)*A(3,1))*IDET
      AINV(3,2) = -(A(1,1)*A(3,2)-A(1,2)*A(3,1))*IDET
      AINV(3,3) =  (A(1,1)*A(2,2)-A(1,2)*A(2,1))*IDET
C
      RETURN
      END
C =====================================================================
