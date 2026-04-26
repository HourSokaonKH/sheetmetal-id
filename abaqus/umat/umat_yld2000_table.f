C =====================================================================
C  UMAT:  Barlat Yld2000-2d + Tabulated isotropic hardening
C  For Abaqus/Standard plane stress elements (CPS4R)
C
C  Material: SGCC JIS G 3302 galvanized steel (BCC, a=6)
C
C  This variant replaces the hard-coded Voce law of umat_yld2000.f with a
C  user-supplied tabulated flow curve sigma_y(kappa).  The Python driver
C  (optimize_hardening_multidir.py, MODEL_TYPE='yld2000_table') bakes any
C  uniaxial hardening analytical form (e.g. Voce + Chaboche(2) monotonic
C  equivalent) into the table so the FEA cost function actually depends
C  on ALL identified parameters, not just sigma_0.
C
C  NTENS=3, NDI=2, NSHR=1  (plane stress)
C
C  PROPS layout (NPROPS = 11 + 2*NTAB; typically NTAB=50 -> NPROPS=111):
C    PROPS(1)              = E         [MPa]
C    PROPS(2)              = NU
C    PROPS(3..10)          = ALPHA(1..8)     Yld2000-2d coefficients
C    PROPS(11)             = NTAB      (real-valued integer)
C    PROPS(12..11+NTAB)    = KAPPA_i   sorted ascending, KAPPA(1) = 0
C    PROPS(12+NTAB..11+2*NTAB)
C                          = SIGY_i    flow stress at KAPPA_i
C    (exponent a=6 for BCC steel is hardcoded)
C
C  STATEV layout (NSTATV=1):
C    STATEV(1) = cumulative plastic strain kappa
C
C  Return mapping:
C    Cutting-plane scheme identical to the Voce variant; the only change
C    is TABLOOK(KAPPA) replacing SIG0 + QINF*(1-exp(-BISO*kappa)).
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
      DOUBLE PRECISION E, NU
      DOUBLE PRECISION ALPHA(8)
      INTEGER NTAB, ITAB0_K, ITAB0_S
      DOUBLE PRECISION C11, C12, C66
      DOUBLE PRECISION CPS(3,3)
      DOUBLE PRECISION KAPPA0
      DOUBLE PRECISION STR_TR(3), STR(3)
      DOUBLE PRECISION SB_TR, SIGY0
      DOUBLE PRECISION DL
      DOUBLE PRECISION M(3)
      DOUBLE PRECISION SB, SIGY, DSIGY
      DOUBLE PRECISION R1T, R2T, R3T, FNORM, FNT, STEP
      DOUBLE PRECISION TOL_NR
      INTEGER ITER, MAXITER, I, J
      LOGICAL ELASTIC
C =====================================================================

C ---- Read material properties ---------------------------------------
      E     = PROPS(1)
      NU    = PROPS(2)
      DO I = 1, 8
        ALPHA(I) = PROPS(2+I)
      END DO
      NTAB = NINT(PROPS(11))
      IF (NTAB .LT. 2) NTAB = 2
      ITAB0_K = 11            ! PROPS(12) is first KAPPA_i  -> PROPS(ITAB0_K+i)
      ITAB0_S = 11 + NTAB     ! PROPS(12+NTAB) is first SIGY_i -> PROPS(ITAB0_S+i)
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
      CALL TABLOOK(PROPS, NPROPS, NTAB, ITAB0_K, ITAB0_S,
     &             KAPPA0, SIGY0, DSIGY)
C
      ELASTIC = (SB_TR .LE. SIGY0*(1.0D0 + 1.0D-10))
C
      IF (ELASTIC) THEN
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
        CALL TABLOOK(PROPS, NPROPS, NTAB, ITAB0_K, ITAB0_S,
     &               KAPPA0+DL, SIGY, DSIGY)
C
        FNORM = (SB - SIGY) / DMAX1(SIGY, 1.0D0)
        IF (DABS(FNORM) .LT. TOL_NR) GOTO 200
C
        R1T = C11*M(1) + C12*M(2)
        R2T = C12*M(1) + C11*M(2)
        R3T = C66*M(3)
C
        FNT = M(1)*R1T + M(2)*R2T + M(3)*R3T + DSIGY
        IF (FNT .LT. 1.0D-8) FNT = 1.0D-8
C
        STEP = (SB - SIGY) / FNT
        IF (STEP .GT.  1.0D-2) STEP =  1.0D-2
        IF (STEP .LT. -1.0D-2) STEP = -1.0D-2
C
        STR(1) = STR(1) - STEP*R1T
        STR(2) = STR(2) - STEP*R2T
        STR(3) = STR(3) - STEP*R3T
        DL     = DL     + STEP
        IF (DL .LT. 0.0D0) DL = 0.0D0
      END DO
 200  CONTINUE
C
C ---- Update state --------------------------------------------------
      DO I = 1, NTENS
        STRESS(I) = STR(I)
      END DO
      STATEV(1) = KAPPA0 + DL
C
C ---- Tangent modulus (elastic — see umat_yld2000.f note) -----------
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
C  TABLOOK — piecewise-linear interpolation of tabulated flow curve
C  Input : PROPS(NPROPS), NTAB, ITAB0_K, ITAB0_S, KAPPA
C  Output: SIGY     = sigma_y(kappa)
C          DSIGDK   = d(sigma_y)/d(kappa) (local slope)
C
C  KAPPA < KAPPA(1)  : SIGY = SIGY(1), DSIGDK = 0  (elastic)
C  KAPPA > KAPPA(N)  : linear extrapolation of last segment
C                     (the driver should tabulate out beyond any strain
C                      ever seen in the FE model, so this branch is a
C                      safety net — clamp with positive slope so the
C                      plastic return keeps moving the right direction).
C =====================================================================
      SUBROUTINE TABLOOK(PROPS, NPROPS, NTAB, ITAB0_K, ITAB0_S,
     &                   KAPPA, SIGY, DSIGDK)
      IMPLICIT NONE
      INTEGER NPROPS, NTAB, ITAB0_K, ITAB0_S
      DOUBLE PRECISION PROPS(NPROPS), KAPPA, SIGY, DSIGDK
      DOUBLE PRECISION K1, K2, S1, S2
      INTEGER I, IBOT, ITOP, IMID
C
      IF (KAPPA .LE. PROPS(ITAB0_K+1)) THEN
        SIGY   = PROPS(ITAB0_S+1)
        DSIGDK = 0.0D0
        RETURN
      END IF
      IF (KAPPA .GE. PROPS(ITAB0_K+NTAB)) THEN
        K1 = PROPS(ITAB0_K+NTAB-1)
        K2 = PROPS(ITAB0_K+NTAB  )
        S1 = PROPS(ITAB0_S+NTAB-1)
        S2 = PROPS(ITAB0_S+NTAB  )
        IF (K2 .GT. K1) THEN
          DSIGDK = (S2 - S1) / (K2 - K1)
        ELSE
          DSIGDK = 0.0D0
        END IF
        SIGY = S2 + DSIGDK * (KAPPA - K2)
        RETURN
      END IF
C
C     Binary search for interval
      IBOT = 1
      ITOP = NTAB
 100  CONTINUE
      IF (ITOP - IBOT .GT. 1) THEN
        IMID = (IBOT + ITOP) / 2
        IF (PROPS(ITAB0_K+IMID) .GT. KAPPA) THEN
          ITOP = IMID
        ELSE
          IBOT = IMID
        END IF
        GOTO 100
      END IF
C
      K1 = PROPS(ITAB0_K+IBOT)
      K2 = PROPS(ITAB0_K+ITOP)
      S1 = PROPS(ITAB0_S+IBOT)
      S2 = PROPS(ITAB0_S+ITOP)
      IF (K2 .GT. K1) THEN
        DSIGDK = (S2 - S1) / (K2 - K1)
      ELSE
        DSIGDK = 0.0D0
      END IF
      SIGY = S1 + DSIGDK * (KAPPA - K1)
C
      RETURN
      END
C =====================================================================


C =====================================================================
C  YLD_PHI — compute Yld2000-2d yield function phi
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
      SP(1) = (2.0D0*A1/3.0D0)*S(1) + (-A1/3.0D0)*S(2)
      SP(2) = (-A2/3.0D0)*S(1) + (2.0D0*A2/3.0D0)*S(2)
      SP(3) = A7*S(3)
C
      SD(1) = ((8.0D0*A5-2.0D0*A3-2.0D0*A6+2.0D0*A4)/9.0D0)*S(1)
     &       +((4.0D0*A6-4.0D0*A5-4.0D0*A4+A3)/9.0D0)*S(2)
      SD(2) = ((4.0D0*A3-4.0D0*A5-4.0D0*A4+A6)/9.0D0)*S(1)
     &       +((8.0D0*A4-2.0D0*A6-2.0D0*A3+2.0D0*A5)/9.0D0)*S(2)
      SD(3) = A8*S(3)
C
      AVG  = (SP(1)+SP(2))*0.5D0
      DISC = DSQRT(((SP(1)-SP(2))*0.5D0)**2 + SP(3)**2 + 1.0D-60)
      SP1  = AVG + DISC
      SP2  = AVG - DISC
C
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
C  YLD_SIGBAR — compute equivalent stress sigbar = (phi/2)^(1/a)
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
C  YLD_GRAD — d(sigbar)/d(sigma) via central differences
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
